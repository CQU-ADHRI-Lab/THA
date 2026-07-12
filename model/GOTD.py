import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torch_intermediate_layer_getter import IntermediateLayerGetter

from libs.detr.models import build_model as build_detr
from src import utils
from src.models.components.GazeTransformer import (
    GazeTransformerLayer,
    GazeTransformer,
) 
from src.models.components.MLP import MLP
from src.utils.AttributeDict import AttributeDict
from src.utils.gaze_ops import get_gaze_cone
from src.utils.misc import load_pretrained, get_annotation_id
from src.models.components.ResNet import resnet18

from transformers import AutoTokenizer, CLIPTextModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = utils.get_pylogger(__name__)


class GOTD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int,
        num_gaze_queries: int,
        gaze_heatmap_size: int,
        num_gaze_decoder_layers: int
    ):
        super().__init__()

        self.dim_feedforward = 2048
        self.hidden_dim = 256
        self.nhead = nhead = 8
        self.dropout = dropout = 0.1

        # Setup backbone
        self.backbone, _, _ = build_detr(
            AttributeDict(
                {
                    "dataset_file": "coco",
                    "device": "cuda",  # TODO: override this using hydra config
                    "num_queries": num_queries,
                    "aux_loss": False,
                    "masks": False,
                    "eos_coef": 0,
                    "hidden_dim": self.hidden_dim,
                    "position_embedding": "sine",
                    "lr_backbone": 1e-5,
                    "backbone": "resnet50",
                    "dilation": False,
                    "nheads": nhead,
                    "dim_feedforward": self.dim_feedforward,
                    "enc_layers": 6,
                    "dec_layers": 6,
                    "dropout": dropout,
                    "pre_norm": False,
                    "set_cost_class": 1,
                    "set_cost_bbox": 5,
                    "set_cost_giou": 2,
                    "mask_loss_coef": 1,
                    "bbox_loss_coef": 5,
                    "giou_loss_coef": 2,
                }
            )
        )
        log.info("Loading DETR pretrained weights")
        load_pretrained(
            self.backbone,
            model_zoo.load_url(
                "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
            ),
        )
        # Extract features from the object detection backbone    获取模型中间层的输出
        self.backbone_getter = IntermediateLayerGetter(
            self.backbone, return_layers={"transformer": "hs"}
        )

        # Setup object detector MLPs
        self.class_embed = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_classes + 1,
            num_layers=1,
        )
        self.bbox_embed = self.head_bbox_embed = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=4,
            num_layers=3,
        )
        # end of object detection

        self.resnet_4channel = resnet18(pretrained=True, channel=4)

        # self.resnet_3channel = resnet18(pretrained=True, channel=3)
        self.ave_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        # CLIP
        CLIP_name = "/home/jly/object-aware-gaze-target-detection-main-2025-TIA/clip-vit-base-patch32"
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_name)
        self.text_model = CLIPTextModel.from_pretrained(CLIP_name)  
        # Setup gaze query embeddings  将离散类别索引映射为密集向量

        self.task_embedding = nn.Sequential(nn.Linear(512, 256),
                                            nn.ReLU(True),
                                            nn.Dropout())

        self.human_embedding = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(True),
                                             nn.Dropout())

        self.task_encoder = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.nhead, dropout=dropout, batch_first=True)

        self.num_gaze_queries = num_gaze_queries
        self.gaze_embed = self.gaze_query_embed = nn.Embedding(
            self.num_gaze_queries, self.hidden_dim
        )

        # Setup gaze transformer
        self.gaze_transformer = GazeTransformer(
            GazeTransformerLayer(
                self.hidden_dim,
                nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                activation="relu",
            ),
            num_gaze_decoder_layers,
            norm=nn.LayerNorm(self.hidden_dim),
        )

        self.gaze_heatmap_obj_embed = MLP(
            self.hidden_dim,
            self.hidden_dim,
            gaze_heatmap_size**2,
            num_layers=5
        )

        self.gaze_watch_outside_embed = MLP(self.hidden_dim, self.hidden_dim, output_dim=1, num_layers=1)

    def forward(self, samples, img_sizes, data):
        backbone_intermediate_layers, _ = self.backbone_getter(samples)
        object_detection_decoder_features = backbone_intermediate_layers["hs"][0]  # 获取中间层的输出
        # repeat:重复嵌入向量，将一个嵌入矩阵扩展为一个更大的张量    query_embed层的权重，是一个矩阵   repeat的参数表示每个维度上重复的次数
        object_detection_decoder_embed = self.backbone.query_embed.weight.repeat(    
            object_detection_decoder_features.shape[0],    # batch_size的维度
            object_detection_decoder_features.shape[1],    # num_queries的维度
            1,
            1,
        )

        outputs_logits = self.class_embed(object_detection_decoder_features)     # 分类
        outputs_bbox = self.bbox_embed(object_detection_decoder_features).sigmoid()   # 定位
        outputs_labels = outputs_logits.argmax(-1)   # 找出最大的一个作为标签

        sort_idx = outputs_labels.argsort(dim=-1, descending=False)  # 对预测类别进行排序，返回排序后的索引  升序
        outputs_logits = outputs_logits.gather(   # 使logits按照预测标签的顺序排序   unsqueeze(-1):在sort_idx的最后一个维度添加一个新的维度  expand:将张量的形状扩展为指定的形状
            2,    # gather用于根据索引从指定维度上收集数据
            sort_idx.unsqueeze(-1).expand(-1, -1, -1, outputs_logits.shape[-1]),
        )
        outputs_bbox = outputs_bbox.gather(
            2, sort_idx.unsqueeze(-1).expand(-1, -1, -1, outputs_bbox.shape[-1])
        )
        object_detection_decoder_features = object_detection_decoder_features.gather(
            2,
            sort_idx.unsqueeze(-1).expand(
                -1, -1, -1, object_detection_decoder_features.shape[-1]
            ),
        )
        object_detection_decoder_embed = object_detection_decoder_embed.gather(
            2,
            sort_idx.unsqueeze(-1).expand(
                -1, -1, -1, object_detection_decoder_embed.shape[-1]
            ),
        )

        # There are 6 layers in the decoder, we only want the last one
        object_detection_decoder_features = object_detection_decoder_features[-1:]
        object_detection_decoder_embed = object_detection_decoder_embed[-1:]
        outputs_bbox = outputs_bbox[-1:]
        outputs_logits = outputs_logits[-1:]
        outputs_labels = outputs_labels[-1:]

        # Keep only the first max_objects objects  保存前‘num_gaze_queries’个对象的解码器输出特征、嵌入特征、边界框和分类
        object_detection_decoder_features = object_detection_decoder_features[
            :, :, : self.num_gaze_queries
        ][-1]  # [1, 4, 20, 256]
        object_detection_decoder_embed = object_detection_decoder_embed[
            :, :, : self.num_gaze_queries
        ][-1]
        outputs_bbox = outputs_bbox[:, :, : self.num_gaze_queries]
        outputs_logits = outputs_logits[:, :, : self.num_gaze_queries]
        outputs_labels = outputs_labels[:, :, : self.num_gaze_queries]
        # the end of object detection

        face_presence = torch.logical_and(
            outputs_logits.argmax(dim=-1) == get_annotation_id("face"),
            outputs_logits.max(dim=-1).values > 0.5,
        )  # [1,4,20]

        objects_presence = torch.logical_and(
            outputs_logits.argmax(dim=-1) != get_annotation_id("no-object"),
            outputs_logits.max(dim=-1).values > 0.5
        )

        # human-environment feature
        image = data['image'].tensors
        human_skeleton_mask = data['human_skeleton_mask'].tensors
        human_bodypart_mask = data['human_bodypart_mask'].tensors

        image_skeleton = torch.cat([image, human_skeleton_mask], dim=1)
        image_bodypart = torch.cat([image, human_bodypart_mask], dim=1)

        fm_in1 = self.resnet_4channel.fm_extract(image_skeleton)
        fm_in2 = self.resnet_4channel.fm_extract(image_bodypart)

        fm_human = fm_in1 + fm_in2
        # task_text = ["arranging objects", "cleaning objects", "having meal", "making cereal", "microwaving food", "picking objects", "stacking objects", "taking medicine", "unstacking objects"]
        task_text = ["arranging objects", "cleaning objects", "making cereal", "microwaving food", "picking objects", "stacking objects", "taking food","taking medicine", "unstacking objects"]
        task_inputs = self.tokenizer(task_text, return_tensors="pt", padding=True)
        task_inputs = {key: value.cuda() for key, value in task_inputs.items()}
        task_embedding = self.text_model(**task_inputs).pooler_output  # [task_num, 512]

        fm_pool = self.ave_pool(fm_human).view(-1, 512)   # [bs, 512]
        task_embeds_pool = torch.matmul(fm_pool, task_embedding.t())

        task_predict = task_embeds_pool.softmax(dim=-1)
        task_max = torch.argmax(task_predict, dim=-1)

        task_max_embed = task_embedding[task_max, :]
        task_max_embed = self.task_embedding(task_max_embed).unsqueeze(1)  # [bs, 1, 256]

        combined_query = torch.cat([task_max_embed, object_detection_decoder_features], dim=1)
        object_detection_decoder_features_with_task = self.task_encoder(combined_query, combined_query, combined_query)[0][:,1:,:]  # [bs, num_query, 256]

        object_detection_decoder_features = object_detection_decoder_features.permute(1,0,2) # [20,4,512]
        object_detection_decoder_embed = object_detection_decoder_embed.permute(1,0,2)  # [20,4,512]

        num_queries, bs, feat_dim = object_detection_decoder_features.shape

        gaze_query_embed = self.gaze_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        gaze_decoder_tgt = torch.zeros_like(gaze_query_embed)

        fm_in1 = self.human_embedding(self.ave_pool(fm_in1).view(-1, 512)).unsqueeze(0)  # [bs, 256]
        fm_in2 = self.human_embedding(self.ave_pool(fm_in2).view(-1, 512)).unsqueeze(0)  # [bs, 256]
        # fm_human = self.human_embedding(self.ave_pool(fm_human).view(-1, 512)).unsqueeze(0)  
        # fm_human = fm_human.repeat(num_queries,1,1)
        fm_in1 = fm_in1.repeat(num_queries, 1, 1)
        fm_in2 = fm_in2.repeat(num_queries, 1, 1)

        objects_hs_1 = self.gaze_transformer(
            sa = gaze_decoder_tgt,
            sa_v = fm_in1,
            ca = object_detection_decoder_features_with_task.permute(1,0,2),
            sa_mask = face_presence.permute(2, 1, 0),
            ca_mask = objects_presence.permute(2, 1, 0),
            sa_pos = gaze_query_embed,
            ca_pos = object_detection_decoder_embed,
            attn_booster = None
        )

        objects_hs_2 = self.gaze_transformer(
            sa = gaze_decoder_tgt,
            sa_v = fm_in2,
            ca = object_detection_decoder_features_with_task.permute(1,0,2),
            sa_mask = face_presence.permute(2, 1, 0),
            ca_mask = objects_presence.permute(2, 1, 0),
            sa_pos = gaze_query_embed,
            ca_pos = object_detection_decoder_embed,
            attn_booster = None
        )

        objects_hs = objects_hs_1[-1:] + objects_hs_2[-1:] + object_detection_decoder_features
        # objects_hs = objects_hs_1[-1:] + object_detection_decoder_features
        # objects_hs = object_detection_decoder_features.unsqueeze(0)
        objects_hs = objects_hs.transpose(1,2)

        outputs_gaze_heatmap = self.gaze_heatmap_obj_embed(objects_hs)

        outputs_watch_outside = self.gaze_watch_outside_embed(objects_hs).sigmoid()

        out = {
            "pred_logits": outputs_logits[-1],
            "pred_boxes": outputs_bbox[-1],
            "pred_gaze_heatmap": outputs_gaze_heatmap[-1],
            "pred_task": task_predict,
            "pred_gaze_watch_outside": outputs_watch_outside[-1] 
        }

        return out
