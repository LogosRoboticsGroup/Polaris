from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block, InteractionBlock
from .layers.wo_time_decoder import WOTimeDecoder
from .layers.relative_transformer import SymmetricFusionTransformer
from .layers.mamba.vim_mamba import init_weights, create_block
from src.datamodule.av2_dataset import polar_to_cartesian, normalize_angle, cartesian_to_polar
from functools import partial
from timm.models.layers import DropPath, to_2tuple
from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        drop_path=0.2,
        future_steps: int = 60,
    ) -> None:
        super().__init__()

        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )
        self.hist_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=False,  
                    rms_norm=True,  
                )
                for i in range(3)
            ]
        )
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path = DropPath(drop_path)

        self.lane_embed = LaneEmbeddingLayer(4, embed_dim)
        self.lane_embed_relative = LaneEmbeddingLayer(4, embed_dim)
        self.lane_cat = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.pos_embed_relative = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.blocks_relative = SymmetricFusionTransformer(n_layer=3)
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))

        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), 
            nn.GELU(), 
            nn.Linear(256, future_steps * 2),
        )

        self.wo_time_decoder = WOTimeDecoder()

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        ### Polar scene context encoding ###
        hist_valid_mask = data["x_valid_mask"]
        hist_key_valid_mask = data["x_key_valid_mask"]
        hist_feat = torch.cat(
            [
                data['x_positions'][..., 0].unsqueeze(-1),
                torch.stack([torch.cos(data['x_positions'][..., 1]), torch.sin(data['x_positions'][..., 1])], dim=-1),
                data['x_velocity_vector'][..., 0].unsqueeze(-1),
                torch.stack([torch.cos(data['x_velocity_vector'][..., 1]), torch.sin(data['x_velocity_vector'][..., 1])], dim=-1),
                data['x_accelerate_vector'][..., 0].unsqueeze(-1),
                torch.stack([torch.cos(data['x_accelerate_vector'][..., 1]), torch.sin(data['x_accelerate_vector'][..., 1])], dim=-1),
                hist_valid_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)

        # Agent encoding
        actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous())
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        actor_feat = fused_add_norm_fn(
            self.drop_path(actor_feat),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )

        actor_feat = actor_feat[:, -1]
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        # Lane encoding
        lane_valid_mask = data["lane_valid_mask"]
        lane_normalized = data["lane_positions"]
        lane_normalized = torch.cat(
            [
                lane_normalized[..., 0].unsqueeze(-1), 
                torch.stack([torch.cos(lane_normalized[..., 1]), torch.sin(lane_normalized[..., 1])], dim=-1),
                lane_valid_mask[..., None],
            ], 
            dim=-1,
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        # Lane change encoding
        lane_relative = data["lane_positions"]
        lane_relative_r = lane_relative[..., 0][..., 1:] - lane_relative[..., 0][..., :-1]
        lane_relative_theta = lane_relative[..., 1][..., 1:] - lane_relative[..., 1][..., :-1]
        lane_valid_mask_relative = lane_valid_mask[..., 1:] & lane_valid_mask[..., :-1]
        lane_relative = torch.cat(
            [
                lane_relative_r.unsqueeze(-1), 
                torch.stack([torch.cos(lane_relative_theta), torch.sin(lane_relative_theta)], dim=-1),
                lane_valid_mask_relative[..., None],
            ], 
            dim=-1,
        )
        lane_feat_relative = self.lane_embed_relative(lane_relative.view(-1, L-1, D).contiguous())
        lane_feat_relative = lane_feat_relative.view(B, M, -1)

        lane_feat = torch.cat((lane_feat, lane_feat_relative), dim=-1)
        lane_feat = self.lane_cat(lane_feat)

        x_centers = torch.cat([data["x_centers"][..., 0], data["lane_centers"][..., 0]], dim=1)
        x_theta = torch.stack([torch.cos(data["x_centers"][..., 1]), torch.sin(data["x_centers"][..., 1])], dim=-1)
        lane_theta = torch.stack([torch.cos(data["lane_centers"][..., 1]), torch.sin(data["lane_centers"][..., 1])], dim=-1)
        x_centers_theta = torch.cat([x_theta, lane_theta], dim=1)
        x_centers = torch.cat([x_centers.unsqueeze(-1), x_centers_theta], dim=-1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed[data["lane_attr"][..., 0].long()]
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask = torch.cat(
            [data["x_key_valid_mask"], data["lane_key_valid_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed

        if isinstance(self, StreamModelForecast) and self.use_stream_encoder:
            if "memory_dict" in data and data["memory_dict"] is not None:
                rel_pos = data["origin"] - data["memory_dict"]["origin"]
                rel_pos = cartesian_to_polar(rel_pos)
                rel_ang = (data["theta"] - data["memory_dict"]["theta"] + torch.pi) % (2 * torch.pi) - torch.pi
                rel_ts = data["timestamp"] - data["memory_dict"]["timestamp"]
                memory_pose = torch.cat([
                    rel_ts.unsqueeze(-1), rel_ang.unsqueeze(-1), rel_pos
                ], dim=-1).float().to(x_encoder.device)
                memory_x_encoder = data["memory_dict"]["x_encoder"]
                memory_valid_mask = data["memory_dict"]["x_mask"]
            else:
                memory_pose = x_encoder.new_zeros(x_encoder.size(0), self.pose_dim)
                memory_x_encoder = x_encoder
                memory_valid_mask = key_valid_mask
            cur_pose = torch.zeros_like(memory_pose)

        if isinstance(self, StreamModelForecast) and self.use_stream_encoder:
            new_x_encoder = x_encoder
            for inter in self.interaction:
                new_x_encoder = inter(new_x_encoder, memory_x_encoder, cur_pose, 
                                      memory_pose, key_padding_mask=~memory_valid_mask)
            x_encoder = new_x_encoder * key_valid_mask.unsqueeze(-1) + x_encoder * ~key_valid_mask.unsqueeze(-1)

        # Relative Embedding Transformer
        relative_embed = self.pos_embed_relative(data['relative_embed'])
        relative_key_valid_mask = data['relative_key_valid_mask']
        x_encoder = self.blocks_relative(x_encoder, relative_embed, ~relative_key_valid_mask)
        x_encoder = self.norm(x_encoder)

        y_hat_new = None
        y_hat_new_polar = None
        pi_new = None
        y_hat_new_new = None 
        y_hat_new_new_polar = None
        pi_new_new = None

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)

        ### Decoding & Polar relationship refinement ###
        y_hat, pi, y_hat_new, pi_new, y_hat_new_new, pi_new_new \
            = self.wo_time_decoder(x_encoder, mask=~key_valid_mask, data=data)

        y_hat_polar = y_hat
        y_hat = polar_to_cartesian(y_hat)
        y_hat_others_polar = y_hat_others
        y_hat_others = polar_to_cartesian(y_hat_others)
        if y_hat_new is not None:
            y_hat_new_polar = y_hat_new
            y_hat_new = polar_to_cartesian(y_hat_new)
        if y_hat_new_new is not None:
            y_hat_new_new_polar = y_hat_new_new
            y_hat_new_new = polar_to_cartesian(y_hat_new_new)

        ret_dict = {
            "y_hat": y_hat,
            "y_hat_polar": y_hat_polar,
            "pi": pi,

            "y_hat_others": y_hat_others,
            "y_hat_others_polar": y_hat_others_polar,

            "y_hat_new": y_hat_new,
            "y_hat_new_polar": y_hat_new_polar,
            "pi_new": pi_new,
            
            "y_hat_new_new": y_hat_new_new,
            "y_hat_new_new_polar": y_hat_new_new_polar,
            "pi_new_new": pi_new_new,
        }

        if isinstance(self, StreamModelForecast):
            memory_dict = {
                "x_encoder": x_encoder,
                "x_mask": key_valid_mask,
                "origin": data["origin"],
                "theta": data["theta"],
                "timestamp": data["timestamp"],
            }
            ret_dict["memory_dict"] = memory_dict

        return ret_dict


class StreamModelForecast(ModelForecast):
    def __init__(self, 
                 use_stream_encoder=True,
                 use_stream_decoder=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_stream_encoder = use_stream_encoder
        self.use_stream_decoder = use_stream_decoder
        self.embed_dim = kwargs["embed_dim"]
        self.pose_dim = 4
        if self.use_stream_encoder:
            self.interaction = nn.ModuleList(
                InteractionBlock(
                    dim=kwargs["embed_dim"],
                    pose_dim=self.pose_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    drop_path=0.2,
                )
                for i in range(1)
            )
