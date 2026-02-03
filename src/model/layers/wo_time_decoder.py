import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
from .relative_transformer_qk import SymmetricFusionTransformer


class GMMPredictor(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian_r = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len),
        )
        self.gaussian_theta = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len),
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
    
    def forward(self, input):
        B, M, _ = input.shape
        res = torch.zeros((B, M, self._future_len, 2), dtype=input.dtype, device=input.device)
        res_r = self.gaussian_r(input)
        res_theta = self.gaussian_theta(input)
        res[..., 0] = res_r
        res[..., 1] = res_theta
        score = self.score(input).squeeze(-1)

        return res, score
    

class WOTimeDecoder(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(WOTimeDecoder, self).__init__()

        ### initial ###
        self.multi_modal_query_embedding = nn.Embedding(6, dim)
        self.register_buffer('modal', torch.arange(6).long())

        self.cross_block_mode = nn.ModuleList(
            Cross_Block()
            for i in range(2)
        )
        self.self_block_mode = nn.ModuleList(
            Block()
            for i in range(2)
        )

        self.predictor = GMMPredictor(future_len)

        ### refine ###
        self.re_input = nn.Sequential(
            nn.Linear(360, 256), 
            nn.GELU(), 
            nn.Linear(256, dim),
        )

        self.pos_embed_relative_refine = nn.Sequential(
            nn.Linear(5, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.blocks_relative_refine = SymmetricFusionTransformer(n_layer=2)
        self.self_block_mode_refine = nn.ModuleList(
            Block()
            for i in range(2)
        )

        self.re_output_r = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, future_len),
        )

        self.re_output_theta = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, future_len),
        )

        self.re_output_pi = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )

        ### refine again ###
        self.re_input_new = nn.Sequential(
            nn.Linear(360, 256), 
            nn.GELU(), 
            nn.Linear(256, dim),
        )

        self.pos_embed_relative_refine_new = nn.Sequential(
            nn.Linear(5, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.blocks_relative_refine_new = SymmetricFusionTransformer(n_layer=2)
        self.self_block_mode_refine_new = nn.ModuleList(
            Block()
            for i in range(2)
        )

        self.re_output_r_new = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, future_len),
        )

        self.re_output_theta_new = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, future_len),
        )

        self.re_output_pi_new = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )

    def forward(self, encoding, mask=None, data=None):      

        y_hat_new = None
        pi_new = None
        y_hat_new_new = None
        pi_new_new = None
        
        ### initial ###
        mode_query = encoding[:, 0]
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        mode = mode_query[:, None] + multi_modal_query

        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask)
        for blk in self.self_block_mode:
            mode = blk(mode)

        y_hat, pi = self.predictor(mode)

        ### refine ###
        y_hat_refine = y_hat.detach()
        
        ###########################################################################
        end_point_refine = y_hat_refine[:, :, -1]
        centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        relative_centers_r = end_point_refine[..., 0].unsqueeze(2) - centers[..., 0].unsqueeze(1)
        relative_centers_theta = end_point_refine[..., 1].unsqueeze(2) - centers[..., 1].unsqueeze(1)
        relative_angles = end_point_refine[..., 1].unsqueeze(2) - angles.unsqueeze(1)
        relative_embed = torch.cat(
            [
                relative_centers_r.unsqueeze(-1), 
                torch.stack([torch.cos(relative_centers_theta), torch.sin(relative_centers_theta)], dim=-1),
                torch.stack([torch.cos(relative_angles), torch.sin(relative_angles)], dim=-1),
            ], 
            dim=-1
        )
        relative_feat = self.pos_embed_relative_refine(relative_embed)
        ###########################################################################

        r = torch.zeros((y_hat_refine.size(0), y_hat_refine.size(1), y_hat_refine.size(2)+1), dtype=y_hat_refine.dtype, device=y_hat_refine.device)
        r[..., 1:] = y_hat_refine[..., 0]
        r_delta = r[..., 1:] - r[..., :-1]
        theta = torch.zeros((y_hat_refine.size(0), y_hat_refine.size(1), y_hat_refine.size(2)+1), dtype=y_hat_refine.dtype, device=y_hat_refine.device)
        theta[..., 1:] = y_hat_refine[..., 1]
        theta_delta = theta[..., 1:] - theta[..., :-1]
        y_hat_refine_input = torch.cat(
            [
                y_hat_refine[..., 0].unsqueeze(-1),
                r_delta.unsqueeze(-1),
                torch.stack([torch.cos(y_hat_refine[..., 1]), torch.sin(y_hat_refine[..., 1])], dim=-1),
                torch.stack([torch.cos(theta_delta), torch.sin(theta_delta)], dim=-1),
            ],
            dim=-1,
        )
        mode_refine = self.re_input(y_hat_refine_input.reshape(y_hat_refine_input.size(0), y_hat_refine_input.size(1), -1))
        
        mode_refine = mode_refine + mode
        # Relative Embedding Transformer
        mode_refine = self.blocks_relative_refine(mode_refine, encoding, encoding, relative_feat)
        for blk in self.self_block_mode_refine:
            mode_refine = blk(mode_refine)
        
        y_hat_diff_r = self.re_output_r(mode_refine) 
        y_hat_diff_theta = self.re_output_theta(mode_refine)
        pi_new = self.re_output_pi(mode_refine).squeeze(-1)
        y_hat_new = torch.zeros_like(y_hat_refine)
        y_hat_new[..., 0] = y_hat_diff_r
        y_hat_new[..., 1] = y_hat_diff_theta

        ### refine again ###
        y_hat_new_refine = y_hat_new.detach()
        
        ###########################################################################
        end_point_refine = y_hat_new_refine[:, :, -1]
        centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        relative_centers_r = end_point_refine[..., 0].unsqueeze(2) - centers[..., 0].unsqueeze(1)
        relative_centers_theta = end_point_refine[..., 1].unsqueeze(2) - centers[..., 1].unsqueeze(1)
        relative_angles = end_point_refine[..., 1].unsqueeze(2) - angles.unsqueeze(1)
        relative_embed = torch.cat(
            [
                relative_centers_r.unsqueeze(-1), 
                torch.stack([torch.cos(relative_centers_theta), torch.sin(relative_centers_theta)], dim=-1),
                torch.stack([torch.cos(relative_angles), torch.sin(relative_angles)], dim=-1),
            ], 
            dim=-1
        )
        relative_feat = self.pos_embed_relative_refine_new(relative_embed)
        ###########################################################################

        r = torch.zeros((y_hat_new_refine.size(0), y_hat_new_refine.size(1), y_hat_new_refine.size(2)+1), dtype=y_hat_new_refine.dtype, device=y_hat_new_refine.device)
        r[..., 1:] = y_hat_new_refine[..., 0]
        r_delta = r[..., 1:] - r[..., :-1]
        theta = torch.zeros((y_hat_new_refine.size(0), y_hat_new_refine.size(1), y_hat_new_refine.size(2)+1), dtype=y_hat_new_refine.dtype, device=y_hat_new_refine.device)
        theta[..., 1:] = y_hat_new_refine[..., 1]
        theta_delta = theta[..., 1:] - theta[..., :-1]
        y_hat_refine_input = torch.cat(
            [
                y_hat_new_refine[..., 0].unsqueeze(-1),
                r_delta.unsqueeze(-1),
                torch.stack([torch.cos(y_hat_new_refine[..., 1]), torch.sin(y_hat_new_refine[..., 1])], dim=-1),
                torch.stack([torch.cos(theta_delta), torch.sin(theta_delta)], dim=-1),
            ],
            dim=-1,
        )
        mode_refine_new = self.re_input_new(y_hat_refine_input.reshape(y_hat_refine_input.size(0), y_hat_refine_input.size(1), -1))
        mode_refine_new = mode_refine + mode_refine_new

        # Relative Embedding Transformer
        mode_refine_new = self.blocks_relative_refine_new(mode_refine_new, encoding, encoding, relative_feat)
        for blk in self.self_block_mode_refine_new:
            mode_refine_new = blk(mode_refine_new)

        y_hat_diff_r = self.re_output_r_new(mode_refine_new) 
        y_hat_diff_theta = self.re_output_theta_new(mode_refine_new)
        pi_new_new = self.re_output_pi_new(mode_refine_new).squeeze(-1)
        y_hat_new_new = torch.zeros_like(y_hat_new_refine)
        y_hat_new_new[..., 0] = y_hat_diff_r
        y_hat_new_new[..., 1] = y_hat_diff_theta

        return y_hat, pi, y_hat_new, pi_new, y_hat_new_new, pi_new_new
