import datetime
from pathlib import Path
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from src.datamodule.av2_dataset import polar_to_cartesian, normalize_angle, cartesian_to_polar
from .model_forecast import ModelForecast, StreamModelForecast


class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()
        
        # TODO: True to collect results, False to generate submission file
        self.pre_ensemble = False  

        model_type = model.pop('type')

        self.net = self.get_model(model_type)(**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "b-minFDE6": brier_minFDE(k=6),
            }
        )
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")
        self.val_metrics_new_new = metrics.clone(prefix="val_new_new_")
    
    def get_model(self, model_type):
        model_dict = {
            'ModelForecast': ModelForecast,
            'StreamModelForecast': StreamModelForecast,
        }
        assert model_type in model_dict
        return model_dict[model_type]

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        memory_dict = None
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            prediction, prob = self.submission_handler.format_data(cur_data, out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    def cal_loss(self, out, data, tag=''):
        # prediction
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        y_hat_new = out.get("y_hat_new", None)
        pi_new = out.get("pi_new", None)
        
        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # mode loss
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        y_hat_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        pi_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

        # main loss
        if y_hat_new is not None:
            l2_norm_new = torch.norm(y_hat_new[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            y_hat_new_best = y_hat_new[torch.arange(y_hat_new.shape[0]), best_mode_new]
            y_hat_new_reg_loss = F.smooth_l1_loss(y_hat_new_best[..., :2], y)            
        else:
            y_hat_new_reg_loss = 0
        if pi_new is not None:
            pi_new_cls_loss = F.cross_entropy(pi_new, best_mode_new.detach(), label_smoothing=0.2)
        else:
            pi_new_cls_loss = 0

        # others loss
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])

        loss = y_hat_reg_loss + pi_cls_loss + others_reg_loss
        loss = loss + y_hat_new_reg_loss + pi_new_cls_loss
        disp_dict = {
            f"{tag}y_hat_reg_loss": y_hat_reg_loss.item(),
            f"{tag}pi_cls_loss": pi_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
        }
        if y_hat_new is not None:
            disp_dict[f"{tag}y_hat_new_reg_loss"] = y_hat_new_reg_loss.item()
        if pi_new is not None:
            disp_dict[f"{tag}pi_new_cls_loss"] = pi_new_cls_loss.item()

        # polar loss
        y_hat_polar = out.get("y_hat_polar", None)
        if y_hat_polar is not None:
            y_polar = cartesian_to_polar(y)
            y_hat_polar_best = y_hat_polar[torch.arange(y_hat_polar.shape[0]), best_mode]
            y_hat_polar_reg_loss = F.smooth_l1_loss(y_hat_polar_best[..., 0], y_polar[..., 0]) + \
                                   F.smooth_l1_loss(y_hat_polar_best[..., 1], y_polar[..., 1])
            loss = loss + y_hat_polar_reg_loss
            disp_dict[f"{tag}y_hat_polar_reg_loss"] = y_hat_polar_reg_loss.item()
        
        y_hat_others_polar = out.get("y_hat_others_polar", None)
        if y_hat_others_polar is not None:
            y_others_polar = cartesian_to_polar(y_others)
            y_hat_others_polar_reg_loss = F.smooth_l1_loss(y_hat_others_polar[others_reg_mask][..., 0], y_others_polar[others_reg_mask][..., 0]) + \
                                          F.smooth_l1_loss(y_hat_others_polar[others_reg_mask][..., 1], y_others_polar[others_reg_mask][..., 1])
            loss = loss + y_hat_others_polar_reg_loss
            disp_dict[f"{tag}y_hat_others_polar_reg_loss"] = y_hat_others_polar_reg_loss.item()
        
        y_hat_new_polar = out.get("y_hat_new_polar", None)
        if y_hat_new_polar is not None:
            y_polar = cartesian_to_polar(y)
            y_hat_new_polar_best = y_hat_new_polar[torch.arange(y_hat_new_polar.shape[0]), best_mode_new]
            y_hat_new_polar_reg_loss = F.smooth_l1_loss(y_hat_new_polar_best[..., 0], y_polar[..., 0]) + \
                                       F.smooth_l1_loss(y_hat_new_polar_best[..., 1], y_polar[..., 1])
            loss = loss + y_hat_new_polar_reg_loss
            disp_dict[f"{tag}y_hat_new_polar_reg_loss"] = y_hat_new_polar_reg_loss.item()
        
        # refine again loss
        y_hat_new_new = out.get("y_hat_new_new", None)
        pi_new_new = out.get("pi_new_new", None)
        if y_hat_new_new is not None:
            l2_norm_new_new = torch.norm(y_hat_new_new[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new_new = torch.argmin(l2_norm_new_new, dim=-1)
            y_hat_new_new_best = y_hat_new_new[torch.arange(y_hat_new_new.shape[0]), best_mode_new_new]
            y_hat_new_new_reg_loss = F.smooth_l1_loss(y_hat_new_new_best[..., :2], y)            
        else:
            y_hat_new_new_reg_loss = 0
        if pi_new_new is not None:
            pi_new_new_cls_loss = F.cross_entropy(pi_new_new, best_mode_new_new.detach(), label_smoothing=0.2)
        else:
            pi_new_new_cls_loss = 0
        
        loss = loss + y_hat_new_new_reg_loss + pi_new_new_cls_loss

        if y_hat_new_new is not None:
            disp_dict[f"{tag}y_hat_new_new_reg_loss"] = y_hat_new_new_reg_loss.item()
        if pi_new_new is not None:
            disp_dict[f"{tag}pi_new_new_cls_loss"] = pi_new_new_cls_loss.item()
        
        y_hat_new_new_polar = out.get("y_hat_new_new_polar", None)
        if y_hat_new_new_polar is not None:
            y_polar = cartesian_to_polar(y)
            y_hat_new_new_polar_best = y_hat_new_new_polar[torch.arange(y_hat_new_new_polar.shape[0]), best_mode_new_new]
            y_hat_new_new_polar_reg_loss = F.smooth_l1_loss(y_hat_new_new_polar_best[..., 0], y_polar[..., 0]) + \
                                           F.smooth_l1_loss(y_hat_new_new_polar_best[..., 1], y_polar[..., 1])
            loss = loss + y_hat_new_new_polar_reg_loss
            disp_dict[f"{tag}y_hat_new_new_polar_reg_loss"] = y_hat_new_new_polar_reg_loss.item()

        disp_dict[f"{tag}loss"] = loss.item()

        return loss, disp_dict

    def relative_embedding(self, data):
        key_valid_mask = torch.cat(
            [data["x_key_valid_mask"], data["lane_key_valid_mask"]], dim=1
        )
        centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        relative_key_valid_mask = key_valid_mask.unsqueeze(1) & key_valid_mask.unsqueeze(2)
        relative_centers_r = centers[..., 0].unsqueeze(1) - centers[..., 0].unsqueeze(2)
        relative_centers_theta = centers[..., 1].unsqueeze(1) - centers[..., 1].unsqueeze(2)
        relative_angles = angles.unsqueeze(1) - angles.unsqueeze(2)
        relative_embed = torch.cat(
            [
                relative_centers_r.unsqueeze(-1), 
                torch.stack([torch.cos(relative_centers_theta), torch.sin(relative_centers_theta)], dim=-1),
                torch.stack([torch.cos(relative_angles), torch.sin(relative_angles)], dim=-1),
            ], 
            dim=-1
        )
        
        data['relative_key_valid_mask'] = relative_key_valid_mask
        data['relative_embed'] = relative_embed

        return data

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        data = self.relative_embedding(data)
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        data = self.relative_embedding(data)
        out = self(data)
        _, loss_dict = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data['target'][:, 0])
        if out['y_hat_new'] is not None:
            out['y_hat'] = out['y_hat_new']
        if out['pi_new'] is not None:
            out['pi'] = out['pi_new']
        if out['y_hat_new'] is not None:
            metrics_new = self.val_metrics_new(out, data['target'][:, 0])
        if out['y_hat_new_new'] is not None:
            out['y_hat'] = out['y_hat_new_new']
        if out['pi_new_new'] is not None:
            out['pi'] = out['pi_new_new']
        if out['y_hat_new_new'] is not None:
            metrics_new_new = self.val_metrics_new_new(out, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['y_hat_new'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
        if out['y_hat_new_new'] is not None:
            self.log_dict(
                metrics_new_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )

    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[-1]
        data = self.relative_embedding(data)
        out = self(data)
        if out['y_hat_new'] is not None:
            out['y_hat'] = out['y_hat_new']
        if out['pi_new'] is not None:
            out['pi'] = out['pi_new']
        if out['y_hat_new_new'] is not None:
            out['y_hat'] = out['y_hat_new_new']
        if out['pi_new_new'] is not None:
            out['pi'] = out['pi_new_new']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        if self.pre_ensemble:
            pass
        else:
            self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]


class StreamTrainer(Trainer):
    def __init__(self,
                 num_grad_frame=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_grad_frame = num_grad_frame
    
    def training_step(self, data, batch_idx):
        total_step = len(data)
        num_grad_frames = min(self.num_grad_frame, total_step)
        num_no_grad_frames = total_step - num_grad_frames

        memory_dict = None
        self.eval()
        with torch.no_grad():
            for i in range(num_no_grad_frames):
                cur_data = data[i]
                cur_data['memory_dict'] = memory_dict
                cur_data = self.relative_embedding(cur_data)
                out = self(cur_data)
                memory_dict = out['memory_dict']
        
        self.train()
        sum_loss = 0
        loss_dict = {}
        for i in range(num_grad_frames):
            cur_data = data[i + num_no_grad_frames]
            cur_data['memory_dict'] = memory_dict
            cur_data = self.relative_embedding(cur_data)
            out = self(cur_data)
            cur_loss, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i + num_no_grad_frames}_')
            loss_dict.update(cur_loss_dict)
            sum_loss += cur_loss
            memory_dict = out['memory_dict']
        loss_dict['loss'] = sum_loss.item()
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return sum_loss
    
    def validation_step(self, data, batch_idx):
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            if cur_data['x_positions_diff'].size(1) == 1:
                return
            cur_data['memory_dict'] = memory_dict
            cur_data = self.relative_embedding(cur_data)
            out = self(cur_data)
            _, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i}_')
            memory_dict = out['memory_dict']
            all_outs.append(out)
        
        metrics = self.val_metrics(all_outs[-1], data[-1]['target'][:, 0])
        if all_outs[-1]['y_hat_new'] is not None:
            all_outs[-1]['y_hat'] = all_outs[-1]['y_hat_new']
        if all_outs[-1]['pi_new'] is not None:
            all_outs[-1]['pi'] = all_outs[-1]['pi_new']
        if all_outs[-1]['y_hat_new'] is not None:
            metrics_new = self.val_metrics_new(all_outs[-1], data[-1]['target'][:, 0])
        if all_outs[-1]['y_hat_new_new'] is not None:
            all_outs[-1]['y_hat'] = all_outs[-1]['y_hat_new_new']
        if all_outs[-1]['pi_new_new'] is not None:
            all_outs[-1]['pi'] = all_outs[-1]['pi_new_new']
        if all_outs[-1]['y_hat_new_new'] is not None:
            metrics_new_new = self.val_metrics_new_new(all_outs[-1], data[-1]['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if all_outs[-1]['y_hat_new'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
        if all_outs[-1]['y_hat_new_new'] is not None:
            self.log_dict(
                metrics_new_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
    
    def test_step(self, data, batch_idx) -> None:
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            cur_data = self.relative_embedding(cur_data)
            out = self(cur_data)
            memory_dict = out['memory_dict']
            all_outs.append(out)

        if all_outs[-1]['y_hat_new'] is not None:
            all_outs[-1]['y_hat'] = all_outs[-1]['y_hat_new']
        if all_outs[-1]['pi_new'] is not None:
            all_outs[-1]['pi'] = all_outs[-1]['pi_new']
        if all_outs[-1]['y_hat_new_new'] is not None:
            all_outs[-1]['y_hat'] = all_outs[-1]['y_hat_new_new']
        if all_outs[-1]['pi_new_new'] is not None:
            all_outs[-1]['pi'] = all_outs[-1]['pi_new_new']

        if self.pre_ensemble:
            ensemble_num = 'en_1'
            batch = all_outs[-1]["pi"].size(0)
            for i in range(batch):
                scenario_id = data[-1]["scenario_id"][i]
                track_id = data[-1]["track_id"][i]
                data_info = {}
                data_info["scenario_id"] = data[-1]["scenario_id"][i]
                data_info["track_id"] = data[-1]["track_id"][i]
                data_info["origin"] = data[-1]["origin"][i]  # torch.Size([2])
                data_info["theta"] = data[-1]["theta"][i]  # torch.Size([1])
                data_info["y_hat"] = all_outs[-1]["y_hat"][i]  # torch.Size([6, 60, 2])
                data_info["pi"] = all_outs[-1]["pi"][i]  # torch.Size([6])
                with open(f'save_for_en/{ensemble_num}/{scenario_id}_with_{track_id}.pkl', 'wb') as outp: 
                    pickle.dump(data_info, outp)
        else:
            self.submission_handler.format_data(data[-1], all_outs[-1]["y_hat"], all_outs[-1]["pi"])
