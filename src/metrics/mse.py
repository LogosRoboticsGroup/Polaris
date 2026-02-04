import numpy as np
from typing import Any, Callable, Dict, Optional
import torch
from torchmetrics import Metric


# 需要继承from torchmetrics import Metric初始化类来支持ddp。
class MSE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k=6,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(MSE, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, out) -> None:
        with torch.no_grad():
            step = len(out)
            for i in range(step):
                ori_lane = out[i]["lane"]
                recon_lane = out[i]["re_lane"]
            
                if ori_lane.shape != recon_lane.shape:
                    raise ValueError("The dimensions of the two images must match.")
            
                mse = torch.mean((ori_lane - recon_lane) ** 2)
                self.sum += mse.sum()
            self.count += step 

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


def calculate_mse(out):

    with torch.no_grad():
        step = len(out)
        mse_dict = {}
        for i in range(step):
            ori_lane = out[i]["lane"]
            recon_lane = out[i]["re_lane"]
        
            if ori_lane.shape != recon_lane.shape:
                raise ValueError("The dimensions of the two images must match.")
        
            mse = torch.mean((ori_lane - recon_lane) ** 2)
            mse_dict[f"mse_step_{i}"] = mse
    
    return mse_dict
