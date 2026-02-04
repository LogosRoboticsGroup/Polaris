from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def cartesian_to_polar(cartesian_coords):
    x = cartesian_coords[..., 0]
    y = cartesian_coords[..., 1]
    
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)  # 角度范围 [-π, π]

    # 对于 r == 0 的情况，将 theta 设置为 0
    theta = torch.where(r == 0, torch.tensor(0.0, device=theta.device, dtype=theta.dtype), theta)
    
    polar_coords = torch.stack((r, theta), dim=-1)
    return polar_coords


def polar_to_cartesian(polar_coords):
    r = polar_coords[..., 0]
    theta = polar_coords[..., 1]
    
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    
    cartesian_coords = torch.stack((x, y), dim=-1)
    return cartesian_coords


def normalize_angle(angles):
    """
    将角度值标准化到 -π 到 π 之间。

    参数:
    angles (torch.Tensor): 表示角度值的张量，单位为弧度。

    返回:
    torch.Tensor: 标准化后的角度值，范围在 -π 到 π 之间。
    """
    normalized_angles = (angles + torch.pi) % (2 * torch.pi) - torch.pi
    return normalized_angles


# 角度转换为弧度
def degrees_to_radians(degrees):
    return degrees * (torch.pi / 180)


# 弧度转换为角度
def radians_to_degrees(radians):
    return radians * (180 / torch.pi)


class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        num_historical_steps: int = 50,
        sequence_origins: List[int] = [50],
        radius: float = 150.0,
        train_mode: str = 'only_focal',

    ):
        assert sequence_origins[-1] == 50 and num_historical_steps <= 50
        assert train_mode in ['only_focal', 'focal_and_scored']
        assert split in ['train', 'val', 'test']
        super(Av2Dataset, self).__init__()
        self.data_folder = Path(data_root) / split
        self.file_list = sorted(list(self.data_folder.glob('*.pt')))
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = 0 if split =='test' else 60
        self.sequence_origins = sequence_origins
        self.mode = 'only_focal' if split != 'train' else train_mode
        self.radius = radius
        # TODO: if use transform
        self.transform = False

        print(
            f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}'
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index])
        data = self.process(data)
        data = self.transform_data(data)
        return data
    
    def transform_data(self, data):
        if self.transform:
            pass
        else:
            pass

        return data
    
    def process(self, data):
        sequence_data = []
        train_idx = [data['focal_idx']]
        if self.mode == 'focal_and_scored':
            train_idx += data['scored_idx']
        for cur_step in self.sequence_origins:
            # focal_and_scored
            for ag_idx in train_idx:
                ag_dict = self.process_single_agent(data, ag_idx, cur_step)
            sequence_data.append(ag_dict)
        return sequence_data

    def process_single_agent(self, data, idx, step=50):
        # info for cur_agent on cur_step
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, step - 1].double()
        theta = data['x_angles'][idx, step - 1].double()  # data['x_angles'] range [-π, π]
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        ag_mask = torch.norm(data['x_positions'][:, step - 1] - origin, dim=-1) < self.radius
        ag_mask = ag_mask * data['x_valid_mask'][:, step - 1]
        ag_mask[idx] = False

        # transform agents to local
        st, ed = step - self.num_historical_steps, step + self.num_future_steps
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]])
        pos = data['x_positions'][:, st: ed]
        pos = torch.cat([pos[[idx]], pos[ag_mask]])
        head = data['x_angles'][:, st: ed]
        head = torch.cat([head[[idx]], head[ag_mask]])
        vel = data['x_velocity'][:, st: ed]
        vel = torch.cat([vel[[idx]], vel[ag_mask]])
        vel_vector = data['x_velocity_vector'][:, st: ed]
        vel_vector = torch.cat([vel_vector[[idx]], vel_vector[ag_mask]])
        valid_mask = data['x_valid_mask'][:, st: ed]
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]])

        pos[valid_mask] = torch.matmul(pos[valid_mask].double() - origin, rotate_mat).to(torch.float32)
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi

        # transform lanes to local
        l_pos = data['lane_positions']
        l_attr = data['lane_attr']
        l_is_int = data['is_intersections']
        l_pos = torch.matmul(l_pos.reshape(-1, 2).double() - origin, rotate_mat).reshape(-1, l_pos.size(1), 2).to(torch.float32)

        l_ctr = l_pos[:, 9:11].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        )
        l_valid_mask = (
            (l_pos[:, :, 0] > -self.radius) & (l_pos[:, :, 0] < self.radius)
            & (l_pos[:, :, 1] > -self.radius) & (l_pos[:, :, 1] < self.radius)
        )

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_is_int = l_is_int[l_mask]
        l_attr = l_attr[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        # remove outliers
        nearest_dist = torch.cdist(pos[:, self.num_historical_steps - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values
        ag_mask = nearest_dist < 5
        ag_mask[0] = True
        pos = pos[ag_mask]
        head = head[ag_mask]
        vel = vel[ag_mask]
        vel_vector = vel_vector[ag_mask]
        attr = attr[ag_mask]
        valid_mask = valid_mask[ag_mask]

        # post_process
        head = head[:, :self.num_historical_steps]
        vel_future = vel[:, self.num_historical_steps:]
        vel = vel[:, :self.num_historical_steps]
        vel_vector_his = vel_vector[:, :self.num_historical_steps]
        pos_ctr = pos[:, self.num_historical_steps - 1].clone()
        if self.num_future_steps > 0:
            type_mask = attr[:, [-1]] != 3
            pos, target = pos[:, :self.num_historical_steps], pos[:, self.num_historical_steps:]
            target_mask = type_mask & valid_mask[:, [self.num_historical_steps - 1]] & valid_mask[:, self.num_historical_steps:]
            valid_mask = valid_mask[:, :self.num_historical_steps]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros(pos_ctr.size(0), 60, 2),   
            )
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :self.num_historical_steps - 1] & valid_mask[:, 1: self.num_historical_steps]
        
        tmp_vel_vector_his = vel_vector_his.clone()
        tmp_vel_vector_his = torch.where(
            valid_mask.unsqueeze(-1),
            tmp_vel_vector_his, torch.zeros(tmp_vel_vector_his.size(0), self.num_historical_steps, 2),   
        )

        # acceleration vector
        vel_vector_his_diff = vel_vector_his[:, 1:self.num_historical_steps] - vel_vector_his[:, :self.num_historical_steps - 1]
        vel_vector_his[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            vel_vector_his_diff, torch.zeros(vel_vector_his.size(0), self.num_historical_steps - 1, 2)
        )
        vel_vector_his[:, 0] = torch.zeros(vel_vector_his.size(0), 2)

        tmp_pos = pos.clone()
        tmp_pos = torch.where(
            valid_mask.unsqueeze(-1),
            tmp_pos - pos_ctr.unsqueeze(1), torch.zeros(pos_ctr.size(0), self.num_historical_steps, 2),   
        )
        pos_diff = pos[:, 1:self.num_historical_steps] - pos[:, :self.num_historical_steps - 1]
        ### add target velocity and acceleration ###
        target_diff = None
        if target is not None:
            target_diff_tmp = torch.cat((pos[:, -1].unsqueeze(1), target), dim=1)
            target_diff = target_diff_tmp[:, 1:self.num_future_steps+1] - target_diff_tmp[:, :self.num_future_steps]
            target_diff_tmp = target_diff.clone()
            diff_mask_target_tmp = torch.cat((valid_mask[:,-1].unsqueeze(1), target_mask), dim=1)
            diff_mask_target = diff_mask_target_tmp[:, 1:self.num_future_steps + 1] & diff_mask_target_tmp[:, : self.num_future_steps]
            target_diff[:, :] = torch.where(
                diff_mask_target.unsqueeze(-1),
                target_diff_tmp, torch.zeros(target_diff_tmp.size(0), self.num_future_steps, 2)
            )
        ### end ###
        pos[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), self.num_historical_steps - 1, 2)
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2)

        tmp_vel = vel.clone()
        vel_diff = vel[:, 1:self.num_historical_steps] - vel[:, :self.num_historical_steps - 1]
        vel[:, 1:self.num_historical_steps] = torch.where(
            diff_mask,
            vel_diff, torch.zeros(vel.size(0), self.num_historical_steps - 1)
        )
        vel[:, 0] = torch.zeros(vel.size(0))
        ### add target velocity and acceleration ###
        if target is not None:
            tmpvel_future = vel_future.clone()
            tmpvel_future = torch.cat((tmp_vel[:, -1].unsqueeze(1), tmpvel_future), dim=1)
            vel_diff_future= tmpvel_future[:, 1:self.num_future_steps+1] - tmpvel_future[:, :self.num_future_steps]
            vel_future[:, :] = torch.where(
                diff_mask_target,
                vel_diff_future, torch.zeros(vel_diff_future.size(0), self.num_future_steps)
            )
        ### end ###
        
        tmp_pos = cartesian_to_polar(tmp_pos)
        pos_ctr = cartesian_to_polar(pos_ctr)
        tmp_vel_vector_his = cartesian_to_polar(tmp_vel_vector_his)
        vel_vector_his = cartesian_to_polar(vel_vector_his)
        l_pos = cartesian_to_polar(l_pos)
        l_ctr = cartesian_to_polar(l_ctr)
        
        return {
            'target': target,  ## loss ##
            'target_mask': target_mask,  ## loss ##

            'target_diff': target_diff,
            'target_vel_diff': vel_future,
            'x_positions_diff': pos,  

            'x_positions': tmp_pos,  #### model, position vector to the center point ####
            'x_velocity_vector': tmp_vel_vector_his,  #### model, velocity vector ####
            'x_accelerate_vector': vel_vector_his,  #### model, acceleration vector ####
            'x_attr': attr,  # model #
            'x_centers': pos_ctr,  # model #
            'x_angles': head,  # model #
            'x_valid_mask': valid_mask,  # model #

            'x_velocity': tmp_vel,
            'x_velocity_diff': vel,  

            'lane_positions': l_pos,  # model #
            'lane_attr': l_attr,  # model #
            'lane_centers': l_ctr,  # model #
            'lane_angles': l_head,  # model #
            'lane_valid_mask': l_valid_mask,  # model #

            'is_intersections': l_is_int,

            'origin': origin.view(1, 2),  ### submission ###
            'theta': theta.view(1),  ### submission ###
            'scenario_id': data['scenario_id'],  ### submission ###
            'track_id': cur_agent_id,  ### submission ###

            'city': data['city'],
            'timestamp': torch.Tensor([step * 0.1]),
        }
    

def collate_fn(seq_batch):
    seq_data = []
    for i in range(len(seq_batch[0])):
        batch = [b[i] for b in seq_batch]
        data = {}

        for key in [
            'x_positions_diff',
            'x_attr',
            'x_positions',
            'x_centers',
            'x_angles',
            'x_velocity',
            'x_velocity_vector',
            'x_accelerate_vector',
            'x_velocity_diff',
            'lane_positions',
            'lane_centers',
            'lane_angles',
            'lane_attr',
            'is_intersections',
        ]:
            data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

        if 'x_scored' in batch[0]:
            data['x_scored'] = pad_sequence(
                [b['x_scored'] for b in batch], batch_first=True
            )

        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_diff'] = pad_sequence([b['target_diff'] for b in batch], batch_first=True)
            data['target_vel_diff'] = pad_sequence([b['target_vel_diff'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )

        for key in ['x_valid_mask', 'lane_valid_mask']:
            data[key] = pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=False
            )

        data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)
        data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)

        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['track_id'] = [b['track_id'] for b in batch]

        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        seq_data.append(data)
    return seq_data
