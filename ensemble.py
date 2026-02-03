import os
import pickle
import numpy as np
import tqdm
import warnings
import torch
from sklearn.cluster import KMeans
from src.utils.submission_av2 import SubmissionAv2

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# Config: 6 个模型目录
# =========================
ENSEMBLE_DIRS = [
    "save_for_en/en_1",
    "save_for_en/en_2",
    "save_for_en/en_3",
    "save_for_en/en_4",
    "save_for_en/en_5",
    "save_for_en/en_6",
]

N_CLUSTERS = 6
STRICT = True  # True: 任一模型缺文件就报错；False: 缺了就跳过该模型

submission_handler = SubmissionAv2(save_dir="ensemble")


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def file_exists(d: str, fname: str) -> bool:
    return os.path.isfile(os.path.join(d, fname))


# =========================
# 主循环：以第一个目录为主遍历文件
# =========================
base_dir = ENSEMBLE_DIRS[0]

for dirpath, dirnames, filenames in os.walk(base_dir):
    for filename in tqdm.tqdm(filenames, desc="Ensembling"):
        # 收集 6 个模型中该文件对应的 data
        datas = []
        missing = []

        for d in ENSEMBLE_DIRS:
            p = os.path.join(d, filename)
            if os.path.isfile(p):
                datas.append(load_pkl(p))
            else:
                missing.append(d)

        if missing:
            msg = f"[Missing] {filename} not found in {len(missing)} dirs."
            if STRICT:
                raise FileNotFoundError(msg + " Missing dirs:\n" + "\n".join(missing))
            else:
                # 弱模式：跳过缺失的模型（仍继续做 ensemble）
                print(msg)
                if len(datas) == 0:
                    continue

        # -------------------------
        # 取出 traj / pi 并 softmax
        # -------------------------
        # datas[i]["y_hat"]: [K_i, T, 2]
        # datas[i]["pi"]:    [K_i] (logits)
        traj_list = []
        pi_list = []

        for di in datas:
            traj_list.append(di["y_hat"])
            pi_list.append(torch.softmax(di["pi"].double(), dim=-1))

        # concat 后：traj_all [sumK, T, 2], pi_sum [sumK]
        traj_all = torch.cat(traj_list, dim=0).cpu().numpy()
        pi_sum = torch.cat(pi_list, dim=0).cpu().numpy()

        # -------------------------
        # Weighted KMeans on endpoints
        # -------------------------
        endpoints = traj_all[:, -1, :2]  # [sumK, 2]
        # sklearn 的 sample_weight 需要 shape [n_samples]
        kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=0,
            init="k-means++",
        ).fit(endpoints, sample_weight=pi_sum)

        labels = kmeans.labels_

        # 按 cluster 聚合轨迹与权重
        clusters = {}
        cluster_scores = {}
        for i, c in enumerate(labels):
            clusters.setdefault(c, []).append(traj_all[i])
            cluster_scores.setdefault(c, []).append(pi_sum[i])

        # 每个 cluster：轨迹取均值；pi 取权重和
        final_traj = []
        final_pi = []
        for c in sorted(clusters.keys()):  # 排序保证输出顺序稳定
            final_traj.append(np.mean(np.stack(clusters[c], axis=0), axis=0))
            final_pi.append(np.sum(cluster_scores[c]))

        final_y_hat = np.asarray(final_traj)[None, ...]  # [1, 6, T, 2]
        final_pi = np.asarray(final_pi)[None, ...]       # [1, 6]

        # -------------------------
        # 组装 submission
        # -------------------------
        ref = datas[0]  # 用第一个模型的 meta 作为 reference
        meta = {
            "scenario_id": [ref["scenario_id"]],
            "track_id": [ref["track_id"]],
            "origin": ref["origin"].to("cpu"),
            "theta": ref["theta"].to("cpu").unsqueeze(0).unsqueeze(0),
        }

        submission_handler.format_data(
            meta,
            torch.from_numpy(final_y_hat),
            torch.from_numpy(final_pi),
        )

    break

submission_handler.generate_submission_file()
