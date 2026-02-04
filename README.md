# Relative Position Matters: Trajectory Prediction and Planning with Polar Representation
### [[Paper]](https://arxiv.org/abs/2508.11492)

> [**Relative Position Matters: Trajectory Prediction and Planning with Polar Representation**](https://arxiv.org/abs/2508.11492)            
> [Bozhou Zhang](https://zbozhou.github.io/), [Nan Song](https://scholar.google.com/citations?hl=zh-CN&user=wLZVtjEAAAAJ), [Bingzhao Gao](https://scholar.google.com/citations?user=GvK2l7sAAAAJ&hl=zh-TW&oi=ao), [Li Zhang](https://lzrobots.github.io)  
> **ICRA 2026**

## Abstract
Trajectory prediction and planning in autonomous driving are highly challenging due to the complexity of predicting surrounding agents' movements and planning the ego agent's actions in dynamic environments. Existing methods encode map and agent positions and decode future trajectories in Cartesian coordinates. However, modeling the relationships between the ego vehicle and surrounding traffic elements in Cartesian space can be suboptimal, as it does not naturally capture the varying influence of different elements based on their relative distances and directions. To address this limitation, we adopt the Polar coordinate system, where positions are represented by radius and angle. This representation provides a more intuitive and effective way to model spatial changes and relative relationships, especially in terms of distance and directional influence. Based on this insight, we propose **Polaris**, a novel method that operates entirely in Polar coordinates, distinguishing itself from conventional Cartesian-based approaches. By leveraging the Polar representation, this method explicitly models distance and direction variations and captures relative relationships through dedicated encoding and refinement modules, enabling more structured and spatially aware trajectory prediction and planning. Extensive experiments on the challenging prediction (Argoverse 2) and planning benchmarks (nuPlan) demonstrate that Polaris achieves state-of-the-art performance. 

## Pipeline
<div align="center">
  <img src="assets/main.png"/>
</div><br/>

## Install the environment
```
# Set up a new virtual environment
conda create -n Polaris python=3.10
conda activate Polaris

# Install dependency packpages
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r ./requirements.txt
pip install av2==0.2.1

# Some packages may be useful
pip install tensorboard
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install protobuf==3.20.3
pip install numpy==1.26.3
pip3 install natten==0.17.3+torch210cu118 -f https://whl.natten.org/old
```

## Prepare the data
### Setup [Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html)
```
data_root
    ├── train
    │   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7
    │   ├── 0000b6ab-e100-4f6b-aee8-b520b57c0530
    │   ├── ...
    ├── val
    │   ├── 00010486-9a07-48ae-b493-cf4545855937
    │   ├── 00062a32-8d6d-4449-9948-6fedac67bfcd
    │   ├── ...
    ├── test
    │   ├── 0000b329-f890-4c2b-93f2-7e2413d4ca5b
    │   ├── 0008c251-e9b0-4708-b762-b15cb6effc27
    │   ├── ...
```

### Preprocess
```
python preprocess.py --data_root=/path/to/data_root -p
```

### The structure of the dataset after processing
```
└── data
    └── Polaris_processed
        ├── train
        ├── val
        └── test
```

## Training and testing
```
# Train
python train.py 

# Val, remember to change the checkpoint to your own in eval.py
python eval.py

# Model ensembling
python ensemble.py

# Test for submission
python eval.py gpus=1 test=true
```

## Results and checkpoints

| Models | minADE1 | minFDE1 | minADE6 | minFDE6 |
| :- | :-: | :-: | :-: | :-: |
| [Polaris_WO_Mamba](https://drive.google.com/file/d/1418sE_6pL8KAtGMncaRs5LiNXuCDojUz/view?usp=sharing) |  1.62  |  3.99  |  0.65  |  1.22  |

## BibTeX
```bibtex
@article{zhang2025polaris,
 title={Relative Position Matters: Trajectory Prediction and Planning with Polar Representation},
 author={Zhang, Bozhou and Song, Nan and Gao, Bingzhao and Zhang, Li},
 journal={ICRA},
 year={2026},
}
```
