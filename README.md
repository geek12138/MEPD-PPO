# MEPD-PPO
Multi-Expert Policy Distillation Guided Proximal Policy Optimization for Efficient Cooperation Emergence in Spatial Public Goods Games

Our work is inspired by the UNESCO-intangible heritage Tribunal de las Aguas de Valencia. Based on this inspiration, an innovative framework, namely the Multi-Expert Policy Distillation guided Proximal Policy Optimization (MEPD-PPO), is proposed. This novel approach resolves fundamental cooperation challenges in spatial public goods games under subcritical conditions. It achieves this by integrating knowledge distillation from diverse experts with deep reinforcement learning. Our framework uniquely employs temperature-scaled KL divergence minimization for efficient policy transfer and the convolution operation for enhanced spatial feature extraction. The proposed MEPD-PPO demonstrates breakthrough capabilities in establishing sustainable cooperation where conventional methods fail. Experimental validation confirms our method’s superior performance in achieving lower cooperation thresholds, faster convergence, and exceptional robustness across adversarial conditions. This research establishes a new paradigm for leadership-guided collective action systems. Multiple experts with diverse knowledge guide and bootstrap the emergence of cooperation in our MEPD-PPO framework. This mechanism bridges evolutionary principles with multi-agent learning, enabling more resilient socio-technical infrastructures.

## Requirements
It is worth mentioning that because python runs slowly, we use cuda library to improve the speed of code running.

```
* Python Version 3.12.2
* CUDA Version: 12.8
* torch Version: 2.2.1
* numpy Version: 1.26.4
* pandas Version: 2.2.3
```

## Installation
```bash
conda env create -f environment.yaml
```

## Usage
First, run to obtain expert models using main_PPO.py:
```bash
sh scripts/run_PPO_MEPD_expert.sh
```
Then, you can set the path of the expert model in scripts/run_CPO.MPD.sh, and then run main_PPO-ACT.py:
```bash
sh scripts/run_PPO_MEPD.sh
```

## Citation

If you use our codebase or models in your research, please cite this work.

```
@article{LI2026117477,
title = {Multi-expert policy distillation guided proximal policy optimization for efficient cooperation emergence in spatial public goods games},
journal = {Chaos, Solitons & Fractals},
volume = {202},
pages = {117477},
year = {2026},
issn = {0960-0779},
doi = {https://doi.org/10.1016/j.chaos.2025.117477},
url = {https://www.sciencedirect.com/science/article/pii/S0960077925014900},
author = {Chanchan Li and Zhaoqilin Yang and Wensheng Jia and Hongxin Zhao and Xin Wang}
}
```

Thanks https://github.com/Tychema/Learning-And-Propagation
