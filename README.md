<div align="center">
  <h1 align="center"> Dex Retargeting </h1>
  <h3 align="center">
    Various retargeting optimizers to translate human hand motion to robot hand motion.
  </h3>
</div>
<p align="center">
  <!-- code check badges -->
  <a href='https://github.com/dexsuite/dex-retargeting/blob/main/.github/workflows/test.yml'>
      <img src='https://github.com/dexsuite/dex-retargeting/actions/workflows/test.yml/badge.svg' alt='Test Status' />
  </a>
  <!-- issue badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/issues">
  <img src="https://img.shields.io/github/issues/dexsuite/dex-retargeting.svg" alt="Issues">
  </a>
  <!-- release badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/tags">
  <img src="https://img.shields.io/github/v/release/dexsuite/dex-retargeting.svg?include_prereleases&sort=semver" alt="Releases">
  </a>
  <!-- license badge -->
  <a href="https://github.com/dexsuite/dex-retargeting/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
  </a>
</p>
<div align="center">
  <h4>This repo is part of the <a href="https://yzqin.github.io/anyteleop/">AnyTeleop Project</a></h4>
  <img src="example/vector_retargeting/teaser.webp" alt="Retargeting with different hands.">
</div>

## Installation

```shell
pip install dex_retargeting
```

To run the example, you may need additional dependencies for rendering and hand pose detection.

```shell
git clone https://github.com/dexsuite/dex-retargeting
cd dex-retargeting
pip install -e ".[example]"
```

## Examples

### Retargeting from human hand video

This type of retargeting can be used for applications like teleoperation.

[Tutorial on retargeting from human hand video](example/vector_retargeting/README.md)

### Retarget from hand object pose dataset

This type of retargeting can be used post-process human data for robot imitation,
e.g. [DexMV](https://yzqin.github.io/dexmv/).

[Tutorial on retargeting from hand-object pose dataset](example/position_retargeting/README.md)

## Citation

This repository is a part of the [AnyTeleop Project](https://yzqin.github.io/anyteleop/). If you use this work, kindly
reference it as:

```shell
@inproceedings{qin2023anyteleop,
  title     = {AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System},
  author    = {Qin, Yuzhe and Yang, Wei and Huang, Binghao and Van Wyk, Karl and Su, Hao and Wang, Xiaolong and Chao, Yu-Wei and Fox, Dieter},
  booktitle = {Robotics: Science and Systems},
  year      = {2023}
}
```

## Acknowledgments

This repository builds on the foundational work from [pinocchio](https://github.com/stack-of-tasks/pinocchio). Examples
within utilize a renderer derived from [SAPIEN](https://github.com/haosulab/SAPIEN).
The [PositionOptimizer](https://github.com/dexsuite/dex-retargeting/blob/main/dex_retargeting/optimizer.py) leverages
methodologies from our earlier
project, [From One Hand to Multiple Hands](https://yzqin.github.io/dex-teleop-imitation/). Additionally,
the [DexPilotOptimizer](https://github.com/dexsuite/dex-retargeting/blob/main/dex_retargeting/optimizer.py) is crafted
using insights from [DexPilot](https://sites.google.com/view/dex-pilot).
