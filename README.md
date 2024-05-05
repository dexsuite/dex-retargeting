Dex Retargeting
---
<p align="center">
    <!-- code check badges -->
    <a href='https://github.com/dexsuite/dex-retargeting/blob/main/.github/workflows/test.yml'>
        <img src='https://github.com/dexsuite/dex-retargeting/actions/workflows/test.yml/badge.svg' alt='Test Status' />
    </a>
    <!-- license badge -->
    <a href="https://github.com/dexsuite/dex-retargeting/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
    </a>
</p>

## Installation

```shell
# Install from pypi
pip install dex_retargeting`` 

# Or install from Github
git clone https://github.com/dexsuite/dex-retargeting
cd dex-retargeting
pip install -e .
```

To run the example, you may need additional dependencies for rendering and hand pose detection.

```shell
pip install -e ".[example]"
```

## Examples

### Retargeting from human hand video

[Tutorial on retargeting from human hand video](example/vector_retargeting/README.md)

### Retarget from hand object pose dataset

[Tutorial on retargeting from hand-object pose dataset](example/position_retargeting/README.md)
