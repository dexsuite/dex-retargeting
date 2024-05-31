import importlib.metadata
import importlib.util
import re
from pathlib import Path

from setuptools import setup, find_packages

_here = Path(__file__).resolve().parent
name = "dex_retargeting"

# Reference: https://github.com/kevinzakka/mjc_viewer/blob/main/setup.py
with open(_here / name / "__init__.py") as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

core_requirements = [
    "numpy",
    "pytransform3d",
    "pin>=2.7.0",
    "nlopt",
    "trimesh",
    "anytree",
    "pyyaml",
    "lxml",
]

# Check whether you have torch installed
torch_info = importlib.util.find_spec("torch")
if torch_info is not None:
    version = importlib.metadata.version("torch")
    major_version = version.split(".")[0]
    if int(major_version) >= 2:
        print(f"A valid torch with version {version}: has been already installed, skip it.")
    else:
        raise RuntimeError(
            f"dex-retargeting requires a torch version of 2.0.0 or higher. Currently, version {version} is installed.\n"
            "Please uninstall the current torch or install torch >= 2.0.0, then reinstall this package."
        )
else:
    print(
        "\033[33m",
        "No pre-installed torch detected. A GPU-only version will be installed.\n"
        "Note that dex-retargeting is compatible with both CPU and GPU versions of torch, as it only requires the CPU features.\n"
        "To save time and space, you can also install a torch cpu version and reinstall this package.\n",
        "\033[39m",
    )

    core_requirements.append("torch")

dev_requirements = [
    "pytest",
    "black",
    "isort",
    "pytest-xdist",
    "pyright",
    "ruff",
    "mypy",
]

example_requirements = ["tyro", "tqdm", "opencv-python", "mediapipe", "sapien==3.0.0b0", "loguru"]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]


def setup_package():
    # Meta information of the project
    author = "Yuzhe Qin"
    author_email = "y1qin@ucsd.edu"
    description = "Hand pose retargeting for dexterous robot hand."
    url = "https://github.com/dexsuite/dex-retargeting"
    with open(_here / "README.md", "r") as file:
        readme = file.read()

    # Package data
    packages = find_packages(".")
    print(f"Packages: {packages}")

    setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        maintainer=author,
        maintainer_email=author_email,
        description=description,
        long_description=readme,
        long_description_content_type="text/markdown",
        url=url,
        license="MIT",
        license_files=("LICENSE",),
        packages=packages,
        python_requires=">=3.7,<3.11",
        zip_safe=True,
        include_package_data=True,
        package_data={'dex_retargeting': ['configs/**']},
        install_requires=core_requirements,
        extras_require={
            "dev": dev_requirements,
            "example": example_requirements,
        },
        classifiers=classifiers,
    )


setup_package()
