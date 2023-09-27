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
    "numpy<1.24",
    "torch",
    "sapien>=2.0.0",
    "nlopt",
    "trimesh",
    "anytree",
    "pycollada",
    "pyyaml",
    "lxml",
]

dev_requirements = [
    "pytest",
    "black",
    "isort",
    "pytest-xdist",
    "pyright",
    "ruff",
]

example_requirements = [
    "tyro",
    "tqdm",
    "opencv-python",
    "mediapipe",
]

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
        install_requires=core_requirements,
        extras_require={
            "dev": dev_requirements,
            "example": example_requirements,
        },
        classifiers=classifiers,
    )


setup_package()
