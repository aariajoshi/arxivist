"""
setup.py — Installation script for physics_motion_loss.

Install in editable mode for development::

    pip install -e .

Or standard install::

    pip install .
"""

from setuptools import setup, find_packages

setup(
    name="physics_motion_loss",
    version="0.1.0",
    description="Physics-Guided Motion Loss for Video Generation (arXiv 2506.02244)",
    author="ArXivist Implementation",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "omegaconf>=2.3.0",
        "peft>=0.9.0",
        "diffusers>=0.27.0",
        "accelerate>=0.27.0",
        "scipy>=1.11.0",
        "pillow>=10.0.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "matplotlib>=3.7.0",
        ],
        "flow": ["torchvision>=0.16.0"],
        "eval": ["open_clip_torch>=2.24.0", "wandb>=0.16.0"],
        "lm": ["openai>=1.0.0"],
    },
)
