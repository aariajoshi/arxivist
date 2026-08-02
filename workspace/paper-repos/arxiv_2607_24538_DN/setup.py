from setuptools import find_packages, setup

setup(
    name="neo-nerf-editing",
    version="0.1.0",
    description="Unofficial reproduction scaffold for NEO: NeRF It Once, Edit It Many Times "
    "for Continuous Object Manipulation (arXiv:2607.24538).",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0,<3.0.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
        "scikit-image>=0.22.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
    ],
)
