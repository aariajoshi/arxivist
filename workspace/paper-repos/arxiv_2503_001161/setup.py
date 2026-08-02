from setuptools import find_packages, setup

setup(
    name="sgdd-repro",
    version="0.1.0",
    description="Reproduction of Split Gibbs Discrete Diffusion Posterior Sampling (NeurIPS 2025)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["numpy>=1.23", "scipy>=1.9", "pyyaml>=6.0", "tqdm>=4.64"],
    extras_require={
        # Real-data SEDD tasks (DNA/MNIST/music) need torch + the official SEDD repo.
        "realdata": ["torch>=2.0", "transformers>=4.30"],
        "dev": ["pytest>=7.0"],
    },
)
