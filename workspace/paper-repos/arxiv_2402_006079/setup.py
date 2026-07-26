from setuptools import find_packages, setup

setup(
    name="discdiff-repro",
    version="0.1.0",
    description="Reproduction of DiscDiff: Latent Diffusion Model for DNA Sequence Generation (ICML 2024)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy>=1.23", "scipy>=1.9", "pyyaml>=6.0", "tqdm>=4.64"],
    extras_require={
        "realdata": ["datasets>=2.14", "transformers>=4.30"],  # EPD-GenDNA + Hyena
        "dev": ["pytest>=7.0"],
    },
)
