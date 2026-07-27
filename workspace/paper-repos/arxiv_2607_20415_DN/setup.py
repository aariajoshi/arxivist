from setuptools import find_packages, setup

setup(
    name="fcdf_diagonal_frog",
    version="0.1.0",
    description="Reproduction of 'Flux-Corrected Diagonal Frog: second order and positivity "
                 "at all time steps' (Itkin, arXiv:2607.20415v1)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<2.0",
        "scipy>=1.11",
        "matplotlib>=3.8",
        "pyyaml>=6.0",
        "pandas>=2.0",
    ],
)
