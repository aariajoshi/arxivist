from setuptools import find_packages, setup

setup(
    name="quantum_cryptanalysis",
    version="0.1.0",
    description="ArXivist reproduction of Quantum Cryptanalysis on IBM Quantum Hardware (arXiv:2607.18340)",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
