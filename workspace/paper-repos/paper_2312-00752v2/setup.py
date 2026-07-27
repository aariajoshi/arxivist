from setuptools import setup, find_packages

setup(
    name="mamba",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "pyyaml>=6.0",
    ],
)
