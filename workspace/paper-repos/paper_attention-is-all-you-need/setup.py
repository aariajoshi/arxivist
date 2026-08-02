from setuptools import setup, find_packages

setup(
    name="transformer",
    version="0.1.0",
    description="Reproduction of Attention Is All You Need",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.2.0",
        "PyYAML==6.0",
        "sacrebleu==2.3.1",
        "tokenizers==0.14.0"
    ],
)
