from setuptools import setup, find_packages

setup(
    name="improving_rf_smoothing",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
