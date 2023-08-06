from setuptools import setup, find_packages
from typing import List
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


def get_requirements(file_name: str) -> List[str]:
    with open(os.path.join(ROOT_DIR, file_name)) as content:
        requirements = [req.replace("\n", "") for req in content.readlines() if not req.find('-e')!=-1]
        return requirements

setup(
    name="MlProject",
    version="1.0.1",
    author="Keith",
    author_email="keithfernandes311@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
