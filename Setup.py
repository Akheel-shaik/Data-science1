from setuptools import find_packages,setup
from typing import List

def get_requirements(filepath: str) -> List[str]:
    packages = []
    try:
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('-e') and not line.startswith('#') and line != '.':
                    packages.append(line)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' does not exist.")
    return packages


setup(
name = 'Datascience1',
version= '0.0.1',
author='Akheel',
author_email='akheelahamed365@gmail.com',
packages= find_packages(),
install_requires = get_requirements('Requirement.txt')
)