
from setuptools import setup, find_packages

setup(
    name='hmg',
    version="0.0.1",
    description='Hiding Secret Messages in Graph Datasets',
    author='Daewon Lee',
    author_email='daewon4you@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'numba',
        'pandas',
        'networkx',
        'bitstring',
        'tqdm'
    ]
)
