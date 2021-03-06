from setuptools import setup, find_packages

setup(
    name='costaware',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'pandas',
        'seaborn',
        'ray',
        'numpy',
        'torch',
        'pyyaml',
        'gym',
        'scipy',
        'wesutils>=0.0.7'
    ]
)
