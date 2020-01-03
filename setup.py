from setuptools import setup, find_packages

setup(
    name='costaware',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'torch',
        'pyyaml',
        'gym'
    ]
)
