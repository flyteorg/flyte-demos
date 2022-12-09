from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(
    name='flyte-demo',
    version='0.0.0',
    url='https://github.com/flyteorg/flyte-demos',
    packages=find_packages(),    
    install_requires=install_requires,
)
