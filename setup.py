from setuptools import setup, find_packages

setup(
    name="labeltools",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        '': ['labelme2coco.py', 'generate_coco.py'],
    },
)