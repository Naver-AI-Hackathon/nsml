#nsml: nvcr.io/nvidia/pytorch:19.07-py3

from distutils.core import setup

setup(
    name='nsml test example',
    version='1.1',
    install_requires=[
        'matplotlib==3.1.1',
        'tqdm==4.32.2',
        'pillow==9.0.0'
    ]
)
