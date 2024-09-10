from setuptools import setup, find_packages
setup(
name='charLM',
version='0.1.2',
author='Jose Cruzado',
author_email='josecruzado2103@gmail.com',
description='This package provides a simpleway to train language models at the character level.',
url = "https://github.com/josecruzado21/charLM",
packages=find_packages(),
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.6',
install_requires=[
    "numpy==2.1.0"
],
)