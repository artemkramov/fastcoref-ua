from setuptools import setup, find_packages

import sys
if sys.version_info < (3,7):
    sys.exit('Python < 3.7 is not supported')

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='fastcoref',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='1.5',
    license='MIT',
    author="Shon Otmazgin, Arie Cattan, Yoav Goldberg",
    author_email='shon711@gmail.com',
    packages=find_packages('src'),
    url='https://github.com/shon-otmazgin/fastcoref',
    install_requires=[
        'tqdm>=4.64.0',
        'numpy>=1.21.6',
        'scipy>=1.7.3',
        'pandas>=1.3.5',
        'spacy>=3.0.6',
        'torch>=1.10.0',
        'transformers>=4.11.3',
        'datasets>=2.1.0',
        'wandb>=0.12.15',
      ],

)