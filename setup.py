from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow==2.6.4',
    'Keras==2.2.4',
    'scikit-learn==0.21',
    'pandas==0.24',
]

# with open('requirements.txt','r') as f:
#     REQUIRED_PACKAGES.append(f.readlines())

setup(
    name='twconvrecsys',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='twconvrecsys'
)
