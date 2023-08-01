from __future__ import absolute_import, division, print_function

import setuptools

__name__ = 'federatedscope'
__version__ = '0.2.0'
URL = 'https://github.com/alibaba/FederatedScope'

minimal_requires = [
    'adversarial-robustness-toolbox == 1.14.1','codecarbon == 2.2.1','dotmap == 1.3.30','numpy==1.22.4', 'scikit-learn==1.0.2', 'scipy==1.7.3', 'pandas ==  == 2.0.1','hashids == 1.3.1',
    'grpcio==1.55.0', 'grpcio-tools','protobuf == 3.19.4','pympler == 1.0.1', 'pyyaml==6.0', 'fvcore', 'iopath',
    'wandb==0.15.3','scikit-learn == 1.1.3','scipy ==1.7.3','shape == 0.41.0','tabulate == 0.9.0', 'tensorboard', 'tensorboardX','tensorflow == 2.12.0','torch == 2.0.1', 'pympler', 'protobuf==3.19.4'
]

test_requires = []

dev_requires = test_requires + ['pre-commit']

org_requires = ['paramiko==2.11.0', 'celery[redis]', 'cmd2']

benchmark_hpo_requires = [
    'configspace==0.5.0', 'hpbandster==0.7.4', 'smac==1.3.3', 'optuna==2.10.0'
]

benchmark_htl_requires = ['learn2learn']

with open("federatedscope/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=__name__,
    version=__version__,
    author="Alibaba Damo Academy",
    author_email="jones.wz@alibaba-inc.com",
    description="Federated learning package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=['deep-learning', 'federated-learning', 'benchmark'],
    packages=[
        package for package in setuptools.find_packages()
        if package.startswith(__name__)
    ],
    install_requires=minimal_requires,
    extras_require={
        'test': test_requires,
        'org': org_requires,
        'dev': dev_requires,
        'hpo': benchmark_hpo_requires,
        'htl': benchmark_htl_requires,
    },
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
