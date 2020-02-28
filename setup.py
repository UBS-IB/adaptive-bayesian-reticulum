from setuptools import setup

version = {}
with open('reticulum/version.py') as f:
    exec(f.read(), version)

with open('requirements.txt') as f:
    requirements = f.readlines()

with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='adaptive-bayesian-reticulum',
    version=version['__version__'],
    description='An implementation of the paper https://arxiv.org/abs/1912.05901 by Nuti et al.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='UBS AG',
    author_email='giuseppe.nuti@ubs.com, lluis.jimenez-rugama@ubs.com, kaspar.thommen@ubs.com',
    license='Apache License 2.0',
    url='https://github.com/UBS-IB/adaptive-bayesian-reticulum',
    packages=['reticulum', 'examples'],
    install_requires=requirements,
    keywords='Bayesian decision tree neural network',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
