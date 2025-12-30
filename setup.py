from setuptools import setup, find_packages

setup(
    name='CausalBootstrapping',
    version='0.2.1',
    author='Jianqiao Mao',
    author_email='jxm1417@student.bham.ac.uk',
    license='GPL-3.0',
    description="CausalBootstrapping is an easy-access implementation and extension of causal bootstrapping (CB) technique for causal analysis. With certain input of observational data, causal graph and variable distributions, CB resamples the data by adjusting the variable distributions which follow intended causal effects.",
    url='https://github.com/JianqiaoMao/CausalBootstrapping',
    packages=find_packages(), 
    python_requires=">=3.9,<3.11",
    install_requires=['numpy ~= 1.25.0', 
                      'grapl-causal == 1.6.1', 
                      'scipy ~= 1.13.1', 
                      'graphviz == 0.20.3',
                      'sympy ~= 1.13.1'],
)