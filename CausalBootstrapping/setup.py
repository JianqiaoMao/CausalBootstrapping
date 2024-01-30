from setuptools import setup

setup(
    name='CausalBootstrapping',
    version='0.0.1',
    author='Jianqiao Mao',
    author_email='jxm1417@student.bham.ac.uk',
    license='GPL-3.0',
    description="CausalBootstrapping is an easy-access implementation and extention of causal bootstrapping (CB) technique for causal analysis. With certain input of observational data, causal graph and variable distributions, CB resamples the data by adjusting the variable distributions which follow intended causal effects.",
    url='https://github.com/JianqiaoMao/CausalBootstrapping',
    #packages=['PFFRA'],
    install_requires=['numpy', 'grapl-causal'],
)