from setuptools import setup, find_packages

long_description = '''
'''

setup(
    name='headliner',
    version='0.0.1',
    author='Christian Schäfer',
    author_email='c.schaefer.home@gmail.com',
    description='',
    long_description=long_description,
    license='MIT',
    install_requires=['scikit-learn==0.21.3', 'nltk==3.4.5', 'pyyaml==5.1.2', 'tensorflow==2.0.0-beta1'],
    extras_require={
        'tests': ['pytest==4.3.0', 'pytest-cov==2.6.1'],
        'docs': ['mkdocs==1.0.4', 'mkdocs-material==4.0.2'],
        'gpu': ['tensorflow-gpu==2.0.0-beta1'],
        'dev': ['bumpversion==0.5.3'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('tests',)),
)