from setuptools import setup, find_packages

long_description = '''
Headliner is a library that was internally created to generate headlines from news articles. 
In particular, we use sequence-to-sequence (seq2seq) under the hood, 
an encoder-decoder framework. We provide a very simple interface to train 
and deploy seq2seq models. Although this library was created internally to 
generate headlines, you can also use it for other tasks like machine translations,
text summarization and many more.

Read the documentation at: https://as-ideas.github.io/headliner/

Headliner is compatible with Python 3.6 and is distributed under the MIT license.
'''

setup(
    name='headliner',
    version='0.0.14',
    author='Christian Schäfer',
    author_email='c.schaefer.home@gmail.com',
    description='Generating headlines from news articles using seq2seq models.',
    long_description=long_description,
    license='MIT',
    install_requires=['scikit-learn', 'nltk', 'pyyaml'],
    extras_require={
        'tests': ['pytest', 'pytest-cov', 'codecov', 'tensorflow~=2.0.0'],
        'docs': ['mkdocs', 'mkdocs-material'],
        'dev': ['bumpversion']
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