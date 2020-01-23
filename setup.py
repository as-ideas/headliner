from setuptools import setup, find_packages

long_description = '''
Headliner is a sequence modeling library that eases the training and 
**in particular, the deployment of custom sequence models** for both researchers and developers. 
You can very easily deploy your models in a few lines of code. It was originally 
built for our own research to generate headlines from news articles. 
That's why we chose the name, Headliner. Although this library was created internally to 
generate headlines, you can also use it for other tasks like machine translations,
text summarization and many more.

Read the documentation at: https://as-ideas.github.io/headliner/

Headliner is compatible with Python 3.6+ and is distributed under the MIT license.
'''

setup(
    name='headliner',
    version='1.0.0',
    author='Christian Schäfer',
    author_email='c.schaefer.home@gmail.com',
    description='Easy training and deployment of seq2seq models.',
    long_description=long_description,
    license='MIT',
    install_requires=['scikit-learn', 'nltk', 'pyyaml', 'transformers', 'spacy'],
    extras_require={
        'tests': ['pytest', 'pytest-cov', 'codecov', 'tensorflow==2.0.0'],
        'docs': ['mkdocs', 'mkdocs-material'],
        'dev': ['bumpversion']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('tests',)),
)
