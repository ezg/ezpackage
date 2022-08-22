from setuptools import setup

setup(
    name='ezpackage',
    version='0.1.0',
    description='A example Python package',
    url='https://github.com/shuds13/pyexample',
    author='Emanuel Zgraggen',
    author_email='emanuel.zgraggen@gmail.com',
    license='BSD 2-clause',
    packages=['ezpackage'],
    install_requires=[
        'numpy',
    ],

    classifiers=[
        'ezpackage',
    ],
)
