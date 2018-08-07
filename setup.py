import setuptools

setuptools.setup(
    name="cvutilities",
    version="0.0.1",
    author="Theodore Quinn",
    author_email="ted.quinn@wildflowerschools.org",
    license='MIT',
    description="Miscellaneous helper functions for fetching and processing OpenPose data and camera calibration data",
    url="https://github.com/WildflowerSchools/cvutilities",
    packages=setuptools.find_packages(),
    install_requires=[
        'opencv-python>=3.4.1',
        'numpy>=1.14',
        'scipy>=1.1',
        'pandas>=0.23',
        'matplotlib>=2.2',
        'networkx>=2.1'
        'boto3>=1.7',
        'python-dateutil>=2.7'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"))
