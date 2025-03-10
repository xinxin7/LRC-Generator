from setuptools import setup, find_packages

setup(
    name="lrc-generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "librosa>=0.9.0",
        "numpy>=1.20.0",
        "soundfile>=0.10.3",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "lrc-gen=lrc_generator.cli:main",
        ],
    },
    author="Xinxin",
    author_email="xinxingu9@outlook.com",
    description="A tool to generate synchronized lyrics (.lrc) files for MP3 songs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lrc-generator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)