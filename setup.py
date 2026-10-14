from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cesNMF",
    version="0.1.0",
    author="Yasmine Baca",
    author_email="ykorte@asu.edu",
    description=(
        "Covariate-Embedded Sparse NMF for bulk tumor CAF deconvolution"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ykorte-coder/cesNMF_code",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "scikit-learn>=1.0",
        "lifelines>=0.27",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        "NMF", "deconvolution", "CAF", "PDAC",
        "bulk RNA-seq", "tumor microenvironment",
        "cancer-associated fibroblasts",
        "covariate embedding",
    ],
)
