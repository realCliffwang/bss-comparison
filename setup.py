"""Setup script for BSS-Test package."""

from setuptools import setup, find_packages

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bss-test",
    version="0.2.0",
    author="BSS-Test Contributors",
    description="Bearing Fault Diagnosis with Blind Source Separation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bss-test",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "extended": [
            "EMD-signal>=1.6.0",
            "vmdpy>=0.1",
            "python-picard>=0.8",
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "torch": [
            "torch>=2.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    entry_points={
        "console_scripts": [
            "bss-test-cwru=main_cwru:main",
            "bss-test-phm=main_phm_milling:main",
            "bss-test-nasa=main_nasa_milling:main",
            "bss-test-compare=generate_comparison_report:main",
        ],
    },
)
