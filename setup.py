"""Setup script for GridFM package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="gridfm",
    version="1.0.0",
    author="GridFM Team",
    author_email="gridfm@example.com",
    description="A Physics-Informed Foundation Model for Multi-Task Energy Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GridFM/GridFM",
    project_urls={
        "Bug Tracker": "https://github.com/GridFM/GridFM/issues",
        "Documentation": "https://github.com/GridFM/GridFM/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "einops>=0.6.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
        "tracking": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "explain": [
            "shap>=0.42.0",
            "captum>=0.6.0",
        ],
        "all": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
            "shap>=0.42.0",
            "captum>=0.6.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gridfm-train=scripts.train:main",
            "gridfm-evaluate=scripts.evaluate:main",
            "gridfm-download=scripts.download_data:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
