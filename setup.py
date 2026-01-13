from setuptools import setup, find_packages
import os

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="arjun-mtl",
    version="0.1.0",
    author="Arjun Team",
    description="A Parameter-Efficient Multi-Objective Pre-Training Framework for Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pidahbus/multi-task-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
    },
    keywords="multi-task-learning, nlp, transformer, language-model, deep-learning, tensorflow",
    project_urls={
        "Bug Reports": "https://github.com/pidahbus/multi-task-learning/issues",
        "Source": "https://github.com/pidahbus/multi-task-learning",
    },
)
