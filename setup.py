"""
Setup script for the Edumate package
"""

from setuptools import setup, find_packages

setup(
    name="edumate",
    version="0.1.0",
    description="AI-powered educational platform",
    author="Edumate Team",
    author_email="info@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.25.0",
        "pandas>=1.5.0",
        "plotly>=5.18.0",
        "langchain>=0.0.335",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "edumate=edumate.app:main",
        ],
    },
) 