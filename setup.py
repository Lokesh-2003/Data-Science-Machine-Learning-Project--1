from setuptools import setup, find_packages

setup(
    name="fraud_detection",
    version="1.0.0",
    description="Real-time fraud detection system",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'imbalanced-learn>=0.8.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'streamlit>=1.0.0',
        'joblib>=1.0.0',
        'pyyaml>=6.0',
        'pytest>=7.0.0'
    ],
    python_requires='>=3.8',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'fraud-detection-train=src.main:main'
        ]
    },
)