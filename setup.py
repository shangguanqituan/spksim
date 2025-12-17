from setuptools import setup, find_packages

setup(
    name="spksim",
    version="0.4.0",
    packages=find_packages(),
    install_requires=[
        "onnxruntime",
        "torch",
        "torchaudio",
        "transformers",
        "torchcodec",
        "librosa",
        "numpy",
        "tqdm",
        "huggingface_hub",
    ],
    entry_points={
        'console_scripts': [
            'spksim=spksim.cli:main',
        ],
    },
)