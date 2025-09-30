from setuptools import setup, find_packages

setup(
    name="spksim",
    version="0.4.0", # 再次更新版本号，代表重要功能变更
    packages=find_packages(),
    install_requires=[
        "onnxruntime",
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "numpy",
        "tqdm",
        "huggingface_hub", # <-- 重新添加这一行
    ],
    entry_points={
        'console_scripts': [
            'spksim=spksim.cli:main',
        ],
    },
)