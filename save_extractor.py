from transformers import Wav2Vec2FeatureExtractor
import os

# --- 配置 ---
# Hugging Face上的模型名称
MODEL_ID = "/home/sgqt/wavelearning/wavlm-base-plus-sv"
# 你想把文件保存到本地的哪个文件夹
OUTPUT_DIR = "./local_models/wavlm-feature-extractor"

# --- 执行 ---
print(f"Downloading feature extractor: {MODEL_ID}")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 从网络加载特征提取器
try:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)

    # 2. 将其所有相关文件保存到本地目录
    feature_extractor.save_pretrained(OUTPUT_DIR)
    
    print("-" * 30)
    print(f"Success! Feature extractor files have been saved to: {OUTPUT_DIR}")
    print("You can now use this path during runtime.")

except Exception as e:
    print(f"Download or save failed: {e}")
    print("Please check your network connection and model name for correctness.")