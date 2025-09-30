import librosa
import numpy as np

def load_audio(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """加载音频并统一采样率"""
    y, sr = librosa.load(wav_path, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y