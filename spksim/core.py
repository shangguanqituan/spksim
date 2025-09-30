from abc import ABC, abstractmethod
from typing import Dict
import librosa

import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from transformers import Wav2Vec2FeatureExtractor

from .utils import load_audio


class SpeakerSimilarityModel(ABC):
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    @abstractmethod
    def extract_features(self, wav_path: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def infer(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        embeddings = self.session.run(self.output_names, features)
        return embeddings[0]

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        # 转换为torch tensor并进行L2归一化，这是标准做法
        emb1 = torch.nn.functional.normalize(torch.from_numpy(emb1), p=2, dim=-1)
        emb2 = torch.nn.functional.normalize(torch.from_numpy(emb2), p=2, dim=-1)

        similarity = torch.nn.CosineSimilarity(dim=-1)(emb1, emb2)
        return similarity.item()


class WavLM(SpeakerSimilarityModel):
    def __init__(self, model_path: str, feature_extractor_repo: str):
        super().__init__(model_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_repo)

    def extract_features(self, wav_path: str) -> Dict[str, np.ndarray]:
        audio = load_audio(wav_path, target_sr=self.feature_extractor.sampling_rate)
        
        inputs = self.feature_extractor(
            audio, 
            padding=True, 
            return_tensors="pt", 
            sampling_rate=self.feature_extractor.sampling_rate
        )
        
        onnx_inputs = {}
        # 遍历ONNX模型需要的所有输入名称
        for name in self.input_names:
            if name in inputs:
                numpy_array = inputs[name].cpu().numpy()
                
                if name == 'attention_mask':
                    onnx_inputs[name] = numpy_array.astype(np.int64)
                else:
                    onnx_inputs[name] = numpy_array
        
        return onnx_inputs


class WeSpeaker(SpeakerSimilarityModel):
    def extract_features(self, wav_path: str) -> Dict[str, np.ndarray]:
        feats = self._compute_fbank(wav_path)
        feats = np.expand_dims(feats, axis=0)  # 增加批次维度
        # 假设WeSpeaker模型的输入名叫'feats'
        return {self.input_names[0]: feats}

    def _compute_fbank(self, wav, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0):
        waveform, sample_rate = torchaudio.load(wav)
        waveform = waveform * (1 << 15)
        mat = kaldi.fbank(
            waveform, num_mel_bins=num_mel_bins, frame_length=frame_length,
            frame_shift=frame_shift, dither=dither, sample_frequency=sample_rate,
            window_type='hamming', use_energy=False
        )
        mat = mat - torch.mean(mat, dim=0)
        return mat.cpu().numpy()


class Resemblyzer(SpeakerSimilarityModel):
    def extract_features(self, wav_path: str) -> Dict[str, np.ndarray]:
        audio = load_audio(wav_path, target_sr=16000)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=400, hop_length=160, n_mels=40
        )
        mel_spectrogram = mel_spectrogram.T.astype(np.float32)
        mel_input = np.expand_dims(mel_spectrogram, axis=0) # 增加批次维度
        return {self.input_names[0]: mel_input}