# spksim/config.py

# 用于 WavLM 和 Resemblyzer 的个人仓库 ID (保持不变)
MY_REPO_ID = "sgqt2369144677/speaker-similarity-models"

MODEL_CONFIGS = {
    # ============================
    # WavLM Series
    # ============================
    "wavlm-base-plus-sv": {
        "source": "remote",
        "repo_id": MY_REPO_ID,
        "onnx_filename": "wavlm-base-plus-sv.onnx",
        "class": "WavLM",
        "feature_extractor_repo": MY_REPO_ID,
    },

    # ============================
    # Resemblyzer
    # ============================
    "resemblyzer": {
        "source": "remote",
        "repo_id": MY_REPO_ID,
        "onnx_filename": "resemblyzer_voice_encoder.onnx",
        "class": "Resemblyzer",
    },

    # ============================
    # WeSpeaker Series (Official)
    # ============================

    # --- ResNet34 ---
    "wespeaker-resnet34": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-resnet34",
        "onnx_filename": "voxceleb_resnet34.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-resnet34-lm": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-resnet34-LM",
        "onnx_filename": "voxceleb_resnet34_LM.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-cnceleb-resnet34": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-cnceleb-resnet34",
        "onnx_filename": "cnceleb_resnet34.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-cnceleb-resnet34-lm": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-cnceleb-resnet34-LM",
        "onnx_filename": "cnceleb_resnet34_LM.onnx",
        "class": "WeSpeaker",
    },

    # --- CAM++ ---
    "wespeaker-campplus": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-campplus",
        "onnx_filename": "voxceleb_CAM++.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-campplus-lm": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-campplus-LM",
        "onnx_filename": "voxceleb_CAM++_LM.onnx",
        "class": "WeSpeaker",
    },

    # --- ECAPA-TDNN ---
    "wespeaker-ecapa-tdnn512": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-ecapa-tdnn512",
        "onnx_filename": "voxceleb_ECAPA512.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-ecapa-tdnn512-lm": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-ecapa-tdnn512-LM",
        "onnx_filename": "voxceleb_ECAPA512_LM.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-ecapa-tdnn1024": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024",
        "onnx_filename": "voxceleb_ECAPA1024.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-ecapa-tdnn1024-lm": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM",
        "onnx_filename": "voxceleb_ECAPA1024_LM.onnx",
        "class": "WeSpeaker",
    },

    # --- Large ResNets & Others ---
    "wespeaker-resnet152-lm": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-resnet152-LM",
        "onnx_filename": "voxceleb_resnet152_LM.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-resnet221-lm": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-resnet221-LM",
        "onnx_filename": "voxceleb_resnet221_LM.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-resnet293-lm": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-resnet293-LM",
        "onnx_filename": "voxceleb_resnet293_LM.onnx",
        "class": "WeSpeaker",
    },
    "wespeaker-dfresnet114-gemini": {
        "source": "remote",
        "repo_id": "Wespeaker/wespeaker-voxceleb-gemini-DFresnet114-LM",
        "onnx_filename": "voxceleb_gemini_dfresnet114_LM.onnx",
        "class": "WeSpeaker",
    },
}