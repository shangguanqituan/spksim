REPO_ID = "sgqt2369144677/speaker-similarity-models" 

MODEL_CONFIGS = {
    "wavlm-base-plus-sv": {
        "repo_id": REPO_ID,
        "onnx_filename": "wavlm-base-plus-sv.onnx",
        "class": "WavLM",
        "feature_extractor_repo": REPO_ID,
    },
    "wespeaker-resnet34": {
        "repo_id": REPO_ID,
        "onnx_filename": "wespeaker_resnet34_LM.onnx",
        "class": "WeSpeaker",
    },
    "resemblyzer": {
        "repo_id": REPO_ID,
        "onnx_filename": "resemblyzer_voice_encoder.onnx",
        "class": "Resemblyzer",
    },
}