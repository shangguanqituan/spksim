# spksim

A batch scoring tool for speaker similarity evaluation.

[](https://opensource.org/licenses/Apache-2.0)

`spksim` is a command-line tool designed for standardized and reproducible speaker similarity evaluation, particularly for Text-to-Speech (TTS) and Voice Cloning applications. It operates on a batch of audio files defined in a meta file and uses ONNX-based models to ensure consistent results across different environments.

-----

## Features

  * **Batch Processing:** Optimized for evaluating large sets of synthesized audio against reference audio.
  * **Flexible Model Loading:** Supports both easy-to-use remote models from Hugging Face Hub and local ONNX files for offline use and development.
  * **Standardized & Reproducible:** Uses ONNX Runtime to eliminate discrepancies between deep learning frameworks.
  * **Simple Interface:** Provides a clean command-line interface with concise aliases for frequently used options.
  * **Extensible:** Designed to be easily extended with new custom speaker similarity models.

## Installation
Install the package and its dependencies. Using the editable (`-e`) flag is recommended for development.
```bash
pip install -e .
```

## Usage

The tool is run from the command line, requiring a meta file, an audio directory, and options to specify the model.

### Command Syntax

```bash
spksim <meta_file> <synth_dir> [MODEL_OPTIONS] [OTHER_OPTIONS]
```

### Arguments

  * **Positional Arguments:**

      * `meta_file`: Path to the metadata file that defines the audio pairs to compare.
      * `synth_dir`: Path to the directory containing the synthesized audio files.

  * **Model Selection (Required - choose one):**

      * `-r, --remote-model [NAME]`: Use a pre-configured model from Hugging Face Hub.
      * `-m, --local-model-path [PATH]`: Use a local `.onnx` model file.

  * **Local Model Options:**

      * `-t, --model-type [TYPE]`: Required when using `-m`. Specifies the model type. Choices: `WavLM`, `WeSpeaker`, `Resemblyzer`.
      * `-f, --feature-extractor-path [PATH]`: Required when using a local `WavLM` model. Path to the feature extractor directory.

  * **Output Options:**

      * `-o, --output-file [PATH]`: (Optional) Path to save the results in a CSV file.

### Examples

**Example 1: Using a Remote Model (Recommended)**

This is the easiest way to run an evaluation. The tool will automatically download and cache the specified model.

```bash
spksim meta.txt ./synth_audio/ -r wavlm-base-plus-sv -o results.csv
```

**Example 2: Using a Local WavLM Model**

This gives you full control for offline use or testing custom models.

```bash
spksim meta.txt ./synth_audio/ \
    -m local_models/wavlm.onnx \
    -t WavLM \
    -f local_models/wavlm-extractor/ \
    -o results.csv
```

## Quick Start Example

The repository includes a `run_example.sh` script to demonstrate a typical batch scoring workflow.

**1. Setup:**
Before running, ensure your audio and model files are placed in the correct directories at the project root:

  * **Audio Files:** Place your `.wav` files inside the `examples/` subdirectories (`synthesized_audio/` and `reference_audio/`).
  * **Model Files:** Place your `.onnx` model and feature extractor files inside the `local_models/` directory.

**2. Run:**
Once the files are in place, execute the example script from the project's root directory:

```bash
bash examples/run_example.sh
```

**3. Verify:**
The script will print a summary to the console and save a detailed report to `examples/results.csv`.

## Meta File Format

The meta file is a plain text file where each line defines a pair of audios to be compared. The format for each line is:

`utt_id|reference_wav_path`

  * `utt_id`: A unique identifier for the utterance. The tool will look for a corresponding synthesized audio file at `<synth_dir>/<utt_id>.wav`.
  * `|`: A pipe character used as a separator.
  * `reference_wav_path`: The full or relative path to the ground truth reference audio file.

**Example `meta.txt` content:**

```
speakerA_001|/path/to/references/speakerA/001.wav
speakerA_002|/path/to/references/speakerA/002.wav
```

## Adding a New Model

To extend the tool with a new speaker model:

1.  Export the desired model to the ONNX format.
2.  In `spksim/core.py`, create a new class that inherits from `SpeakerSimilarityModel`.
3.  Implement the `extract_features` method within the new class to handle its specific audio preprocessing.
4.  In `spksim/manager.py`, add the new model's name and class to the `model_class_map` dictionary for local mode support.
5.  (Optional) To add remote support, update the `spksim/configs.py` file and upload the model to the designated Hugging Face Hub repository.

## License

This project is licensed under the Apache 2.0 License.