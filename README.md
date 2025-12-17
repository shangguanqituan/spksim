# spksim

A flexible batch scoring tool for speaker similarity evaluation.

[](https://opensource.org/licenses/Apache-2.0)

`spksim` is a command-line tool designed for standardized and reproducible speaker similarity evaluation. It supports various state-of-the-art models (WeSpeaker, WavLM, Resemblyzer) via ONNX Runtime, ensuring consistent results across different environments.

It is ideal for evaluating Voice Conversion (VC), Text-to-Speech (TTS), or generic Speaker Verification tasks.

-----

## Features

  * **Config-Driven:** Manage all your remote (Hugging Face) and local models in a central `config.py`.
  * **Unified Interface:** Switch between different models (e.g., ResNet, ECAPA-TDNN, WavLM) using a simple model name.
  * **Flexible Input:** Supports both absolute paths and relative paths with a base directory.
  * **Batch Processing:** Optimized for evaluating large sets of audio pairs.
  * **Standardized:** Uses ONNX Runtime to eliminate discrepancies between deep learning frameworks.

## Installation

Install the package and its dependencies in editable mode:

```bash
pip install -e .
```

## Configuration & Supported Models

Models are defined in `spksim/config.py`. You can easily add new models by editing this file.

### Currently Configured Models

**WeSpeaker Series (FBank based):**

**ResNet Family:**

* wespeaker-resnet34 / wespeaker-resnet34-lm: Standard ResNet34 (VoxCeleb).

* wespeaker-cnceleb-resnet34 / wespeaker-cnceleb-resnet34-lm: ResNet34 trained on CnCeleb (Chinese).

* wespeaker-resnet152-lm / wespeaker-resnet221-lm / wespeaker-resnet293-lm: Large ResNet models for higher accuracy.

* wespeaker-dfresnet114-gemini: DF-ResNet114 (Gemini).

**CAM++ & ECAPA-TDNN:**

* wespeaker-campplus / wespeaker-campplus-lm: CAM++ model (Efficient & Strong).

* wespeaker-ecapa-tdnn512 / wespeaker-ecapa-tdnn512-lm: ECAPA-TDNN (512 channels).

* wespeaker-ecapa-tdnn1024 / wespeaker-ecapa-tdnn1024-lm: ECAPA-TDNN (1024 channels).

(Note: Models with -lm suffix usually include Large Margin loss optimization)

**WavLM Series:**

* `wavlm-base-plus-sv`: WavLM Base+ model fine-tuned for Speaker Verification.

**Resemblyzer:**

* `resemblyzer`: Resemblyzer voice encoder.

*Tip: Check `spksim/config.py` to see the full list of available model keys or to add your own local models.*

## Usage

The tool is run from the command line using the `spksim` command.

### Command Syntax

```bash
spksim <meta_file> [audio_dir] -m <model_name> [options]
```

### Arguments

* **Positional Arguments:**
* `meta_file`: Path to the file defining the audio pairs to compare.
* `audio_dir` (Optional): Base directory for audio files. If provided, relative paths in the meta file will be resolved against this directory.


* **Required Options:**
* `-m, --model`: The name of the model to use (must be a key defined in `config.py`, e.g., `wespeaker-resnet34`).


* **Output Options:**
* `-o, --output-file`: Path to save the results in a CSV file.



### Examples

**Example 1: Basic Usage (Absolute Paths)**
If your `meta.txt` contains absolute paths, you don't need the audio directory argument.

```bash
spksim data/meta_absolute.txt -m wespeaker-campplus -o results.csv
```

**Example 2: Using a Base Directory (Relative Paths)**
If your `meta.txt` contains relative paths (filenames), provide the folder where they are located.

```bash
spksim data/pairs.txt /home/data/audio_corpus/ -m wavlm-base-plus-sv
```

**Example 3: Switching Models**
Testing the same dataset with a different model is super easy:

```bash
spksim data/pairs.txt /home/data/audio_corpus/ -m resemblyzer -o results_resemblyzer.csv
```

## Meta File Format

The meta file is a plain text file. Each line defines a pair of audio files to be compared (Candidate vs Reference).

**Format:**

```text
Candidate_Path | Reference_Path
```

* **Candidate_Path**: The audio you want to evaluate (e.g., synthesized/converted speech).
* **Reference_Path**: The ground truth or target speaker audio.
* **Separator**: A pipe character `|`.

**Example Content (`meta.txt`):**

```text
# Absolute paths (use without audio_dir)
/data/gen/001.wav|/data/ref/speaker_A.wav

# Relative paths (use with audio_dir)
gen/002.wav|ref/speaker_B.wav
```

## Adding Custom Models

To add a new model (e.g., a private ONNX file):

1. Open `spksim/config.py`.
2. Add a new entry to `MODEL_CONFIGS`:

```python
"my-local-model": {
    "source": "local",
    "path": "/abs/path/to/model.onnx",
    "class": "WeSpeaker",  # or "WavLM", "Resemblyzer"
}
```

3. Run with `-m my-local-model`.

## License

This project is licensed under the Apache 2.0 License.