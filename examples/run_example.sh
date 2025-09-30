#!/bin/bash
set -e

echo "Running speaker similarity scoring example..."
echo "Model: WavLM"

spksim examples/meta.txt examples/synthesized_audio/ \
    -m local_models/wavlm-base-plus-sv.onnx \
    -t WavLM \
    -f local_models/wavlm-feature-extractor/ \
    -o examples/results.csv

echo ""
echo "Example run finished successfully!"
echo "Scores have been saved to: examples/results.csv"