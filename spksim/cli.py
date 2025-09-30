# spksim/cli.py

import argparse
import os
import sys
from tqdm import tqdm
from .manager import ModelManager
from .config import MODEL_CONFIGS

def main():
    parser = argparse.ArgumentParser(
        description="A batch scoring tool for speaker similarity evaluation.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Positional Arguments ---
    parser.add_argument('meta_file', 
                        help="Path to the meta file (format: utt_id|ref_wav_path).")
    parser.add_argument('synth_dir', 
                        help="Directory containing the synthesized audio files.")

    # --- Model Selection Arguments (mutually exclusive) ---
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('-r', '--remote-model', type=str, choices=MODEL_CONFIGS.keys(),
                             help="Name of the model to use from Hugging Face Hub.")
    model_group.add_argument('-m', '--local-model-path', type=str,
                             help="Path to a local ONNX model file.")

    # --- Additional Arguments for Local Mode ---
    parser.add_argument('-t', '--model-type', type=str, choices=['WavLM', 'WeSpeaker', 'Resemblyzer'],
                        help="Type of the local model (required if using -m/--local-model-path).")
    parser.add_argument('-f', '--feature-extractor-path', type=str,
                        help="Path to the local feature extractor directory (for local WavLM model).")

    # --- Output Argument ---
    parser.add_argument('-o', '--output-file', type=str,
                        help="Path to save the results in a CSV file.")

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.local_model_path and not args.model_type:
        parser.error("--model-type (-t) is required when using --local-model-path (-m).")
    if args.model_type == 'WavLM' and args.local_model_path and not args.feature_extractor_path:
        parser.error("--feature-extractor-path (-f) is required when using a local WavLM model.")
    
    # --- Model Loading ---
    manager = ModelManager()
    print("Loading model...", file=sys.stderr)
    
    try:
        if args.remote_model:
            model = manager.get_model_from_hub(args.remote_model)
        else: # If remote_model is not used, local_model_path must have been
            model = manager.get_model_from_local_paths(
                model_type=args.model_type,
                model_path=args.local_model_path,
                feature_extractor_path=args.feature_extractor_path
            )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Model loaded.", file=sys.stderr)

    # --- Batch Processing ---
    scores = []
    lines = []
    try:
        with open(args.meta_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Meta file not found at {args.meta_file}", file=sys.stderr)
        sys.exit(1)

    for line in tqdm(lines, desc="Processing audio pairs"):
        line = line.strip()
        if not line or '|' not in line:
            continue
        
        try:
            utt_id, ref_path = line.split('|', 1)
            synth_path = os.path.join(args.synth_dir, f"{utt_id}.wav")

            if not os.path.exists(ref_path):
                print(f"\nWarning: Reference file not found, skipping pair '{utt_id}': {ref_path}", file=sys.stderr)
                continue
            if not os.path.exists(synth_path):
                print(f"\nWarning: Synthesized file not found, skipping pair '{utt_id}': {synth_path}", file=sys.stderr)
                continue
            
            features1 = model.extract_features(synth_path)
            features2 = model.extract_features(ref_path)
            emb1 = model.infer(features1)
            emb2 = model.infer(features2)
            similarity = model.compute_similarity(emb1, emb2)
            scores.append({'utt_id': utt_id, 'score': similarity})

        except Exception as e:
            print(f"\nError processing line '{line}': {e}", file=sys.stderr)

    # --- Results Summary ---
    if not scores:
        print("No valid pairs were processed. Please check your meta file and audio paths.", file=sys.stderr)
        return

    average_score = sum(item['score'] for item in scores) / len(scores)
    
    print("\n" + "=" * 40)
    print("Batch Scoring Summary")
    print(f"  Total pairs processed successfully: {len(scores)}")
    print(f"  Average Similarity Score: {average_score:.6f}")
    print("=" * 40 + "\n")

    # --- Save to File ---
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write("utt_id,score\n")
                for item in scores:
                    f.write(f"{item['utt_id']},{item['score']:.6f}\n")
            print(f"Results saved to {args.output_file}")
        except IOError as e:
            print(f"Error writing to output file {args.output_file}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
spksim examples/meta.txt examples/synthesized_audio/ \
    -m local_models/wavlm-base-plus-sv.onnx \
    -t WavLM \
    -f local_models/wavlm-feature-extractor/ \
    -o examples/results.csv

spksim examples/meta.txt examples/synthesized_audio/ \
     -r wavlm-base-plus-sv -o examples/results_remote.csv
"""