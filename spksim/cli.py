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
                        help="Path to the meta file. Format: 'Candidate_Path|Reference_Path'")

    # 把 audio_dir 设为可选，如果 meta 文件里都是绝对路径，就不需要这个了
    parser.add_argument('audio_dir', nargs='?', default=None,
                        help="(Optional) Base directory for audio files if meta file uses relative paths.")

    # --- Model Selection ---
    # 现在只需要一个参数，直接填 config.py 里定义的名字
    parser.add_argument('-m', '--model', type=str, required=True, choices=MODEL_CONFIGS.keys(),
                        help="Name of the model to use (defined in config.py).")

    # --- Output ---
    parser.add_argument('-o', '--output-file', type=str,
                        help="Path to save the results in a CSV file.")

    args = parser.parse_args()

    # --- Model Loading ---
    manager = ModelManager()
    try:
        # 直接通过名字加载，Manager 会自己去查表知道它是 WeSpeaker 还是 WavLM
        model = manager.get_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Batch Processing ---
    scores = []
    try:
        with open(args.meta_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Meta file not found at {args.meta_file}", file=sys.stderr)
        sys.exit(1)

    for line in tqdm(lines, desc="Processing pairs"):
        line = line.strip()
        if not line or '|' not in line:
            continue

        try:
            # 现在我们认为左边是 Candidate (待测)，右边是 Reference (参考)
            path_a_str, path_b_str = line.split('|', 1)
            path_a_str = path_a_str.strip()
            path_b_str = path_b_str.strip()

            # 辅助函数：处理路径（如果是相对路径，且提供了 audio_dir，则拼接）
            def resolve_path(p):
                if args.audio_dir and not os.path.isabs(p):
                    return os.path.join(args.audio_dir, p)
                return p

            cand_path = resolve_path(path_a_str)
            ref_path = resolve_path(path_b_str)

            # 检查文件是否存在
            if not os.path.exists(cand_path):
                print(f"\nWarning: Candidate file not found: {cand_path}", file=sys.stderr)
                continue
            if not os.path.exists(ref_path):
                print(f"\nWarning: Reference file not found: {ref_path}", file=sys.stderr)
                continue

            # 提取特征并计算
            # 注意：utt_id 现在不一定有了，我们可以用文件名或者路径作为标识
            utt_id = os.path.basename(cand_path)

            features1 = model.extract_features(cand_path)
            features2 = model.extract_features(ref_path)
            emb1 = model.infer(features1)
            emb2 = model.infer(features2)
            similarity = model.compute_similarity(emb1, emb2)

            scores.append({'utt_id': utt_id, 'score': similarity, 'ref': os.path.basename(ref_path)})

        except Exception as e:
            print(f"\nError processing line '{line}': {e}", file=sys.stderr)

    # --- Summary & Save (保持原有逻辑，稍作调整) ---
    if not scores:
        print("No valid pairs processed.", file=sys.stderr)
        return

    avg_score = sum(s['score'] for s in scores) / len(scores)
    print(f"\nDone! Processed {len(scores)} pairs. Average Score: {avg_score:.4f}")

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write("candidate,reference,score\n") # 稍微改一下表头更清晰
            for item in scores:
                f.write(f"{item['utt_id']},{item['ref']},{item['score']:.6f}\n")
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()