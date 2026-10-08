import argparse
from metrics.evaluate_results_corrected import eval_results as eval_results_corrected

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on predictions.jsonl file")
    parser.add_argument("predict_file", type=str, help="Path to predictions.jsonl file")
    parser.add_argument("--subset", action="store_true", default=True, help="Evaluate only on subset (ground truth in graph)")
    parser.add_argument("--no_subset", action="store_false", dest="subset", help="Evaluate on full dataset")
    parser.add_argument("--split", type=str, default=None, help="Answer splitting delimiter (default: None)")
    parser.add_argument("--eval_hops", type=int, default=-1, help="Filter evaluation by hops (-1 for all)")

    args = parser.parse_args()

    print(f"=== Standalone Evaluation ===")
    print(f"File: {args.predict_file}")
    print(f"Subset: {args.subset}")
    print(f"Split delimiter: {args.split}")
    print(f"Eval Hops: {args.eval_hops}")
    print("=============================")

    eval_results_corrected(
        predict_file=args.predict_file,
        cal_f1=True,
        split=args.split,
        subset=args.subset,
        eval_hops=args.eval_hops
    )

if __name__ == "__main__":
    main()
