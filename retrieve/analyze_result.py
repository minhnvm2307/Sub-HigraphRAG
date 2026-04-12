import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def _format_triple(triple, with_score=True):
    if with_score and len(triple) >= 4:
        h, r, t, score = triple[:4]
        return f"({h}) --[{r}]--> ({t})  score={score:.6f}"

    h, r, t = triple[:3]
    return f"({h}) --[{r}]--> ({t})"


def export_to_text(input_path, output_path=None, max_triples=None):
    data = torch.load(input_path, map_location="cpu")

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_readable.txt"

    num_questions = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample_id, sample in data.items():
            num_questions += 1
            question = sample.get("question", "")
            scored_triples = sample.get("scored_triples", [])

            if max_triples is not None:
                scored_triples = scored_triples[:max_triples]

            f.write(f"ID: {sample_id}\n")
            f.write(f"Query: {question}\n")
            f.write("Retrieved Triples:\n")

            if not scored_triples:
                f.write("  [No triples retrieved]\n")
            else:
                for idx, triple in enumerate(scored_triples, start=1):
                    f.write(f"  {idx}. {_format_triple(triple, with_score=True)}\n")

            f.write("\n")

    return output_path, num_questions, data


# =========================
# 🔥 ANALYSIS PART
# =========================

def analyze_by_rank(data, input_path, max_k=200):
    save_dir = os.path.splitext(input_path)[0] + "_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # collect scores theo rank
    rank_scores = [[] for _ in range(max_k)]
    high = 0
    medium = 0
    low = 0
    high_th = 0.2
    low_th = 0.01

    pairwise_diffs = []
    unique_counts = []
    entropies = []

    for sample in data.values():
        triples = sample.get("scored_triples", [])
        sample_scores = []

        for i in range(min(len(triples), max_k)):
            triple = triples[i]
            if len(triple) >= 4:
                score = float(triple[3])
                rank_scores[i].append(score)
                sample_scores.append(score)

                if score > high_th:
                    high += 1
                elif score > low_th:
                    medium += 1
                else:
                    low += 1

        # ===== 1. Pairwise difference =====
        if len(sample_scores) >= 2:
            scores_arr = np.asarray(sample_scores, dtype=float)
            diff_matrix = np.abs(scores_arr[:, None] - scores_arr[None, :])
            mean_diff = diff_matrix.mean()
        else:
            mean_diff = 0.0
        pairwise_diffs.append(mean_diff)

        # ===== 3. Entropy =====
        if len(sample_scores) > 0:
            scores_arr = np.asarray(sample_scores, dtype=float)
            hist, _ = np.histogram(scores_arr, bins=min(20, len(scores_arr)), density=False)
            total_hist = hist.sum()
            if total_hist > 0:
                probs = hist.astype(float) / total_hist
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log(probs))
            else:
                entropy = 0.0
            unique_counts.append(len(np.unique(scores_arr)))
        else:
            entropy = 0.0
            unique_counts.append(0)
        entropies.append(entropy)

    total = high + medium + low

    # compute stats
    mean_scores = []

    for scores in rank_scores:
        if len(scores) == 0:
            mean_scores.append(0)
        else:
            scores = np.array(scores)
            mean_scores.append(scores.mean())

    # ===== PLOT 1: Mean score vs rank =====
    plt.figure()
    plt.plot(range(1, max_k + 1), mean_scores)
    plt.title("Mean Score vs Rank")
    plt.xlabel("Rank")
    plt.ylabel("Mean Score")
    plt.savefig(os.path.join(save_dir, "mean_score_vs_rank.png"))
    plt.close()

    print("\n=== SCORE SEGMENTATION ===")
    if total > 0:
        print(f"High (> {high_th}): {high} ({high/total:.2%})")
        print(f"Medium ({low_th} - {high_th}): {medium} ({medium/total:.2%})")
        print(f"Near-zero (<= {low_th}): {low} ({low/total:.2%})")
    else:
        print(f"High (> {high_th}): {high} (0.00%)")
        print(f"Medium ({low_th} - {high_th}): {medium} (0.00%)")
        print(f"Near-zero (<= {low_th}): {low} (0.00%)")

    # Pie chart
    labels = [
        f"> {high_th}",
        f"{low_th} - {high_th}",
        f"<= {low_th}",
    ]
    sizes = [high, medium, low]

    plt.figure()
    plt.pie(sizes, labels=labels, autopct="%1.2f%%")
    plt.title("Score Distribution Segments")
    save_path = os.path.join(save_dir, "score_pie_chart.png")
    plt.savefig(save_path)
    plt.close()

    print("\n=== SCORE DIVERSITY ===")
    print(f"Avg pairwise diff: {np.mean(pairwise_diffs):.6f}")
    print(f"Avg unique scores: {np.mean(unique_counts):.2f}")
    print(f"Avg entropy: {np.mean(entropies):.6f}")

    # ===== Plot 1: Pairwise diff =====
    plt.figure()
    plt.hist(pairwise_diffs, bins=50)
    plt.title("Pairwise Score Difference per Query")
    plt.xlabel("Mean |si - sj|")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "pairwise_diff_hist.png"))
    plt.close()

    # ===== Plot 3: Entropy =====
    plt.figure()
    plt.hist(entropies, bins=50)
    plt.title("Score Entropy per Query")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "entropy_hist.png"))
    plt.close()

    print(f"Saved rank-based analysis to: {save_dir}")

    
# =========================
# MAIN
# =========================

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to retrieval_result.pth",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional output text file path",
    )
    parser.add_argument(
        "--max_triples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run score analysis",
    )

    args = parser.parse_args()

    output_path, num_questions, data = export_to_text(
        input_path=args.path,
        output_path=args.output,
        max_triples=args.max_triples,
    )

    print(f"Saved readable retrieval results to: {output_path}")
    print(f"Total queries exported: {num_questions}")

    if args.analyze:
        analyze_by_rank(data, args.path, max_k=args.max_triples or 400)