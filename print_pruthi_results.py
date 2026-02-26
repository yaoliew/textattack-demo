"""
Print the same attack results summary as at the end of a run (metrics table +
confusion matrix), computed from a Pruthi results JSON file in attack_results.
"""

import argparse
import glob
import json
import os

from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def load_pruthi_results(path):
    """Load Pruthi results from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_from_json(entries):
    """
    Compute the same metrics as AttackMetricsTracker.get_metrics() from
    Pruthi JSON entries (each has original_label, original_text, perturbed_text,
    original_prediction, perturbed_prediction, result_type, successful, etc.).
    """
    num_successful = sum(1 for e in entries if e.get("successful") is True)
    num_failed = sum(1 for e in entries if e.get("result_type") == "FailedAttackResult")
    num_skipped = sum(
        1 for e in entries
        if e.get("result_type") == "SkippedAttackResult" or "error" in e
    )

    # Original accuracy: among entries with original_prediction
    with_orig_pred = [e for e in entries if "original_prediction" in e and e["original_prediction"] is not None]
    if with_orig_pred:
        original_correct = sum(1 for e in with_orig_pred if e["original_label"] == e["original_prediction"])
        original_accuracy = original_correct / len(with_orig_pred) * 100
    else:
        original_accuracy = 0.0

    # Accuracy under attack: among entries with perturbed_prediction (exclude errors)
    with_pert_pred = [e for e in entries if "perturbed_prediction" in e and e["perturbed_prediction"] is not None]
    if with_pert_pred:
        attacked_correct = sum(1 for e in with_pert_pred if e["original_label"] == e["perturbed_prediction"])
        attacked_accuracy = attacked_correct / len(with_pert_pred) * 100
    else:
        attacked_accuracy = 0.0

    total_attempted = num_successful + num_failed
    attack_success_rate = (num_successful / total_attempted * 100) if total_attempted > 0 else 0.0

    # Average perturbed word % (string-based fallback; no AttackedText in JSON)
    perturbed_pcts = []
    word_counts = []
    for e in entries:
        if "error" in e:
            continue
        orig = e.get("original_text") or ""
        pert = e.get("perturbed_text") or orig
        orig_words = orig.split()
        pert_words = pert.split()
        if not orig_words:
            continue
        word_counts.append(len(orig_words))
        min_len = min(len(orig_words), len(pert_words))
        num_changed = sum(1 for i in range(min_len) if orig_words[i] != pert_words[i])
        num_changed += abs(len(pert_words) - len(orig_words))
        pct = (num_changed / len(orig_words)) * 100
        perturbed_pcts.append(pct)
    avg_perturbed_word_pct = sum(perturbed_pcts) / len(perturbed_pcts) if perturbed_pcts else 0.0

    # Average words per input (from original_text)
    if not word_counts:
        word_counts = [len((e.get("original_text") or "").split()) for e in entries if "original_text" in e]
    avg_words_per_input = sum(word_counts) / len(word_counts) if word_counts else 0.0

    # Avg num queries: not stored in JSON, use 0.0 to match table format
    avg_num_queries = 0.0

    return {
        "num_successful": num_successful,
        "num_failed": num_failed,
        "num_skipped": num_skipped,
        "original_accuracy": original_accuracy,
        "attacked_accuracy": attacked_accuracy,
        "attack_success_rate": attack_success_rate,
        "avg_perturbed_word_pct": avg_perturbed_word_pct,
        "avg_words_per_input": avg_words_per_input,
        "avg_num_queries": avg_num_queries,
    }


def confusion_matrix_from_json(entries):
    """
    Return (labels=[0,1], y_true, y_pred) for confusion matrix.
    Only entries with perturbed_prediction (exclude errors).
    """
    with_pert = [e for e in entries if "perturbed_prediction" in e and e["perturbed_prediction"] is not None]
    y_true = [e["original_label"] for e in with_pert]
    y_pred = [e["perturbed_prediction"] for e in with_pert]
    return y_true, y_pred


def print_metrics(metrics):
    """Print metrics in the exact same format as AttackMetricsTracker.print_metrics()."""
    print("\n" + "+" + "-" * 31 + "+" + "-" * 8 + "+")
    print("|" + " Attack Results".ljust(31) + "|" + "".ljust(8) + "|")
    print("+" + "-" * 31 + "+" + "-" * 8 + "+")
    print(f"| Number of successful attacks: | {metrics['num_successful']:6d} |")
    print(f"| Number of failed attacks:     | {metrics['num_failed']:6d} |")
    print(f"| Number of skipped attacks:    | {metrics['num_skipped']:6d} |")
    print(f"| Original accuracy:            | {metrics['original_accuracy']:6.2f}% |")
    print(f"| Accuracy under attack:        | {metrics['attacked_accuracy']:6.2f}% |")
    print(f"| Attack success rate:          | {metrics['attack_success_rate']:6.2f}% |")
    print(f"| Average perturbed word %:      | {metrics['avg_perturbed_word_pct']:6.2f}% |")
    print(f"| Average num. words per input: | {metrics['avg_words_per_input']:6.2f} |")
    print(f"| Avg num queries:              | {metrics['avg_num_queries']:6.2f} |")
    print("+" + "-" * 31 + "+" + "-" * 8 + "+")


def print_confusion_matrix(y_true, y_pred):
    """Print confusion matrix in the exact same format as AttackMetricsTracker.print_confusion_matrix()."""
    if not y_true or not y_pred:
        print("No data available for confusion matrix.")
        return
    cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)
    print(f"\n                    Predicted")
    print(f"                  Legitimate  Smishing")
    print(f"  Actual Legitimate    {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"  Actual Smishing     {cm[1][0]:4d}      {cm[1][1]:4d}")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Print Pruthi attack results (same format as end of run) from JSON.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to pruthi_results_*.json. If omitted, use latest in attack_results/.",
    )
    args = parser.parse_args()

    if args.path:
        path = args.path
        if not os.path.isfile(path):
            raise SystemExit(f"File not found: {path}")
        entries = load_pruthi_results(path)
    else:
        results_dir = "attack_results"
        pattern = os.path.join(results_dir, "pruthi_results_*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not files:
            raise SystemExit(f"No Pruthi results found: {pattern}")
        entries = []
        for path in files:
            entries.extend(load_pruthi_results(path))
        print(f"Combined {len(files)} Pruthi results: {', '.join(files)}")
    if not entries:
        print("No entries in results file.")
        return

    metrics = compute_metrics_from_json(entries)
    print_metrics(metrics)

    y_true, y_pred = confusion_matrix_from_json(entries)
    print_confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    main()
