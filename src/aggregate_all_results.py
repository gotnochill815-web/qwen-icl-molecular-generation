import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/evaluation/all_experiments"

EXPERIMENTS = {
    "logp": ["logp"],
    "qed": ["qed"],
    "tpsa": ["tpsa"],
    "sas": ["sas"],
    "tpsa_sas": ["tpsa", "sas"],
    "logp_tpsa": ["logp", "tpsa"],
    "sas_logp": ["sas", "logp"],
    "logp_sas_tpsa": ["logp", "sas", "tpsa"],
}

PROPERTIES = ["logp", "qed", "tpsa", "sas"]


def get_metrics(summary):
    # Current ChemFM-compatible evaluator format
    if "overall" in summary:
        return summary["overall"]

    # Backward compatibility with the original evaluator format
    experiment = summary.get("experiment")

    if experiment and experiment in summary:
        return summary[experiment]

    return summary


def normalize_metrics(metrics, properties):
    result = {
        "total_generations": metrics.get("total_generations"),
        "valid_count": metrics.get("valid_count"),
        "validity": metrics.get("validity"),
        "unique_valid_count": metrics.get("unique_valid_count"),
        "uniqueness": metrics.get(
            "uniqueness",
            metrics.get("uniqueness_among_valid")
        ),
        "novel_unique_count": metrics.get("novel_unique_count"),
        "novelty": metrics.get(
            "novelty",
            metrics.get("novelty_among_valid")
        ),
    }

    property_mae = metrics.get("property_mae", {})

    for prop in properties:
        result[f"{prop}_mad"] = metrics.get(
            f"{prop}_diff_mean",
            property_mae.get(prop)
        )
        result[f"{prop}_diff_std"] = metrics.get(
            f"{prop}_diff_std"
        )

    return result


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    target_rows = []
    complete_results = {}
    missing = []

    for experiment, properties in EXPERIMENTS.items():

        summary_path = (
            ROOT / "outputs/evaluation"
            / f"{experiment}_zero_shot_chemfm"
            / "summary.json"
        )

        if not summary_path.exists():
            print(f"MISSING: {experiment}")
            missing.append(experiment)
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        overall = get_metrics(summary)

        row = {
            "experiment": experiment,
            "properties": "+".join(properties),
            **normalize_metrics(overall, properties)
        }

        overall_rows.append(row)

        per_target = summary.get("per_target", {})

        for target_id, metrics in per_target.items():
            target_row = {
                "experiment": experiment,
                "target_id": target_id,
                **{
                    f"target_{prop}": metrics.get(f"target_{prop}")
                    for prop in properties
                },
                **normalize_metrics(metrics, properties)
            }

            target_rows.append(target_row)

        complete_results[experiment] = {
            "properties": properties,
            "overall": overall,
            "per_target": per_target
        }

        print(f"Loaded: {experiment}")

    overall_fields = [
        "experiment",
        "properties",
        "total_generations",
        "valid_count",
        "validity",
        "unique_valid_count",
        "uniqueness",
        "novel_unique_count",
        "novelty",
    ]

    for prop in PROPERTIES:
        overall_fields.extend([
            f"{prop}_mad",
            f"{prop}_diff_std"
        ])

    target_fields = [
        "experiment",
        "target_id",
        "target_logp",
        "target_qed",
        "target_tpsa",
        "target_sas",
        "total_generations",
        "valid_count",
        "validity",
        "unique_valid_count",
        "uniqueness",
        "novel_unique_count",
        "novelty",
    ]

    for prop in PROPERTIES:
        target_fields.extend([
            f"{prop}_mad",
            f"{prop}_diff_std"
        ])

    write_csv(
        OUTPUT / "all_results.csv",
        overall_rows,
        overall_fields
    )

    write_csv(
        OUTPUT / "per_target_results.csv",
        target_rows,
        target_fields
    )

    with open(OUTPUT / "all_results.json", "w") as f:
        json.dump(complete_results, f, indent=2)

    with open(OUTPUT / "missing_results.txt", "w") as f:
        if missing:
            f.write("\n".join(missing))
        else:
            f.write("All 8 experiments have evaluation results.\n")

    print("\n========== OVERALL RESULTS ==========")

    for row in overall_rows:
        print(f"\n{row['experiment'].upper()}")

        for key, value in row.items():
            if key == "experiment":
                continue
            print(f"  {key}: {value}")

    print("\n========== SUMMARY ==========")
    print(f"Experiments completed: {len(overall_rows)}/8")
    print(f"Per-target result rows: {len(target_rows)}")

    if missing:
        print("Missing evaluations:", ", ".join(missing))
    else:
        print("All experiments evaluated successfully!")

    print("\nSaved results:")
    print(OUTPUT / "all_results.csv")
    print(OUTPUT / "per_target_results.csv")
    print(OUTPUT / "all_results.json")
    print(OUTPUT / "missing_results.txt")


if __name__ == "__main__":
    main()
