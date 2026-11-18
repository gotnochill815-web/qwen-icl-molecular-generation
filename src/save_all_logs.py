import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs/logs"

EXPERIMENTS = [
    ("logp", "single"),
    ("qed", "single"),
    ("tpsa", "single"),
    ("sas", "single"),
    ("tpsa_sas", "multi"),
    ("logp_tpsa", "multi"),
    ("sas_logp", "multi"),
    ("logp_sas_tpsa", "multi"),
]

def reconstruct_generation_log(name):
    raw_path = ROOT / f"outputs/raw/{name}_zero_shot.jsonl"
    log_path = LOG_DIR / f"{name}_generation.log"

    records = [
        json.loads(line)
        for line in raw_path.read_text().splitlines()
        if line.strip()
    ]

    with log_path.open("w") as f:
        f.write(f"Experiment: {name}\n")
        f.write(f"Total records: {len(records)}\n")
        f.write("=" * 70 + "\n\n")

        for i, r in enumerate(records, 1):
            f.write(
                f"[{i}/{len(records)}] {r.get('job_id')} | "
                f"{r.get('experiment')} | "
                f"targets={r.get('targets')}\n"
            )
            f.write(
                f"Valid: {r.get('valid')} | "
                f"SMILES: {r.get('smiles', '')}\n"
            )
            f.write(
                f"Measured properties: "
                f"{r.get('measured_properties', {})}\n\n"
            )

        f.write("Generation batch finished.\n")
        f.write(f"Records written: {len(records)}\n")

    print(f"Generation log saved: {log_path}")
    return len(records)


def save_evaluation_log(name, evaluator):
    config = ROOT / f"configs/{name}_zero_shot.json"
    raw = ROOT / f"outputs/raw/{name}_zero_shot.jsonl"

    tag = f"{name}_zero_shot_chemfm"

    script = (
        "evaluate_chemfm_property.py"
        if evaluator == "single"
        else "evaluate_chemfm_multi.py"
    )

    command = [
        sys.executable,
        str(ROOT / "src" / script),
        "--config", str(config),
        "--input", str(raw),
        "--tag", tag,
    ]

    log_path = LOG_DIR / f"{name}_evaluation.log"

    print(f"\n========== EVALUATING {name.upper()} ==========")

    with log_path.open("w") as log:
        log.write("Command: " + " ".join(command) + "\n\n")

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        return_code = process.wait()

        if return_code != 0:
            log.write(f"\nEvaluation failed with exit code {return_code}\n")
            raise RuntimeError(f"Evaluation failed: {name}")

    print(f"Evaluation log saved: {log_path}")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Check all inputs before starting.
    for name, _ in EXPERIMENTS:
        config = ROOT / f"configs/{name}_zero_shot.json"
        raw = ROOT / f"outputs/raw/{name}_zero_shot.jsonl"

        if not config.exists():
            raise FileNotFoundError(config)
        if not raw.exists():
            raise FileNotFoundError(raw)

    print("========== SAVING GENERATION LOGS ==========")

    for name, _ in EXPERIMENTS:
        reconstruct_generation_log(name)

    print("\n========== SAVING EVALUATION LOGS ==========")

    for name, evaluator in EXPERIMENTS:
        save_evaluation_log(name, evaluator)

    print("\nALL EXPERIMENT LOGS SAVED SUCCESSFULLY.")
    print("Log directory:", LOG_DIR)


if __name__ == "__main__":
    main()
