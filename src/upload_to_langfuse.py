import argparse
import hashlib
import json
from pathlib import Path

from langfuse import get_client

PROJECT = "qwen_icl_molecular_generation"

EXPERIMENTS = [
    "logp",
    "qed",
    "tpsa",
    "sas",
    "tpsa_sas",
    "logp_tpsa",
    "sas_logp",
    "logp_sas_tpsa",
]


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    records = []
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path}, line {line_number}: {exc}"
                    )
    return records


def file_hash(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def get_experiment_files(name):
    return {
        "config": Path(f"configs/{name}_zero_shot.json"),
        "raw": Path(f"outputs/raw/{name}_zero_shot.jsonl"),
        "summary": Path(
            f"outputs/evaluation/{name}_zero_shot_chemfm/summary.json"
        ),
        "generation_log": Path(f"outputs/logs/{name}_generation.log"),
        "evaluation_log": Path(f"outputs/logs/{name}_evaluation.log"),
    }


def validate_experiment(name, files):
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{name}: missing required files:\n" + "\n".join(missing)
        )

    config = read_json(files["config"])
    summary = read_json(files["summary"])
    records = read_jsonl(files["raw"])

    if len(records) != 100:
        raise ValueError(
            f"{name}: expected 100 generation records, found {len(records)}"
        )

    for record in records:
        if record.get("experiment") != name:
            raise ValueError(
                f"{name}: unexpected experiment in record "
                f"{record.get('job_id')}: {record.get('experiment')}"
            )

    return config, summary, records


def upload_experiment(langfuse, name, files, config, summary, records):
    generation_config = config.get("generation", {})

    root_metadata = {
        "project": PROJECT,
        "experiment": name,
        "model": config.get("model", {}).get("model_id"),
        "shots": generation_config.get("shots", 0),
        "evaluation_method": summary.get("evaluation_method"),
        "source": "historical_hpc_experiment_import",
    }

    with langfuse.start_as_current_observation(
        as_type="span",
        name=f"qwen-icl-{name}-zero-shot",
        input={
            "experiment": name,
            "config": config,
            "generation_record_count": len(records),
        },
        metadata=root_metadata,
    ) as root:

        # One Langfuse generation observation for each molecule.
        for record in records:
            model_parameters = {
                key: generation_config[key]
                for key in (
                    "temperature",
                    "top_p",
                    "max_new_tokens",
                    "do_sample",
                    "repetition_penalty",
                )
                if key in generation_config
            }
            model_parameters["shots"] = record.get("shots", 0)
            model_parameters["seed"] = record.get("seed")

            with langfuse.start_as_current_observation(
                as_type="generation",
                name="molecule-generation",
                model=record.get("backend"),
                model_parameters=model_parameters,
                input={
                    "prompt": record.get("prompt"),
                    "properties": record.get("properties"),
                    "targets": record.get("targets"),
                },
                output={
                    "raw_output": record.get("raw_output"),
                    "raw_smiles": record.get("raw_smiles"),
                    "smiles": record.get("smiles"),
                    "valid": record.get("valid"),
                    "measured_properties": record.get(
                        "measured_properties"
                    ),
                },
                metadata={
                    "job_id": record.get("job_id"),
                    "target_id": record.get("target_id"),
                    "target_index": record.get("target_index"),
                    "replicate_index": record.get("replicate_index"),
                    "sample_index": record.get("sample_index"),
                    "seed": record.get("seed"),
                    "shots": record.get("shots"),
                    "timestamp": record.get("timestamp"),
                    "experiment": name,
                },
            ):
                pass

        # Store the complete evaluation summary, including per-target metrics.
        with langfuse.start_as_current_observation(
            as_type="span",
            name="chemfm-evaluation-summary",
            input={
                "evaluation_method": summary.get("evaluation_method"),
                "novelty_definition": summary.get("novelty_definition"),
            },
            output=summary,
            metadata={
                "experiment": name,
                "record_count": summary.get(
                    "overall", {}
                ).get("total_generations"),
            },
        ):
            pass

        # Store both saved console log files as text observations.
        for log_type in ("generation_log", "evaluation_log"):
            log_path = files[log_type]
            log_text = log_path.read_text(errors="replace")

            with langfuse.start_as_current_observation(
                as_type="span",
                name=log_type.replace("_", "-"),
                input={"file_name": log_path.name},
                output=log_text,
                metadata={
                    "experiment": name,
                    "file_size_bytes": log_path.stat().st_size,
                },
            ):
                pass

        root.update(
            output={
                "generation_records_uploaded": len(records),
                "evaluation_overall": summary.get("overall", {}),
                "logs_uploaded": [
                    files["generation_log"].name,
                    files["evaluation_log"].name,
                ],
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all files without uploading anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload again even if an experiment is in the local checkpoint.",
    )
    args = parser.parse_args()

    state_path = Path("outputs/langfuse_import_state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        state = read_json(state_path)
    else:
        state = {"completed": {}}

    prepared = {}
    total_records = 0
    total_logs = 0

    # Validate every experiment before beginning any upload.
    for name in EXPERIMENTS:
        files = get_experiment_files(name)
        config, summary, records = validate_experiment(name, files)
        signature = file_hash(list(files.values()))

        prepared[name] = (files, config, summary, records, signature)
        total_records += len(records)
        total_logs += 2

        print(
            f"[READY] {name}: {len(records)} generations, "
            f"evaluation summary, 2 logs"
        )

    print("\nValidation complete.")
    print(f"Experiments: {len(prepared)}")
    print(f"Generation records: {total_records}")
    print(f"Evaluation summaries: {len(prepared)}")
    print(f"Log files: {total_logs}")

    if args.dry_run:
        print("\nDRY RUN: nothing was uploaded.")
        return

    required_env = (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
    )
    import os

    missing_env = [key for key in required_env if not os.environ.get(key)]
    if missing_env:
        raise EnvironmentError(
            "Missing Langfuse environment variables: "
            + ", ".join(missing_env)
        )

    langfuse = get_client()
    if not langfuse.auth_check():
        raise RuntimeError("Langfuse authentication failed.")

    print("\nLangfuse authentication verified.")

    for name, data in prepared.items():
        files, config, summary, records, signature = data

        if (
            not args.force
            and state["completed"].get(name) == signature
        ):
            print(f"[SKIP] {name}: already uploaded with these files.")
            continue

        print(f"\n[UPLOADING] {name} ...")

        upload_experiment(
            langfuse,
            name,
            files,
            config,
            summary,
            records,
        )

        # Flush each experiment before marking it complete locally.
        langfuse.flush()

        state["completed"][name] = signature
        temp_path = state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(state, indent=2))
        temp_path.replace(state_path)

        print(
            f"[DONE] {name}: 100 generations, "
            "evaluation summary, 2 logs uploaded."
        )

    langfuse.flush()
    print("\nAll requested experiment uploads are complete.")


if __name__ == "__main__":
    main()
