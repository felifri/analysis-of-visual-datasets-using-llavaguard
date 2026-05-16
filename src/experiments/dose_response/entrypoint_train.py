"""Step 4: Train PRX-1.2B on each experimental condition via SLURM.

Generates and submits SLURM job scripts for each condition (C1-C6).
Each job runs the PRX speedrun training recipe with the appropriate
dataset YAML config pointing to the condition's MDS data.

Two-phase training:
  Phase 1: 512px for 100K steps (batch 1024)
  Phase 2: 1024px for 20K steps (batch 512, no REPA)

Usage:
    python entrypoint_train.py [--conditions C1 C2 ...] [--dry-run]
"""

import argparse
import json
import logging
import os
import subprocess
import textwrap


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["C1", "C2", "C3", "C0", "C4", "C6"]

SLURM_TEMPLATE_PHASE1 = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name=dose-{condition}-p1
    #SBATCH --output={log_dir}/train_{condition}_phase1_%j.out
    #SBATCH --error={log_dir}/train_{condition}_phase1_%j.err
    #SBATCH --qos={qos}
    #SBATCH --nodes={nodes}
    #SBATCH --ntasks-per-node={gpus_per_node}
    #SBATCH --cpus-per-task={cpus_per_task}
    #SBATCH --gres=gpu:{gpus_per_node}
    #SBATCH --time=48:00:00

    set -euo pipefail

    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
    export MASTER_PORT=29500
    export WORLD_SIZE=$SLURM_NTASKS
    export FSDP_VERSION=2

    cd {prx_dir}

    # Phase 1: 512px, 100K steps, batch 1024
    srun composer train.py \\
        --config-path configs/yamls/dose-response \\
        --config-name speedrun \\
        "dataset@dataset.train_dataset=train_dose_{condition_lower}" \\
        name={condition} \\
        group=dose-response \\
        image_size=512 \\
        global_batch_size=1024 \\
        device_train_microbatch_size=16 \\
        trainer.max_duration=100_000ba \\
        trainer.save_folder={checkpoint_dir}/{condition}/phase1 \\
        trainer.run_name=dose-response-{condition}-phase1 \\
        seed=42
""")

SLURM_TEMPLATE_PHASE2 = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name=dose-{condition}-p2
    #SBATCH --output={log_dir}/train_{condition}_phase2_%j.out
    #SBATCH --error={log_dir}/train_{condition}_phase2_%j.err
    #SBATCH --qos={qos}
    #SBATCH --nodes={nodes}
    #SBATCH --ntasks-per-node={gpus_per_node}
    #SBATCH --cpus-per-task={cpus_per_task}
    #SBATCH --gres=gpu:{gpus_per_node}
    #SBATCH --time=24:00:00

    set -euo pipefail

    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
    export MASTER_PORT=29500
    export WORLD_SIZE=$SLURM_NTASKS
    export FSDP_VERSION=2

    cd {prx_dir}

    # Phase 2: 1024px, 20K steps, batch 512, no REPA
    srun composer train.py \\
        --config-path configs/yamls/dose-response \\
        --config-name speedrun \\
        "dataset@dataset.train_dataset=train_dose_{condition_lower}" \\
        name={condition} \\
        group=dose-response \\
        image_size=1024 \\
        global_batch_size=512 \\
        device_train_microbatch_size=8 \\
        trainer.max_duration=20_000ba \\
        trainer.save_folder={checkpoint_dir}/{condition}/phase2 \\
        trainer.run_name=dose-response-{condition}-phase2 \\
        trainer.load_path={checkpoint_dir}/{condition}/phase1/latest-rank0.pt \\
        "~algorithms.repa" \\
        seed=42
""")


def main():
    parser = argparse.ArgumentParser(description="Launch PRX training for dose-response conditions")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=CONDITIONS,
        help="Which conditions to train (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but don't submit")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Only run phase 1 or 2 (default: both)")
    parser.add_argument("--nodes", type=int, default=None, help="Override number of nodes")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    prx_dir = config["training"]["prx_dir"]
    checkpoint_dir = config["training"]["checkpoint_dir"]
    slurm_config = config["slurm"]

    nodes = args.nodes or slurm_config["nodes"]
    gpus_per_node = slurm_config["gpus_per_node"]
    cpus_per_task = slurm_config["cpus_per_node"] // gpus_per_node
    qos = slurm_config["qos"]

    log_dir = os.path.join(config["base_output_dir"], "slurm_logs")
    scripts_dir = os.path.join(config["base_output_dir"], "slurm_scripts")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    submitted_jobs = {}

    for condition in args.conditions:
        if condition not in CONDITIONS:
            logger.warning(f"Unknown condition {condition}, skipping")
            continue

        fmt_args = {
            "condition": condition,
            "condition_lower": condition.lower(),
            "prx_dir": prx_dir,
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "qos": qos,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "cpus_per_task": cpus_per_task,
        }

        phases = []
        if args.phase is None or args.phase == 1:
            phases.append(("phase1", SLURM_TEMPLATE_PHASE1))
        if args.phase is None or args.phase == 2:
            phases.append(("phase2", SLURM_TEMPLATE_PHASE2))

        for phase_name, template in phases:
            script_content = template.format(**fmt_args)
            script_path = os.path.join(scripts_dir, f"train_{condition}_{phase_name}.sh")

            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)

            logger.info(f"Generated: {script_path}")

            if not args.dry_run:
                # Submit with dependency if phase 2 depends on phase 1
                submit_cmd = ["sbatch"]
                if phase_name == "phase2" and f"{condition}_phase1" in submitted_jobs:
                    dep_job_id = submitted_jobs[f"{condition}_phase1"]
                    submit_cmd.extend(["--dependency", f"afterok:{dep_job_id}"])

                submit_cmd.append(script_path)
                result = subprocess.run(submit_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    submitted_jobs[f"{condition}_{phase_name}"] = job_id
                    logger.info(f"Submitted {condition} {phase_name}: job {job_id}")
                else:
                    logger.error(f"Failed to submit {condition} {phase_name}: {result.stderr}")

    if args.dry_run:
        logger.info(f"Dry run complete. {len(args.conditions) * len(phases)} scripts generated in {scripts_dir}")
    else:
        logger.info(f"Submitted {len(submitted_jobs)} jobs")
        summary_path = os.path.join(log_dir, "submitted_jobs.json")
        with open(summary_path, "w") as f:
            json.dump(submitted_jobs, f, indent=2)
        logger.info(f"Job IDs saved to {summary_path}")


if __name__ == "__main__":
    main()
