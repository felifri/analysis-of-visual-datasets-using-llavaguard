"""Launch PRX scaling experiments for dose-response study.

Two experiments on C1 (0% unsafe) and C0 (original 1.21% unsafe):
  1. PRX-7.3B with FLUX VAE (latent-space diffusion)
  2. PRX-Medium (~3.8B) pixel-space (isolates parameter count)

Single-phase training: 512px for 100K steps.

Usage:
    python entrypoint_train_scaling.py [--experiments 7b_vae medium] [--conditions C1 C0] [--dry-run]
    python entrypoint_train_scaling.py --experiments 7b_vae --conditions C1 --dry-run
"""

import argparse
import json
import logging
import os
import subprocess
import textwrap


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONDITIONS = ["C1", "C0"]

EXPERIMENTS = {
    "7b_vae": {
        "config_name": "dose-response-speedrun-7b-vae",
        "group": "dose-response-7b-vae",
        "checkpoint_subdir": "checkpoints_7b_vae",
        "global_batch_size": 256,
        "device_train_microbatch_size": 4,
        "time": "120:00:00",
        "offline_exports": "export HF_HUB_OFFLINE=0\n    export HF_HUB_DISABLE_TELEMETRY=1",  # needs HF download for FLUX VAE
    },
    "medium": {
        "config_name": "dose-response-speedrun-medium",
        "group": "dose-response-medium",
        "checkpoint_subdir": "checkpoints_medium",
        "global_batch_size": 256,
        "device_train_microbatch_size": 8,
        "time": "60:00:00",
        "offline_exports": "export HF_HUB_OFFLINE=1\n    export TRANSFORMERS_OFFLINE=1",
    },
}

SLURM_TEMPLATE = textwrap.dedent("""\
    #!/bin/bash
    #SBATCH --job-name={experiment}-{condition}
    #SBATCH --output={log_dir}/train_{experiment}_{condition}_%j.out
    #SBATCH --error={log_dir}/train_{experiment}_{condition}_%j.err
    #SBATCH --qos={qos}
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task={cpus_per_node}
    #SBATCH --gres=gpu:{gpus_per_node}
    #SBATCH --time={time}

    set -euo pipefail

    echo "{experiment} {condition} training started on $(hostname) at $(date)"

    export FSDP_VERSION=2
    export TORCH_HOME=<your folder>
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    {offline_exports}
    export SGLANG_DISABLE_CUDNN_CHECK=1
    export HYDRA_FULL_ERROR=1

    COMPOSER=<your folder>
    CKPT_DIR="{checkpoint_dir}"

    cd {prx_dir}

    # 512px, 100K steps
    $COMPOSER -n {gpus_per_node} -m prx.training.train \\
        --config-name {config_name} \\
        "dataset@dataset.train_dataset=train_dose_{condition_lower}" \\
        "dataset@dataset.eval_dataset=train_dose_{condition_lower}" \\
        name={condition} group={group} image_size=512 \\
        global_batch_size={global_batch_size} device_train_microbatch_size={device_train_microbatch_size} \\
        trainer.max_duration=100_000ba \\
        trainer.save_folder=${{CKPT_DIR}}/{condition} \\
        trainer.run_name={group}-{condition} \\
        eval_first=false trainer.eval_interval=0 seed=42 {resume_args}

    echo "{experiment} {condition} training complete at $(date)"
""")


def main():
    parser = argparse.ArgumentParser(
        description="Launch PRX scaling experiments for dose-response study"
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=list(EXPERIMENTS.keys()),
        choices=list(EXPERIMENTS.keys()),
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=CONDITIONS,
        help="Which conditions to train (default: C1 C0)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but don't submit")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint (adds +trainer.load_path)")
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "config.json")) as f:
        config = json.load(f)

    prx_dir = config["training"]["prx_dir"]
    base_checkpoint_dir = os.path.dirname(config["training"]["checkpoint_dir"])
    slurm_config = config["slurm"]

    gpus_per_node = slurm_config["gpus_per_node"]
    cpus_per_node = slurm_config["cpus_per_node"]
    qos = slurm_config["qos"]

    log_dir = os.path.join(config["base_output_dir"], "slurm_logs_scaling")
    scripts_dir = os.path.join(config["base_output_dir"], "slurm_scripts_scaling")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    submitted_jobs = {}

    for experiment_name in args.experiments:
        exp = EXPERIMENTS[experiment_name]
        checkpoint_dir = os.path.join(base_checkpoint_dir, exp["checkpoint_subdir"])

        for condition in args.conditions:
            if condition not in CONDITIONS:
                logger.warning("Unknown condition %s, skipping", condition)
                continue

            fmt_args = {
                "experiment": experiment_name,
                "condition": condition,
                "condition_lower": condition.lower(),
                "config_name": exp["config_name"],
                "group": exp["group"],
                "prx_dir": prx_dir,
                "checkpoint_dir": checkpoint_dir,
                "log_dir": log_dir,
                "qos": qos,
                "gpus_per_node": gpus_per_node,
                "cpus_per_node": cpus_per_node,
                "global_batch_size": exp["global_batch_size"],
                "device_train_microbatch_size": exp["device_train_microbatch_size"],
                "time": exp["time"],
                "offline_exports": exp["offline_exports"],
            }

            # Add resume args if checkpoint exists
            resume_args = ""
            if args.resume:
                ckpt_cond_dir = os.path.join(checkpoint_dir, condition)
                latest = os.path.join(ckpt_cond_dir, "latest-rank0.pt")
                if os.path.exists(latest):
                    resume_args = (
                        f"+trainer.load_path={ckpt_cond_dir}/latest-rank0.pt "
                        f"+trainer.load_weights_only=true "
                        f"+trainer.load_ignore_keys=['state/model/vae*','state/model/text_tower*']"
                    )
                    logger.info("Will resume %s %s from %s", experiment_name, condition, latest)
                else:
                    logger.warning("No checkpoint found for %s %s to resume from", experiment_name, condition)
            fmt_args["resume_args"] = resume_args

            script_content = SLURM_TEMPLATE.format(**fmt_args)
            script_path = os.path.join(scripts_dir, f"train_{experiment_name}_{condition}.sh")

            with open(script_path, "w") as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)

            logger.info("Generated: %s", script_path)

            if not args.dry_run:
                submit_cmd = ["sbatch", script_path]
                result = subprocess.run(submit_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    submitted_jobs[f"{experiment_name}_{condition}"] = job_id
                    logger.info("Submitted %s %s: job %s",
                                experiment_name, condition, job_id)
                else:
                    logger.error("Failed to submit %s %s: %s",
                                 experiment_name, condition, result.stderr)

    if args.dry_run:
        total = len(args.experiments) * len(args.conditions)
        logger.info("Dry run complete. %d scripts generated in %s", total, scripts_dir)
    else:
        logger.info("Submitted %d jobs", len(submitted_jobs))
        summary_path = os.path.join(log_dir, "submitted_jobs_scaling.json")
        with open(summary_path, "w") as f:
            json.dump(submitted_jobs, f, indent=2)
        logger.info("Job IDs saved to %s", summary_path)


if __name__ == "__main__":
    main()
