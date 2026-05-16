"""Normalize cross-judge annotations into unified parquet files.

For each source (dose_C1..C5, prx-*, training_data), creates a parquet with columns:
  - id (index)
  - llavaguard_unsafe (0/1)
  - llavaguard_category (str, e.g. "O3: Sexual Content" or "NA: None applying")
  - llamaguard3_unsafe (0/1)
  - llamaguard3_category (str, e.g. "S8" or "NA")
  - shieldgemma_unsafe (0/1)
  - shieldgemma_category (str, e.g. "dangerous" / "sexual" / "violence" or "NA")
  - shieldgemma_p_dangerous (float)
  - shieldgemma_p_sexual (float)
  - shieldgemma_p_violence (float)
  - sd_safety_checker_unsafe (0/1)

Usage:
    python normalize_annotations.py [--cross_judge_dir PATH] [--llavaguard_dir PATH] [--output_dir PATH]
"""

import argparse
import json
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CROSS_JUDGE_DIR = "<your folder>"
DEFAULT_LLAVAGUARD_DIR = "<your folder>"
DEFAULT_OUTPUT_DIR = "<your folder>"

# LlamaGuard3 MLCommons categories
LLAMAGUARD3_CATEGORIES = {
    "S1": "S1: Violent Crimes",
    "S2": "S2: Non-Violent Crimes",
    "S3": "S3: Sex-Related Crimes",
    "S4": "S4: Child Sexual Exploitation",
    "S5": "S5: Defamation",
    "S6": "S6: Specialized Advice",
    "S7": "S7: Privacy",
    "S8": "S8: Intellectual Property",
    "S9": "S9: Indiscriminate Weapons",
    "S10": "S10: Hate",
    "S11": "S11: Suicide & Self-Harm",
    "S12": "S12: Sexual Content",
    "S13": "S13: Elections",
    "S14": "S14: Code Interpreter Abuse",
}


def load_cross_judge_json(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def normalize_llamaguard3(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        cats = r.get("categories", [])
        # Take first category, expand to full name
        if cats:
            cat = LLAMAGUARD3_CATEGORIES.get(cats[0], cats[0])
        else:
            cat = "NA"
        rows.append({
            "id": r["id"],
            "llamaguard3_unsafe": 1 if r["rating"] == "Unsafe" else 0,
            "llamaguard3_category": cat,
        })
    return pd.DataFrame(rows).set_index("id")


def normalize_shieldgemma(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        # Determine primary category from highest probability
        probs = {
            "dangerous": r.get("p_dangerous", 0.0),
            "sexual": r.get("p_sexual", 0.0),
            "violence": r.get("p_violence", 0.0),
        }
        if r["rating"] == "Unsafe":
            cat = max(probs, key=probs.get)
        else:
            cat = "NA"
        rows.append({
            "id": r["id"],
            "shieldgemma_unsafe": 1 if r["rating"] == "Unsafe" else 0,
            "shieldgemma_category": cat,
            "shieldgemma_p_dangerous": r.get("p_dangerous", 0.0),
            "shieldgemma_p_sexual": r.get("p_sexual", 0.0),
            "shieldgemma_p_violence": r.get("p_violence", 0.0),
        })
    return pd.DataFrame(rows).set_index("id")


def normalize_sd_safety_checker(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "id": r["id"],
            "sd_safety_checker_unsafe": 1 if r["rating"] == "Unsafe" else 0,
        })
    return pd.DataFrame(rows).set_index("id")


def normalize_llavaguard_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return pd.DataFrame({
        "llavaguard_unsafe": (df["rating"] == "Unsafe").astype(int),
        "llavaguard_category": df["category"],
    })


def normalize_llavaguard_from_cross_judge(results: list[dict]) -> pd.DataFrame:
    """For training_data.json which embeds llavaguard annotations."""
    rows = []
    for r in results:
        if "llavaguard_rating" in r:
            rows.append({
                "id": r["id"],
                "llavaguard_unsafe": 1 if r["llavaguard_rating"] == "Unsafe" else 0,
                "llavaguard_category": r.get("llavaguard_category", "NA"),
            })
    return pd.DataFrame(rows).set_index("id") if rows else None


def process_generated_source(source_name: str, cross_judge_dir: str,
                             llavaguard_dir: str, output_dir: str):
    """Process a generated image source (dose_C1, prx-512-base, etc.)."""
    logger.info(f"Processing {source_name}")

    dfs = []

    # LlavaGuard (from parquet)
    llavaguard_path = os.path.join(llavaguard_dir, f"{source_name}.parquet")
    if os.path.exists(llavaguard_path):
        dfs.append(normalize_llavaguard_parquet(llavaguard_path))
        logger.info(f"  LlavaGuard: {len(dfs[-1])} annotations")

    # Cross-judge models
    for prefix, normalize_fn in [
        ("llamaguard3", normalize_llamaguard3),
        ("shieldgemma", normalize_shieldgemma),
        ("sd_safety_checker", normalize_sd_safety_checker),
    ]:
        path = os.path.join(cross_judge_dir, f"{prefix}_{source_name}.json")
        if os.path.exists(path):
            results = load_cross_judge_json(path)
            df = normalize_fn(results)
            dfs.append(df)
            logger.info(f"  {prefix}: {len(df)} annotations")

    if not dfs:
        logger.warning(f"  No annotations found for {source_name}, skipping")
        return

    # Join all on index
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")

    output_path = os.path.join(output_dir, f"{source_name}.parquet")
    merged.to_parquet(output_path)
    logger.info(f"  Saved {len(merged)} rows to {output_path}")


def process_training_data(cross_judge_dir: str, output_dir: str):
    """Process training data cross-judge annotations."""
    logger.info("Processing training_data")

    dfs = []

    for prefix, normalize_fn in [
        ("llamaguard3", normalize_llamaguard3),
        ("shieldgemma", normalize_shieldgemma),
        ("sd_safety_checker", normalize_sd_safety_checker),
    ]:
        path = os.path.join(cross_judge_dir, f"{prefix}_training_data.json")
        if os.path.exists(path):
            results = load_cross_judge_json(path)
            df = normalize_fn(results)
            dfs.append(df)
            logger.info(f"  {prefix}: {len(df)} annotations")

            # Extract embedded LlavaGuard annotations (only need once)
            if prefix == "llamaguard3":
                llavaguard_df = normalize_llavaguard_from_cross_judge(results)
                if llavaguard_df is not None:
                    dfs.insert(0, llavaguard_df)
                    logger.info(f"  llavaguard (embedded): {len(llavaguard_df)} annotations")

    if not dfs:
        logger.warning("  No training data annotations found")
        return

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how="outer")

    output_path = os.path.join(output_dir, "cross_judge_training_sample.parquet")
    merged.to_parquet(output_path)
    logger.info(f"  Saved {len(merged)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize cross-judge annotations")
    parser.add_argument("--cross_judge_dir", default=DEFAULT_CROSS_JUDGE_DIR)
    parser.add_argument("--llavaguard_dir", default=DEFAULT_LLAVAGUARD_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Dose-response conditions
    for c in ["C1", "C2", "C0", "C4", "C6", "C5"]:
        process_generated_source(f"dose_{c}", args.cross_judge_dir,
                                 args.llavaguard_dir, args.output_dir)

    # PRX existing models
    for m in ["prx-1024-beta", "prx-256-base", "prx-256-sft",
              "prx-512-base", "prx-512-dc-ae", "prx-512-sft",
              "prx-512-sft-distilled"]:
        process_generated_source(m, args.cross_judge_dir,
                                 args.llavaguard_dir, args.output_dir)

    # Training data sample
    process_training_data(args.cross_judge_dir, args.output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
