from collections import defaultdict
import json
import os
from util.file_utils import get_file_paths

input_dir = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/safety_token_granular/results/7B_laion2B-en_test_10000_336_25_02_25_01"
output_path = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/finngu/LlavaGuard/src/experiments/compare_annotations/results/results_7B_granular_laion2B-en_test_10000_336_25_02_25_01.json"


def summarize_annotations(annotation_paths: list[str], annotation_ids: list[str], output_path: str):
    """
    Summarizes a list of annotation files into a single JSON output file.

    Parameters:
        annotation_paths (list[str]): List of file paths to input .json files.
        annotation_ids (list[str]): List of corresponding annotation IDs.
        output_path (str): Path to the output summary .json file.
    """
    # Initialize summary structure
    summary = {
        "safe_ids": [],
        "unsafe_ids": [],
        "categories": defaultdict(lambda: {"safe_ids": [], "unsafe_ids": []})
    }
    
    for path, annotation_id in zip(annotation_paths, annotation_ids):
        # Read each annotation file
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract "rating" and "category"
        rating = data.get("rating")
        category = data.get("category")
        
        # Populate "safe_ids" and "unsafe_ids"
        if rating == "Safe":
            summary["safe_ids"].append(annotation_id)
            if category:
                summary["categories"][category]['safe_ids'].append(annotation_id)
        elif rating == "Unsafe":
            summary["unsafe_ids"].append(annotation_id)
            if category:
                summary["categories"][category]['unsafe_ids'].append(annotation_id)
    
    # Convert defaultdict to a regular dictionary for serialization
    summary["categories"] = dict(summary["categories"])
    
    # Write the summary to the output JSON file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(summary, output_file, indent=4)

    print(f"Summary saved to {output_path}")

def main():
    if not os.path.exists(input_dir):
        raise Exception("The input directory doesn't exist!")

    if os.path.exists(output_path):
        raise Exception("The output file already exists! Change or delete it to avoid losing data.")

    # Get annotations of all dataset splits
    annotation_paths = []
    annotation_ids = []

    # for split in ['test', 'train', 'validation']:
    #     paths, ids = get_file_paths(input_dir, file_extension='.json')

    #     ids = map(lambda x: f"{split}/{x.removesuffix('.jpg')}", ids)

    #     annotation_paths.extend(paths)
    #     annotation_ids.extend(ids)

    annotation_paths, annotation_ids = get_file_paths(input_dir, file_extension='.json')

    summarize_annotations(annotation_paths, annotation_ids, output_path)

if __name__ == '__main__':
    main()
