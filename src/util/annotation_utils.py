import json
import os

import pandas as pd

from collections import defaultdict
from util.categories import ALL_CATEGORIES
from util.file_utils import get_file_paths

def summarize_annotations(input_dir: str, output_path: str):
    """
    Summarizes a directory of annotation files into a single JSON output file.

    Parameters:
        input_dir (str): Path to the directory containing the input .json files.
        output_path (str): Path to the output summary .json file.
    """

    if not os.path.exists(input_dir):
        raise Exception("The input directory doesn't exist!")

    if os.path.exists(output_path):
        raise Exception("The output file already exists! Change or delete it to avoid losing data.")

    annotation_paths, annotation_ids = get_file_paths(input_dir, file_extension='.json')
    
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


def inspect_summarized_annotations(summary_path: str):
    print(f"Inspecting summarized annotations: {summary_path.split('/')[-1]}")

    with open(summary_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        print(f"Total IDs: {len(set([*data['safe_ids'], *data['unsafe_ids']]))}")
        print("\t" + f"Safe: {len(data['safe_ids'])}")
        print("\t" + f"Unsafe: {len(data['unsafe_ids'])}")
        print(f"Categories: {len(data['categories'].keys())}")
        print("\t" + f"{'Category':<35} {'# IDs':<7} {'Safe/Unsafe':<11} {'Safe Samples':<36} {'Unsafe Samples':<36}")

        for cat in sorted(data['categories'].keys()):
            safe_ids = data['categories'][cat]['safe_ids']
            unsafe_ids = data['categories'][cat]['unsafe_ids']

            num_safe = len(safe_ids)
            num_unsafe = len(unsafe_ids)
            num_total = num_safe + num_unsafe

            print("\t" + f"{cat:<35} {num_total:>7} {str(num_safe) + '/' + str(num_unsafe):>11} {str(sorted(safe_ids)[:3]):<36} {str(sorted(unsafe_ids)[:3]):<36}")


def visualize_summarized_annotations(summary_path: str, title: str = "Count of Safe and Unsafe IDs per Category"):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    with open(summary_path, 'r', encoding='utf-8') as file:
        summary = json.load(file)
        data = []

        for category, ids_dict in summary["categories"].items():
            safe_count = len(ids_dict["safe_ids"])
            unsafe_count = len(ids_dict["unsafe_ids"])
            data.append({"Category": category, "Count": safe_count, "Rating": "Safe"})
            data.append({"Category": category, "Count": unsafe_count, "Rating": "Unsafe"})
        
        # Convert to a DataFrame
        df = pd.DataFrame(data)
        
        # Plot using seaborn
        ax = sns.barplot(
            data=df, 
            x="Category", 
            y="Count", 
            hue="Rating",
        )

        ax.set_yscale("log")
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Categories")


def visualize_summarized_annotations_stacked_barplot(summary_path: str, title: str = "Count of Safe and Unsafe IDs per Category"):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    with open(summary_path, 'r', encoding='utf-8') as file:
        summary = json.load(file)
        data = []

        for category, ids_dict in summary["categories"].items():
            safe_count = len(ids_dict["safe_ids"])
            unsafe_count = len(ids_dict["unsafe_ids"])
            data.append({"Category": category, "Count": safe_count, "Rating": "Safe"})
            data.append({"Category": category, "Count": unsafe_count, "Rating": "Unsafe"})
        
        # Convert to a DataFrame
        df = pd.DataFrame(data)
        
        # Plot using seaborn
        ax = sns.barplot(
            data=df, 
            x="Category", 
            y="Count",
            stacked=True,
            hue="Rating",
        )

        ax.set_yscale("log")
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Categories")

def visualize_annotation_dataframe(df: pd.DataFrame, title: str = "Count of Safe and Unsafe IDs per Category"):
    """
    Draws a stacked bar plot of the annotations using seaborn.
    Each category has two bars: one showing the total number of samples,
    and another in front showing how many of these are rated as 'Unsafe'.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Count total samples per category
    df_total = df.groupby('category', observed=False).size().reset_index(name='total_count')

    # Insert any missing categories with zero counts. Possible categories are NA, O1, O2, ..., O9
    for cat in ALL_CATEGORIES:
        if cat not in df_total['category'].values:
            df_total = pd.concat([df_total, pd.DataFrame({'category': [cat], 'total_count': [0]})], ignore_index=True)

    df_total = df_total.sort_values(by=['category'])

    # Count 'Unsafe' samples per category
    df_unsafe = df[df['rating'] == 'Unsafe'].groupby('category', observed=False).size().reset_index(name='unsafe_count')

    # Ensure 'unsafe_count' column exists in df_unsafe and convert to int
    df_unsafe['unsafe_count'] = df_unsafe['unsafe_count'].astype(int)
    
    # Merge the two dataframes
    df_counts = pd.merge(df_total, df_unsafe, on='category', how='left').fillna({'unsafe_count': 0})
    df_counts['unsafe_count'] = df_counts['unsafe_count'].astype(int)
    
    # Ensure category remains categorical after merging
    df_counts['category'] = pd.Categorical(df_counts['category'], categories=sorted(ALL_CATEGORIES), ordered=True)
    
    # Melt dataframe for seaborn barplot
    df_melted = df_counts.melt(id_vars=['category'], value_vars=['total_count', 'unsafe_count'], var_name='Type', value_name='count')

    # Only show first two letters of category names
    df_melted['category'] = df_melted['category'].str[:2]

    # Only show categories NA, O1, O2, ..., O9 in plot and inform user which categories were omitted
    if not df_melted['category'].isin(['NA', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9']).all():
        excluded_categories = df_melted[~df_melted['category'].isin(['NA', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9'])]['category'].unique()
        print(f"Excluded categories: {excluded_categories}")

        df_melted = df_melted[df_melted['category'].isin(['NA', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9'])]
    
    # Create the stacked bar plot
    plt.figure(figsize=(10, 6))
    sns.set(style='whitegrid')
    ax = sns.barplot(
        data=df_melted,
        x='category',
        y='count',
        hue='Type',
        dodge=False,  # Stack bars per category
        errorbar=None,
    )
    ax.set_yscale('log')
    
    # Apply hatching only to 'unsafe_count' bars
    hatch_patterns = {"unsafe_count": "//"}  # Diagonal lines for unsafe bars
    for bar, (_, row) in zip(ax.patches, df_melted.iterrows()):
        if row['Type'] == 'unsafe_count':
            bar.set_hatch(hatch_patterns["unsafe_count"])

    # Function to format large numbers
    def format_label(value):
        if value >= 1000:
            return f"{value/1000:.1f}k"
        return str(int(value))
    
    # Add simplified absolute values above bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                text=format_label(height), 
                xy=(p.get_x() + p.get_width() / 2, height),
                ha='center', 
                va='bottom'
            )
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('# of Samples')
    
    # Customize legend to remove hatching
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        handle.set_hatch(hatch_patterns.get(label, ""))
    plt.legend(handles=handles, title='', labels=['# Category Detections', '# Unsafe Detections'])
    
    # Show the plot
    plt.show()


def show_image_details(img_dir: str, img_id: str, include_category: bool = False, include_safety: bool = False, include_rationale: bool = False, annotation_dirs: list[str] = [], img_extension: str = ".jpg"):
    from IPython.display import Image, display
    found_path = ""

    for path, name in zip(*list(get_file_paths(img_dir, img_extension))):
        if name == img_id:
            found_path = path
            break
            
    if found_path:
        print(found_path)
        display(Image(filename=found_path))
    else: 
        print("Image not found")

    if include_category or include_safety or include_rationale:
        assert len(annotation_dirs) != 0

        for annotation_dir in annotation_dirs:
            print("Searching dir for annotations: " + annotation_dir)
            annotation_path = ""

            for root, _, files in os.walk(annotation_dir):
                for file in files:
                    if file.lower() == f"{img_id}.json":  # (case-insensitive)
                        annotation_path = os.path.join(root, file)
                        break
                else:
                    continue
                break

            if not annotation_path:
                print("\tNo matching annotation in directory.")
                continue

            with open(annotation_path, 'r', encoding='utf-8') as annotation_file:
                annotation_json = json.load(annotation_file)

                if include_category:
                    print(f"\tCategory: \t{annotation_json.get('category', '')}")
                if include_safety:
                    print(f"\tSafety: \t{annotation_json.get('rating', '')}")
                if include_rationale:
                    print(f"\tRationale: \t{annotation_json.get('rationale', '')}")


def save_json_annotations(annotations: list[str], output_dir: str, file_names: list[str] = None):
    """
    Processes a list of JSON strings and saves each valid entry as a .json file.
    Invalid JSON strings are stored as .txt files.

    Parameters:
    - json_list (list): List of JSON strings.
    - output_dir (str): Directory to save the .json and .txt files.

    Returns:
    - List of file names where annotations are not valid JSON.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Ensure there is exactly one name per file
    if file_names:
        assert len(annotations) == len(file_names)
    else:
        file_names = [str(idx) for idx in range(len(annotations))]

    invalid_json = []

    for json_str, file_name in zip(annotations, file_names):
        try:
            # Attempt to parse the JSON string
            data = json.loads(json_str)
            
            # Generate a file name
            file_path = os.path.join(output_dir, f"{file_name}.json")
            
            # Write the valid JSON data to a file
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4)
        except json.JSONDecodeError:
            # Handle invalid JSON: Save as .txt file
            invalid_json.append(file_name)

            # Generate a file name
            file_path = os.path.join(output_dir, f"{file_name}.txt")
            
            # Write the invalid JSON string to a .txt file
            with open(file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(json_str)

    return invalid_json


def compress_annotations(annotation_dir: str | os.PathLike, output_dir: str | os.PathLike, parquet_size: int = 100000) -> list[str | os.PathLike]:
    """
    Compress all JSON annotations in a directory and its subdirectories into one or multiple parquet files. 
    The annotations are expected to include the following keys: "rating", "category", "rationale" and to be named after 
    the id of the image they describe.
    The output file(s) will be one or multiple parquet files containing the same information as the input files.
    
    :param annotation_dir: The directory containing the JSON files.
    :param output_dir: The directory where the output parquet file(s) are created.
    :param parquet_size: The maximum number of annotations to store in each parquet file.
    :return: The path(s) to the output parquet file(s).
    """

    def write_to_parquet(annotations: list[dict], output_path: str) -> None:
        """
        Write a list of annotations to a parquet file.
        
        :param annotations: The list of annotations to write.
        :param output_path: The path to the output parquet file.
        """
        df = pd.DataFrame(annotations)

        # Set the column 'id' as index of the dataframe
        df.set_index('id', inplace=True)

        # Mark columns as categorical to save memory
        df['rating'] = df['rating'].astype('category')
        df['category'] = df['category'].astype('category')
        df['rationale'] = df['rationale'].astype('category')

        df.to_parquet(output_path)

    def get_output_path(idx: int, num_parquet_files: int) -> str:
        """
        Get the path to the output parquet file.
        
        :param idx: The index of the parquet file.
        :param num_parquet_files: The total number of parquet files.
        :return: The path to the output parquet file.
        """
        annotation_dir_name = os.path.basename(annotation_dir)

        if num_parquet_files == 1:
            return os.path.join(output_dir, f"{annotation_dir_name}.parquet")
        else:
            num_digits = len(str(num_parquet_files))

            return os.path.join(output_dir, f"{annotation_dir_name}.part_{str(idx).zfill(num_digits)}_of_{str(num_parquet_files).zfill(num_digits)}.parquet")

    # Count the number of annotations
    num_annotations = 0

    for root, _, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith(".json"):
                num_annotations += 1

    print(f"Found {num_annotations} annotations in {annotation_dir}")

    # Calculate the number of parquet files to create
    num_parquet_files = num_annotations // parquet_size + min(1, num_annotations % parquet_size)
    print(f"Creating {num_parquet_files} parquet file(s)")
    
    # Load all annotations in batches of size parquet_size
    annotations = []
    idx_parquet = 1
    parquet_paths = []

    for root, _, files in os.walk(annotation_dir):
        for file in files:
            if file.endswith(".json"):
                img_id = file.removesuffix(".json").removesuffix(".jpg").removesuffix(".jpeg").removesuffix(".png")
                
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)

                    data['id'] = img_id

                    annotations.append(data)

                # Write the annotations to a parquet file
                if len(annotations) == parquet_size:
                    parquet_path = get_output_path(idx_parquet, num_parquet_files)
                    write_to_parquet(annotations, parquet_path)
                    parquet_paths.append(parquet_path)
                    idx_parquet += 1
                    annotations = []

    # Write the remaining annotations to a parquet file
    if annotations:
        parquet_path = get_output_path(idx_parquet, num_parquet_files)
        write_to_parquet(annotations, parquet_path)
        parquet_paths.append(parquet_path)
    
    print("Finished creating {0} parquet file(s):\n\t{1}".format(len(parquet_paths), '\n\t'.join(parquet_paths)))

    return parquet_paths