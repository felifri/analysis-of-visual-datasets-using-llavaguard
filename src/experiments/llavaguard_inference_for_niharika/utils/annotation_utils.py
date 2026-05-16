import json
import os
from utils.file_utils import get_file_paths

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