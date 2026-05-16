import os

def get_file_paths(search_dir, file_extension: str, recursive: bool = True):
    """
    Retrieves all paths to files with the specified file extension in the specified directory.
    
    Parameters:
    - search_dir: Directory to search in.
    - file_extension: File extension to search for.
    - recursive: Whether to search subdirectories recursively (default: True).
    
    Returns:
    - A tuple containing:
      1. List of os.PathLike objects pointing to the found files.
      2. List of filenames without extensions.
    """
    file_paths = []
    file_names = []

    if recursive:
        # Walk through the directory recursively
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(file_extension):  # (case-insensitive)
                    file_paths.append(os.path.join(root, file))
                    file_names.append(os.path.splitext(file)[0])
    else:
        # Search only the top-level directory
        for file in os.listdir(search_dir):
            full_path = os.path.join(search_dir, file)
            if os.path.isfile(full_path) and file.lower().endswith(file_extension):
                file_paths.append(full_path)
                file_names.append(os.path.splitext(file)[0])

    return file_paths, file_names
