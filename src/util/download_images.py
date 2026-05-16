import os
import requests
from urllib.parse import urlparse

def download_images(image_urls: list[str], download_folder: str = 'test_images', filename_handler: callable = None):
    """
    Downloads images from a list of URLs and saves them to a specified folder.
    
    Parameters:
    - image_urls: List of image URLs to download.
    - download_folder: Folder to save the downloaded images.
    - filename_handler: A callable that takes a URL as input and returns a filename.
    
    Returns:
    - List of paths to the downloaded images.
    """

    # Create the folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    # List to store paths to the downloaded images
    downloaded_files = []
    
    for idx, url in enumerate(image_urls):
        try:
            # Fetch the image content
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Determine the file name
            if filename_handler:
                file_name = filename_handler(url)
            else:
                # Default filename handling: extract from URL path
                parsed_url = urlparse(url)
                file_name = os.path.basename(parsed_url.path)
                if not file_name or '.' not in file_name:
                    file_name = f"image_{idx + 1}.jpg"  # Default to .jpg if no valid name
            
            # Create a file path
            file_path = os.path.join(download_folder, file_name)
            
            # Save the image
            with open(file_path, 'wb') as file:
                file.write(response.content)
            
            # Append the file path to the list
            downloaded_files.append(file_path)
            print(f"Downloaded: {file_name}")
        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")
    
    # Return the list of downloaded file paths
    return downloaded_files
