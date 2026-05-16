import os
import time
import requests
import logging
import rtpt
from datasets import load_dataset
from tqdm.contrib.concurrent import thread_map

print(os.environ['HF_HOME'])

if "finngu" in os.environ['HF_HOME']:
    print("HF configured locally")
else:
    print("HF uses shared dir")

dataset = load_dataset("stylebreeder/stylebreeder", split='4M_full')

print("Successfully loaded all arrow files.")
# print(dataset['4M_full'].filter(lambda row: row['image_id'] == 295543312)[:1])

def download_image(image_folder, image_key, width=384, height=384):
    image_url = f"https://artbreeder.b-cdn.net/imgs/{image_key}.jpeg?width={width}&height={height}"
    save_path = os.path.join(image_folder, f"{image_key}.jpeg")

    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
        else:
            raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        logging.error(f"Failed to download image {image_key}: {str(e)}")

def throttled_downloader(image_folder, image_key, width=384, height=384, rate_limit=50):
    """
    Wrapper to throttle the download rate. Limits network usage dynamically.
    """
    start_time = time.time()
    download_image(image_folder, image_key, width, height)
    elapsed = time.time() - start_time
    sleep_time = max(0, (1 / rate_limit) - elapsed)  # Ensure no excess bandwidth is used
    time.sleep(sleep_time)

start_shard = 1
num_shards = 400

print(f"Downloading all 4M images in 384x384 resolution in {num_shards} shards.")
rt = rtpt.RTPT(name_initials='FG', experiment_name=f'Download Stylebreeder dataset', max_iterations=400)
rt.start()

base_folder = os.path.join(os.environ['HF_HOME'], "datasets", "stylebreeder___stylebreeder", "default", "0.0.0", "images")
os.makedirs(base_folder, exist_ok=True)  # Ensure dir exists, but don't fail if it is already present

# Configure logging for failed downloads
log_file = os.path.join(base_folder, "failed_downloads.log")
logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - %(message)s')

for shard_idx in range(start_shard, num_shards):
    shard = dataset.shard(num_shards=num_shards, index=shard_idx)
    print(f"Processing shard {shard_idx + 1}/{num_shards} with {shard.num_rows} entries.")

    # Define the base image folder
    image_folder = os.path.join(base_folder, str(shard_idx).zfill(3))
    os.makedirs(image_folder, exist_ok=False)  # Prevent overriding data

    # dd = dataset.select_columns(['image_key']).to_pandas()
    # dd.image_key.apply(lambda image_key: download_image(image_folder, image_key))
    
    # Use thread_map for concurrent downloads
    image_keys = shard.select_columns(["image_key"])["image_key"]

    thread_map(
        lambda key: throttled_downloader(image_folder, key),
        image_keys,
        max_workers=128
    )

    rt.step(subtitle=f"Processing shard {shard_idx + 2}/{num_shards}.")
