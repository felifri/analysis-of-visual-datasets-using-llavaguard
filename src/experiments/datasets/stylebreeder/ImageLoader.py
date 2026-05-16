from experiments.datasets.BaseImageLoader import BaseImageLoader
from PIL import Image, UnidentifiedImageError
import requests

class ImageLoader(BaseImageLoader):
    def __init__(self):
        self.split_names = ["4M_full"]

    def load_image(self, img_key: str) -> list[(str, Image.Image | None)]:
        image_url = f"https://artbreeder.b-cdn.net/imgs/{img_key}.jpeg?width={256}&height={256}"
        split = "4M_full"
        
        try:
            im = Image.open(requests.get(image_url, stream=True).raw)
            return [(split, im)]
        except UnidentifiedImageError:
            print(f"UnidentifiedImageError: {image_url}")
            return [(split, None)]
