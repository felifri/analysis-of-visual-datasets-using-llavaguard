from experiments.datasets.BaseImageLoader import BaseImageLoader
from PIL.Image import Image
import os
import tarfile

class ImageLoader(BaseImageLoader):
    def __init__(self, tar_path: str | os.PathLike):
        self.tar_path = tar_path

    def load_image(self, idx: str, split: list[str] | None = None) -> list[Image | None]:
        raise NotImplementedError("This method is not implemented for models. Use extract_image_batch instead.")

    def extract_image_batch(self, target_image_names: list[str], output_dir: str | os.PathLike, output_name_prefixes: list[str] | None = None) -> None:
        print(f"Extracting {len(target_image_names)} images to {output_dir}")

        assert len(target_image_names) == len(output_name_prefixes) if output_name_prefixes is not None else True, \
            "target_image_names and output_name_prefixes must have the same length if output_name_prefixes is provided."

        with tarfile.open(self.tar_path, 'r') as tar:
            for i, image_name in enumerate(target_image_names):
                try:
                    member = tar.getmember(f"images/{image_name}.jpg")

                    extracted = tar.extractfile(member)
                    if extracted is None:
                        # print(f"Warning: {image_name} not found in {tarfile_name}")
                        continue

                    output_name = f"{output_name_prefixes[i]}{image_name}.jpg" if output_name_prefixes else f"{image_name}.jpg"
                    with open(os.path.join(output_dir, output_name), 'wb') as f:
                        f.write(extracted.read())
                except KeyError:
                    # print(f"{image_name} not found in {tarfile_name}")
                    pass
