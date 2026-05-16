from experiments.datasets.BaseImageLoader import BaseImageLoader
from PIL.Image import Image
import os
import tarfile

class ImageLoader(BaseImageLoader):
    def __init__(self):
        self.base_img_dir = "/pfss/mlde/workspaces/mlde_wsp_Shared_Datasets/lsun/images"
        self.split_names = ["train", "val"]
        self.categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']

    def load_image(self, idx: str, split: list[str] | None = None) -> list[Image | None]:
        raise NotImplementedError("This method is not implemented for LSUN dataset. Use load_images_batch instead.")
    
    # def extract_image_batch(self, target_image_names: list[str], output_dir: str | os.PathLike):
    #     print(f"Extracting {len(target_image_names)} images to {output_dir}")

    #     # Directory where .tar archives are stored
    #     tar_dir = os.path.join(self.base_img_dir, "train")

    #     # Determine top-level tar archive name (e.g., 'bedroom' from your context)
    #     # Here, you might have a mapping or logic to determine which .tar to look into
    #     for category in self.categories:
    #         tarfile_name = f"{category}.tar"
    #         print(f"Processing {tarfile_name}...")

    #         os.makedirs(os.path.join(output_dir, category), exist_ok=True)

    #         # Check if the tar file contains this file
    #         tar_path = os.path.join(tar_dir, tarfile_name)
    #         with tarfile.open(tar_path, 'r') as tar:
    #             for image_name in target_image_names:
    #                 # Determine relative path inside the tar
    #                 rel_path = f"{'/'.join(image_name[:6])}/{image_name}.jpg"
    #                 full_path_in_tar = f"{os.path.splitext(tarfile_name)[0]}/{rel_path}"
                    
    #                 try:
    #                     member = tar.getmember(full_path_in_tar)

    #                     extracted = tar.extractfile(member)
    #                     if extracted is None:
    #                         # print(f"Warning: {image_name} not found in {tarfile_name}")
    #                         continue

    #                     with open(os.path.join(output_dir, category, image_name + ".jpg"), 'wb') as f:
    #                         f.write(extracted.read())
    #                 except KeyError:
    #                     # print(f"{image_name} not found in {tarfile_name}")
    #                     pass

    def extract_image_batch(self, target_image_names: list[str], output_dir: str | os.PathLike):
        print(f"Extracting {len(target_image_names)} images to {output_dir}")

        tarfile_name = "val.tar"
        tar_path = os.path.join(self.base_img_dir, tarfile_name)
        print(f"Processing {tarfile_name}...")            

        with tarfile.open(tar_path, 'r') as tar:
            for category in self.categories:
                for image_name in target_image_names:
                    # Determine relative path inside the tar
                    rel_path = f"{'/'.join(image_name[:6])}/{image_name}.jpg"
                    full_path_in_tar = f"{os.path.splitext(tarfile_name)[0]}/{category}/{rel_path}"
                    print(f"Extracting {image_name} from {full_path_in_tar}")
                    
                    try:
                        member = tar.getmember(full_path_in_tar)

                        extracted = tar.extractfile(member)
                        if extracted is None:
                            # print(f"Warning: {image_name} not found in {tarfile_name}")
                            continue

                        with open(os.path.join(output_dir, category, image_name + ".jpg"), 'wb') as f:
                            f.write(extracted.read())
                    except KeyError:
                        # print(f"{image_name} not found in {tarfile_name}")
                        pass
