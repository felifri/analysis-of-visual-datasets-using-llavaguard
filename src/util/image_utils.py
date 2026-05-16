from PIL import Image

def is_image_valid(path):
    try:
        Image.open(path)
        return True
    except IOError:
        return False