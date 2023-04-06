import os
from PIL import Image
from tqdm import tqdm

RAW_DIR = "/media/mera/Mera/AI/Selfcode/Anime-Face-Generation-DCGAN/data_raw"
CLEAN_DIR = "/media/mera/Mera/AI/Selfcode/Anime-Face-Generation-DCGAN/data"

## Filter if size smaller than 64 and resize to 64
for image in tqdm(os.listdir(RAW_DIR)):
    image_path = os.path.join(RAW_DIR, image)
    save_path = os.path.join(CLEAN_DIR, image)
    pil_image = Image.open(image_path)
    w, h = pil_image.size
    if w >= 64 or h >= 64:
        pil_image = pil_image.resize((64, 64))
        pil_image.save(save_path)