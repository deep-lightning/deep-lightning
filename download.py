import sys
from pathlib import Path

import hub
import cv2
from tqdm import tqdm

from common import required_images

if len(sys.argv) == 2:
    _, source = sys.argv
    target = source
elif len(sys.argv) == 3:
    _, source, target = sys.argv
else:
    print("Need source to download")
    sys.exit()

ds = hub.load(f"hub://deep-lightning/{source}", read_only=True)
dataset_path = Path(target)

try:
    dataset_path.mkdir()
    with ds:
        for key, value in required_images.items():
            for data in tqdm(ds, desc=f"Downloading {key} images"):
                current = dataset_path / data.folder.data()
                current.mkdir(exist_ok=True)
                cv2.imwrite(str((current / value).resolve()), data[key].data())
except FileExistsError:
    print("Folder already exists")
