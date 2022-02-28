import re
import sys
import hub
import cv2

from pathlib import Path
from tqdm import tqdm

from common import required_images

if len(sys.argv) == 3:
    _, source, target = sys.argv
    print(source, target)
else:
    print("Need source and target to upload")
    sys.exit()

dataset_path = Path(source)
ds = hub.empty(f"hub://deep-lightning/{target}", overwrite=True)

with ds:
    ds.create_tensor("folder", dtype="str")
    for img in required_images:
        ds.create_tensor(img, htype="image", dtype="float32", sample_compression="lz4")

    folders = list(dataset_path.glob("[!.]*"))
    folders.sort(key=lambda x: (int(re.findall("[0-9]+", str(x))[-1])))

    for folder in tqdm(folders):
        ds["folder"].append(str(folder.stem))
        for key, value in required_images.items():
            ds[key].append(cv2.imread(str((folder / value).resolve()), flags=cv2.IMREAD_ANYDEPTH))

del ds
