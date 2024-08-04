from PIL import Image
import matplotlib.pyplot as plt
import os
num = 466986
padded_num = str(num).zfill(12)
f = f"/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_{padded_num}.jpg"

output_dir = "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/coco"

img = Image.open(f)

plt.imshow(img)

plt.savefig(os.path.join(output_dir, f"{num}.jpg"))











