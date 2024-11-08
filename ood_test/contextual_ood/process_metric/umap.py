import cupy as cp 
import cuml
from cuml.manifold.umap import UMAP
import matplotlib.pyplot as plt 
import os
import torch
#load data onto GPU 

root_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states"
file_path = "fft/coco_advqa_val_image_joint_new.pth"

test_hidden_state = os.path.join(root_dir, file_path)
train_hidden_state = os.path.join(root_dir, "fft/coco_vqav2_train_val_image_joint_new.pth")

test_dict = torch.load(test_hidden_state) 
train_dict = torch.load(train_hidden_state)

train_tensor = torch.cat([value[1,:].unsqueeze(0) for _,value in train_dict.items()], dim=0)
test_tensor = torch.cat([value[1,:].unsqueeze(0) for _,value in test_dict.items()], dim=0)

data = cp.array(torch.cat([train_tensor, test_tensor], dim = 0).numpy())


umap = UMAP(n_neighbors=15)
emb  = umap.fit_transform(data)

plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.scatter(emb[:, 0], emb[:, 1], s=10, alpha=0.8)  # Customize marker size (s) and transparency (alpha)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP Embeddings')
plt.show()
