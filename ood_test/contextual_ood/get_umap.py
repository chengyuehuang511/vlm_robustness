from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
# import umap.umap_ as umap
import umap


splits =[
    #ds_name, split, file_name
    #(train, test, concept)
    #vqav2 train with all others
    #train_stuff = sample[0], test_stuff = sample[1]
    [("vqa_v2","train"), ("vqa_v2","train")],
    [("vqa_v2","train"), ("vqa_v2","val")], 
    [("vqa_v2","train"), ("advqa", "test")],
    [("vqa_v2","train"), ("cvvqa", "test")], 
    [("vqa_v2","train"), ("ivvqa", "test")],
    [("vqa_v2","train"), ("okvqa", "test")], 
    [("vqa_v2","train"), ("textvqa", "test")], 
    [("vqa_v2","train"), ("vizwiz", "test")], 
    [("vqa_v2","train"), ("vqa_ce", "test")], 
    [("vqa_v2","train"), ("vqa_rephrasings", "test")],
    # [("vqa_v2","train"), ("vqa_vs", "id_val")], 
    # [("vqa_v2","train"), ("vqa_v2","test")], 
    [("vqa_v2", "train"), ("vqa_cp", "test")]
    # [("vqa_v2", "train"), ("vqa_lol", "test")]
]


concept_type = ["image", "joint"]

for split in tqdm(splits, desc="Processing splits"): 
    print("Running", split)
    train_split = split[0]
    train_ds_name, train_split = split[0]
    test_ds_name, test_split = split[1]

    train_ds_split = f"{train_ds_name}_{train_split}" 
    test_ds_split = f"{test_ds_name}_{test_split}"

    train_vectors = torch.load(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{train_ds_split}_joint_image.pth")
    test_vectors = torch.load(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{test_ds_split}_joint_image.pth")
    if train_vectors.dtype == torch.bfloat16: 
        train_vectors = train_vectors.to(dtype=torch.float32)
        torch.save(train_vectors, f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{train_ds_split}_joint_image.pth")
    if test_vectors.dtype == torch.bfloat16:
        test_vectors = test_vectors.to(dtype=torch.float32)
        torch.save(test_vectors, f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{test_ds_split}_joint_image.pth")

    dataset_labels = [train_ds_name, test_ds_name] 

    # datasets = [train_vectors, test_vectors]  
    print(train_vectors.size())
    #train_vectors : (n, 256)
    for i in tqdm(range(len(concept_type)), desc="Processing concept types", leave=False): 
        cur_train_vectors = train_vectors[i]
        cur_test_vectors = test_vectors[i]
        n_train = train_vectors.size(1)
        combined_emb = torch.cat((cur_train_vectors, cur_test_vectors), dim=0)

        combined_emb = combined_emb.numpy() 

        # Initialize UMAP
        reducer = umap.UMAP(random_state=42, verbose=True,low_memory=True)

        # Fit and transform the data
        embedding = reducer.fit_transform(combined_emb)

        # combined_emb = tsne.fit_transform(combined_emb)
        train_emb = combined_emb[:n_train]
        test_emb = combined_emb[n_train:]
        plt.figure(figsize=(10,10))
        plt.scatter(train_emb[:, 0], train_emb[:,1], c="blue", label=train_ds_split, s=10)
        plt.scatter(test_emb[:, 0], test_emb[:,1], c="red", label=test_ds_split, s=10)

        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title(f"UMAP plot {concept_type[i]} - {train_ds_split}, {test_ds_split}")

        plt.legend()

        plt.savefig(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/umap/annot_{concept_type[i]}_{train_ds_split}_{test_ds_split}.jpg")
        plt.close()

