from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig,AutoTokenizer, AutoModel

from PIL import Image
import os 
import torch 
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import gc 
import math 
import torch.nn.functional as F
# from measure import score_func

# from data.builders import load_dataset, DatasetZoo

# dataset_zoo = DatasetZoo()
# # coco_vqa_vs
# vqa_vs = load_dataset("coco_vqa_vs")
torch.cuda.empty_cache()

#globally set no gradient tracking
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GLOBAL_CONCEPT = "joint"
INCLUDE_ANSWER = False
COR_SHIFT = True 
ds_2_img = { 
    "advqa" : "/srv/datasets/coco/", 
    "cvvqa" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/IMAGES_counting_del1_edited_VQA_v2/",
    "vqa_v2" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "ivvqa": "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/final_edited_VQA_v2/Images/", 
    "okvqa" : "/srv/datasets/coco/",
    "textvqa" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/", 
    "vizwiz" : "/srv/datasets/vizwiz/data/Images/", 
    "vqa_ce" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_cp" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_lol" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_rephrasings" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/",
    "vqa_vs" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/"
}

ds_split_2_file = { 
    "vqa_v2_train" : "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json",
    "vqa_v2_val" : "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_eval.json", 
    "vqa_v2_test": "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json" , 
    "advqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json", 
    "cvvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/combined_data.json", 
    "ivvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/combined_data.json",
    "okvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/ok-vqa/val/combined_data.json",
    "textvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/combined_data.json",
    "textvqa_train" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/train/combined_data.json", 
    "vizwiz_test" :  "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/combined_data.json",
    "vqa_ce_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/combined_data_subset.json",
    "vqa_cp_train" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_questions.json", 
    "vqa_cp_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/combined_data.json", 
    "vqa_lol_train": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/train/combined_data.json",
    "vqa_lol_test": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/test/combined_data.json", 
    "vqa_rephrasings_test": "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqa_rephrasings/combined_data.json",
    "vqa_vs_train" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/combined_data.json", 
    "vqa_vs_id_val" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/combined_data.json", 
    "vqa_vs_id_test" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
    "vqa_vs_ood_test" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
    "vqa_vs_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/combined_data.json",
    "vqa_vs_KOP" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/combined_data.json", 
    "vqa_vs_KW" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/combined_data.json", 
    "vqa_vs_KW_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/combined_data.json", 
    "vqa_vs_KWP" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/combined_data.json", 
    "vqa_vs_QT" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/combined_data.json", 
    "vqa_vs_QT_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/combined_data.json", 
    "vqa_vs_QT_KW" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/combined_data.json", 
    "vqa_vs_QT_KW_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/combined_data.json"
}

ans_label_map = {} 
id_2_ans_map = {} 

#store ans_label_map in json 


#don't forget to put model to device 
model_id = "google/paligemma-3b-ft-vqav2-224"

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#quantize model to reduce size 
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, 
                                                        # device_map ="auto",
                                                        # torch_dtype=torch.bfloat16,
                                                        # revision="bfloat16").eval()

# question_model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)

processor = PaliGemmaProcessor.from_pretrained(model_id)

# output = model.generate(**inputs, max_new_tokens=20)
# # print(output)
# print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])

#image_tokens = 256, text_tokens = question length, SEP/EOS token = 2 

#dataloader + processor : 

IMAGE_SIZE = 224
BATCH_SIZE = 1000
HIDDEN_SIZE = 2048 
COMP_BATCH_SIZE = 1000 

resize_transform = transforms.Resize(
    (IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC
)
def score_func(train_vectors, test_vectors,cov_path_file, label_id, metric="maha", peranswerType=False) : 
    #do them in batches to avoid memory error 
    
    print("Train vector size", train_vectors.size())
    print("Test vector size", test_vectors.size())

    #perAnswerType Logic
    #train_vectors : (#concepts C , batch size N, hidden size H)
    #test_vectors : (#concepts, batch size, hidden size)
    train_vectors = train_vectors.to(device) 
    # train_vectors = train_vectors
    print(train_vectors.dtype)
    u_vector = torch.mean(train_vectors, dim=1, keepdim=True) #find mean in vector (concept, batch size , hidden size) -> (C, 1, H)
    print("Size of mean vector", u_vector.size())
    print(torch.cuda.memory_allocated())

    if torch.isinf(u_vector).any() or torch.isnan(u_vector).any() : 
        raise Exception("u vector has NaN values")

    train_vectors = train_vectors.detach().cpu()   
    c, n, h = train_vectors.size() 
    print(f"size (num_concept, samples, hidden size) : {c, n , h}")
    
    #store u_vectors for certain things? 
    n_batch = n // COMP_BATCH_SIZE 
    final_cov = torch.zeros(c,h,h) #(C,H,H)

    print("========================")
    print(f"TRAINING SAMPLE {n} samples")
    if metric == "maha" :
        label_exist = False 
        cov_dict = {}
        if os.path.exists(cov_path_file) : 
            cov_dict = torch.load(cov_path_file)
            if label_id in cov_dict : 
                inv_cov = cov_dict[label_id]
                label_exist = True 
        
        if not os.path.exists(cov_path_file) or not label_exist : 
            for i in range(n_batch) : 
                start_ind = i * COMP_BATCH_SIZE
                end_ind = min(n - 1, (i+1)*COMP_BATCH_SIZE)
                print(f"Running batch {start_ind} - {end_ind}")
                cur_vectors = train_vectors[:, start_ind : end_ind, :]
                cur_vectors = cur_vectors.to(device)

                mean_vector = u_vector.expand(c, cur_vectors.size(1), h)
                
                diff =  cur_vectors - mean_vector  #f - u_c : (C,N ,H)

                diff = diff.permute(0,2,1)  #(C, H, N)
                print("diff size", diff.size())

                #covariance matrix : * = element wise multiplication 
                print("diff dtype", diff.dtype)
                cov = torch.matmul(diff, diff.permute(0,2,1))  #(C, H, N) x (C, N, H) -> (C, H, H)
                print("covariance matrix size", cov.size())
                cov  = cov.detach().cpu() #(C, H, H)
                final_cov = final_cov + cov 

            final_cov = final_cov / n

            print("min val cov", torch.min(final_cov))
            print("max val cov", torch.max(final_cov))
            
            if torch.isinf(final_cov).any() or torch.isnan(final_cov).any() : 
                raise Exception("cov matrix has NaN values")
            
            inv_cov = torch.empty(c,h,h) #(C, H, H)
            for i in range(c) :
                # reg_matrix = final_cov[i] + torch.eye(final_cov[0].size(1)) 
                #regularize matrix to ensure positive semi-definite 
                reg_term = 1e-8 * torch.eye(final_cov[i].size(-1)) #(H, H)
                reg_cov = final_cov[i] + reg_term #(H, H)

                inv_cov[i] = torch.linalg.pinv(reg_cov) # Σ^-1

            print("min val cov", torch.min(inv_cov))
            print("max val cov", torch.max(inv_cov))

            if torch.isinf(inv_cov).any() or torch.isnan(inv_cov).any() : 
                raise Exception("inv cov matrix has NaN values")
            
            cov_dict[label_id] = inv_cov 

            torch.save(cov_dict, cov_path_file)
            del cov
        
    del train_vectors 
    
    total_res = torch.zeros(c)
    c, n_test, _ = test_vectors.size()
    print("========================")
    print(f"TESTING SAMPLE {n_test} samples")
    n_batch = n_test // COMP_BATCH_SIZE
    #maha metric 
    

    if metric == "euclidean" : 
        inv_cov = torch.eye(h)
    inv_cov = inv_cov.to(dtype=test_vectors.dtype)
    inv_cov = inv_cov.to(device)

    test_results_vectors = [] #(c, n, 1)
    #test batches 
    for j in range(n_batch): 
        start_ind = j * COMP_BATCH_SIZE
        end_ind = min(n_test - 1, (j+1) * COMP_BATCH_SIZE)
        print(f"Running batch {start_ind} - {end_ind}")
        cur_vectors = test_vectors[:, start_ind : end_ind, :] #(C, comp_batch_size, h)
        cur_vectors = cur_vectors.to(device)
        mean_vector = u_vector.expand(c, cur_vectors.size(1),h)

        #Z_test - μ 
        test_diff = (cur_vectors - mean_vector)#(concept, BATCH SIZE, hidden size)
        print("verify test diff size", test_diff.size())
        res_1 = torch.matmul(test_diff, inv_cov) #(concept, BATCH SIZE, hidden size)
        res_2 = torch.matmul(res_1, test_diff.permute(0, 2, 1)) #(concept, BATCH SIZE, BATCH SIZE)


        res = torch.zeros(c)
        cur_batch_results = None
        for i in range(c) : 
            diag = torch.diag(res_2[i]) #(1, d)
            #verify correctness via shape 
            print("diag size",diag.size())

            if (diag < 0).any() : 
                raise Exception("Diagonal values can't be negative")
            
            assert diag.size(0) == cur_vectors.size(1) , "mismatch shape in diagonal values and n samples" 

            #avg dist across all samples 

            maha_score = -1 * torch.sqrt(diag) # (batch_size)

            if cur_batch_results == None : 
                cur_batch_results = maha_score.unsqueeze(0)
                print("verify maha score shape", maha_score.size())

            else : 
                cur_batch_results = torch.cat([cur_batch_results, maha_score.unsqueeze(0)], dim = 0) #will be (c, n)
            
            total_score = torch.sum(maha_score) #sum maha score across all batch samples 
            total_score = total_score.item() #would this be on the CPU now 
            res[i] = total_score #(1, c)

        test_results_vectors.append(cur_batch_results) #(c, n)

        total_res = total_res + res 
        print("Done a sample")
        del res_1 
        del res_2
        torch.cuda.empty_cache()

    test_results_vectors = torch.cat(test_results_vectors, dim=1) #(c, n)
    total_res = total_res / n_test #average maha score across all test samples
    
    #next perAnswerType 

    
    del inv_cov 
    del test_vectors

    # print("scores shape", res.size()) #(c, )
    print(f"res shape : {res.size()}")
    # return avg score for each concept + (c, n, 1) 
    return total_res.tolist(), test_results_vectors #(concept, 1) 

class MeasureOODDataset(Dataset) : 
    global ans_label_map
    global id_2_ans_map
    
    def __init__(self, data, ds_name, ans_label_map = {}, id_2_ans_map = {}):
        self.data = data #list of dictionary elements 
        self.ds_name = ds_name 
        self.vis_root = ds_2_img[ds_name]
        self.ans_label_map = ans_label_map 
        self.id_2_ans_map = id_2_ans_map 

        # co = len(ans_label_map)
        # for sample in self.data : 
        #     try : 
        #         label = sample["answer"]
        #         if isinstance(sample["answer"], list) : 
        #             label = Counter(sample['answer']).most_common(1)[0][0]
        #             print("verify label", label)

        #         if label not in self.ans_label_map : 
        #             self.ans_label_map[label] = co 
        #             self.id_2_ans_map[co] = label
        #             co += 1 

        #     except Exception as e : 
        #         raise Exception(self.ds_name, "doesn't have an answer key")
            



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if "answer" in sample : 
            label = sample["answer"]
            if isinstance(sample["answer"], list) : 
                label = Counter(sample['answer']).most_common(1)[0][0]
                print("verify label", label)

        else : 
            label = 0 
            print("NOOO Dataset doesn't contain answer labels")

        question = sample["question"]
        if GLOBAL_CONCEPT == "joint" : 
            image_path = os.path.join(self.vis_root , sample["image"]) 
            input_image = Image.open(image_path).convert('RGB')
            resized_image = resize_transform(input_image)

            label = sample["answer"]
            if isinstance(sample["answer"], list) : 
                label = Counter(sample['answer']).most_common(1)[0][0]
                print("verify label", label)
            # inputs = processor(question, resized_image, padding=True)

            #convert label to id for conversion to tensors
            # if label not in ans_label_map : 
                # print("converting label")
                # n = len(ans_label_map)
                # ans_label_map[label] = n #{word: idx}
                # id_2_ans_map[n] = label
                # print("Updated ans_label_map:", len(ans_label_map))  # Show the state after update
                # print(ans_label_map.keys())

            print("label ", label)
            if INCLUDE_ANSWER : 
                question = f"{question} : {label}"
                print("new question: ", question)

            label = self.ans_label_map[label]
            
            return question, resized_image, label #must return tensor to parallelize 
    
        return question 
    @staticmethod 
    def collate_fn(batch):
        if GLOBAL_CONCEPT == "joint" : 
            questions, images, labels = zip(*batch)

            inputs = processor(text=questions, images=images, padding=True)
            # print(type(labels), labels.size())
            # print(labels)
            labels = torch.tensor(labels)
            #extract inputs if needed 
            # print("DATALOADERR Verify", ans_label_map)

            return inputs, labels
        else : 
            # print("helloo")
            questions = batch #didn't do processing with images? 
            # print("question", questions)
            return questions 
    


#measure ood score : 
"""
train ds name, split 
test ds name, split

concept : joint/image/question etc. 

"""

#how to organize when you get the score 
concept_type = ["joint", "image"] #
visited = []

answer_list_file = "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/answer_list.json"
splits =[
    #ds_name, split, file_name
    #(train, test, concept)
    #vqav2 train with all others
    #train_stuff = sample[0], test_stuff = sample[1]
    # [("vqa_v2","train"), ("vqa_v2","train")],
    [("vqa_v2","train"), ("vqa_v2","val")], 
    [("vqa_v2","train"), ("advqa", "test")],
    [("vqa_v2","train"), ("cvvqa", "test")],
    [("vqa_v2","train"), ("ivvqa", "test")],
    [("vqa_v2","train"), ("okvqa", "test")], 
    [("vqa_v2","train"), ("textvqa", "test")], 
    [("vqa_v2","train"), ("vizwiz", "test")], 
    [("vqa_v2","train"), ("vqa_cp", "test")], 
    [("vqa_v2","train"), ("vqa_ce", "test")], 
    
    [("vqa_v2","train"), ("vqa_rephrasings", "test")],
    [("vqa_v2","train"), ("vqa_vs", "id_val")]
    # [("vqa_v2","train"), ("vqa_v2","test")],

    # [("vqa_v2","train"), ("vqa_vs", "KO")],
    # [("vqa_v2","train"), ("vqa_vs", "KOP")],
    # [("vqa_v2","train"), ("vqa_vs", "KW")],
    # [("vqa_v2","train"), ("vqa_vs", "KW_KO")],
    # [("vqa_v2","train"), ("vqa_vs", "KWP")],
    # [("vqa_v2","train"), ("vqa_vs", "QT")],
    # [("vqa_v2","train"), ("vqa_vs", "QT_KO")],
    # [("vqa_v2","train"), ("vqa_vs", "QT_KW")],
    # [("vqa_v2","train"), ("vqa_vs", "QT_KW_KO")]
    # [("vqa_v2", "train"), ("vqa_cp", "test")],
    # [("vqa_v2", "train"), ("vqa_lol", "test")]
]

results_file = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/maha_ans_score_dict.json"
if os.path.exists(results_file) : 
    with open(results_file, 'r') as file : 
        results_dict = json.load(file) #read from results dict 
else : 
    results_dict = {} 

hidden_layer_name = ["ans_corr_image", "ans_corr_joint"]

# for i in range(1,20) : 
#     hidden_layer_name.append(f"joint_l{i}")
ans_map_file = '/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/ans_map.json'
id_2_map_file = '/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/id_ans_map.json'

if os.path.exists(ans_map_file) : 
    with open(ans_map_file, 'r') as file : 
        ans_label_map = json.load(file)

    with open(id_2_map_file, 'r') as file : 
        id_2_ans_map = json.load(file)
        id_2_ans_map = {int(k): v for k, v in id_2_ans_map.items()}

                
#store result vector for each tensor size (n_samples, 1) in order index
if __name__ == "__main__" : 

    for measure_instance in splits : 
        # print(torch.cuda.memory_allocated())
        # print("Memory summary", torch.cuda.memory_summary(device=None, abbreviated=False))
        # results_file = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/maha_score_dict.json"
        # if os.path.exists(results_file) : 
        #     with open(results_file, 'r') as file : 
        #         results_dict = json.load(file) #read from results dict 
        # else : 
        #     results_dict = {} 
        
        train_ds_name, train_split = measure_instance[0]
        test_ds_name, test_split = measure_instance[1]

        train_ds_split = f"{train_ds_name}_{train_split}" 
        test_ds_split = f"{test_ds_name}_{test_split}"
        
        train_file = ds_split_2_file[train_ds_split]
        test_file = ds_split_2_file[test_ds_split]
        print(f"Measure Instance {train_ds_split, test_ds_split}")

        # if (f"{train_ds_split}" in results_dict): 
        #     if (f"{test_ds_split}" in results_dict[f"{train_ds_split}"]) : 
        #         # if (concept_type in results_dict[f"{train_split}"][f"{test_split}"]) : 
        #         print(f"already measured")
        #         continue  

        with open(train_file, 'r') as f :
            train_data = json.load(f)

        with open(test_file, 'r') as f : 
            test_data = json.load(f)

        indiv_result_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{train_ds_split}_{'_'.join(concept_type)}.pth"

        # if train_ds_split not in visited : 
        dataset = MeasureOODDataset(train_data, train_ds_name, ans_label_map, id_2_ans_map)
        # concept_type = "joint"

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn, num_workers=1)

        #VQAV2 -> store results of torch
        # train_vectors = []
        train_ans_vectors = [None for i in range(len(id_2_ans_map))] 
        co = 1 
        #indiv results : (C, samples, hidden size)
        train_vectors = torch.load(indiv_result_file)
        i = 0 
        for batch in dataloader : 
            _, labels = batch 
            print(f"Current train batch {co}")

            # break
            
            cur_train_vectors = train_vectors[:, BATCH_SIZE*i: min(BATCH_SIZE*(i+1), train_vectors.size(1) - 1), :] #(c, batchsize, h)
            print("cur_batch_size", cur_train_vectors)
            assert (cur_train_vectors.size(1) == labels.size(0)), "The number of samples in vectors and labels must match."
            
            # inputs = inputs.to(device) #don't forget to add puts to gpu 

            if COR_SHIFT : 
                for label_id in id_2_ans_map : 
                    # print(type(label_id))
                    mask = (labels == int(label_id))
                    indices = torch.nonzero(mask).squeeze() #non zero = condition true 
                    selected_vectors = cur_train_vectors[:, indices,:] #(concept, no. matched samples, hidden size)
                    print("verify select samples size",)
                    if train_ans_vectors[int(label_id)] == None : 
                        train_ans_vectors[int(label_id)] = selected_vectors

                    elif selected_vectors.dim() == 3:
                        print(selected_vectors.size())
                        train_ans_vectors[int(label_id)] = torch.cat([train_ans_vectors[label_id], selected_vectors], dim=1)
            i+=1 
        
            co += 1

        
        visited.append(train_ds_split)
        # ans_label_map = dataset.ans_label_map
        # id_2_ans_map = dataset.id_2_ans_map
        # print(ans_label_map)
        # print("ans_label_map",len(ans_label_map))
        # print("id 2 ans map", len(id_2_ans_map))

        # with open('/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/ans_map.json', 'w') as ans_file : 
        #     json.dump(ans_label_map, ans_file)

        # with open('/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/id_ans_map.json', 'w') as ans_file : 
        #     json.dump(id_2_ans_map, ans_file)
    
        #on cpu now 
        print("verify ans vector size", train_ans_vectors.size())
        #store train ans vectors 
        train_ans_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/ans/{train_ds_split}_{'_'.join(concept_type)}.pth"
        torch.save(train_ans_vectors, train_ans_file)

        # torch.cuda.empty_cache()

        # with open(ans_map_file, 'r') as ans_file : 
        #     ans_label_map = json.load(ans_file)

        # with open(id_2_map_file, 'r') as ans_file : 
        #     id_2_ans_map = json.load(ans_file)
  
        dataset = MeasureOODDataset(test_data, test_ds_name, ans_label_map, id_2_ans_map)
        # concept_type = "joint"

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=dataset.collate_fn, num_workers=1)

        #VQAV2 -> store results of torch
        test_vectors = []
        indiv_result_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{test_ds_split}_{'_'.join(concept_type)}.pth"

        co = 1 
        test_ans_vectors = [None for i in range(len(id_2_ans_map))] 

        #indiv results : (C, samples, hidden size)
        test_vectors = torch.load(indiv_result_file)
        i = 0 
        for batch in dataloader : 
            _, labels = batch 
            print(f"Current test batch {co}")
            co += 1
            # break 

            cur_test_vectors = test_vectors[:, BATCH_SIZE*i: min(BATCH_SIZE*(i+1), test_vectors.size(1) - 1), :] #(c,batchsize, h)
            print("cur_batch_size", cur_test_vectors)
            assert (cur_test_vectors.size(1) == labels.size(0)), "The number of samples in vectors and labels must match."

            # inputs = inputs.to(device) #don't forget to add puts to gpu 

            if COR_SHIFT : 
                for label_id in id_2_ans_map : 
                    mask = (labels == int(label_id))
                    indices = torch.nonzero(mask).squeeze() #non zero = condition true 
                    selected_vectors = cur_test_vectors[:, indices,:] #(concept, no. matched samples, hidden size)
                    print("verify select samples size",)
                    if test_ans_vectors[label_id] == None : 
                        test_ans_vectors[label_id] = selected_vectors

                    elif selected_vectors.dim() == 3 : 
                        test_ans_vectors[label_id] = torch.cat([test_ans_vectors[label_id], selected_vectors], dim=1)
            i+=1 
        
        #on cpu now 
        print("verify ans vector size", test_ans_vectors.size())
        test_ans_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/ans/{test_ds_split}_{'_'.join(concept_type)}.pth"
        torch.save(test_ans_vectors, test_ans_file)
        #store test ans vectors 
        # ans_label_map = dataset.ans_label_map
        # id_2_ans_map = dataset.id_2_ans_map
        # print("ans_label_map",len(ans_label_map))
        # print("id 2 ans map", len(id_2_ans_map))
        # with open('/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/ans_map.json', 'w') as ans_file : 
        #     json.dump(ans_label_map, ans_file)

        # with open('/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/id_ans_map.json', 'w') as ans_file : 
        #     json.dump(id_2_ans_map, ans_file)

        # torch.cuda.empty_cache()
        train_ans_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/ans/{train_ds_split}_{'_'.join(concept_type)}.pth"

        train_vectors = torch.load(train_ans_file)
        n_train = train_vectors.size(1)
        n_test = test_vectors.size(1)
        
        batch_inc = n_train // COMP_BATCH_SIZE
        ans_scores = [0,0]
        ans_count = 0
        answer_shift_list = [0 for i in range(len(id_2_ans_map))]
        #per answer 
        for label_id in id_2_ans_map : 
            train_vectors = train_ans_vectors[label_id]
            test_vectors = test_ans_vectors[label_id] #(c, no. of samples, h)
            print(f"word : {id_2_ans_map[label_id]}, {train_vectors.size()}, {test_vectors.size()}")
            if (train_vectors == None or test_vectors == None) or (train_vectors.size(1) < 10 or test_vectors.size(1) < 10): 
                print("too few words", id_2_ans_map[label_id])

            cov_path_file = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/vqa_v2_ans_inv_cov.pth"

            scores, test_results_vectors = score_func(train_vectors, test_vectors, cov_path_file, label_id) #list of score per concept 
            word_ans = id_2_ans_map[label_id]
            for j in range(len(hidden_layer_name)) : 
                ans_scores[j] += scores[j]
            ans_count += 1
            answer_shift_list[int(label_id)] = scores 
            print(f"Score {scores}") 
        ans_scores = [elem/ans_count for elem in ans_scores] #average score across all answer labels 
        
        with open(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/ans_shift/{test_ds_split}_{'_'.join(concept_type)}.pth", 'w') as ans_file:
            json.dumps(answer_shift_list, ans_file)

        del test_results_vectors
    
        torch.cuda.empty_cache()
        gc.collect()

        #train_vectors, test_vectors : (batch size, concepts, hidden size)
        
        for concept_name, score in zip(hidden_layer_name, ans_scores) : 
            if math.isnan(score) : 
                score = None

            if train_ds_split in results_dict : 
                if test_ds_split not in results_dict[train_ds_split] : 
                    results_dict[train_ds_split][test_ds_split] = {} 
                results_dict[train_ds_split][test_ds_split][concept_name] = score
                    
            else : 
                results_dict[train_ds_split] = {}
                results_dict[train_ds_split][test_ds_split] = {} 
                results_dict[train_ds_split][test_ds_split][concept_name] = score 
                

with open(results_file, 'w') as file : 
    json.dump(results_dict, file)