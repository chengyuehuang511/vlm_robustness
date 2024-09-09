from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig

from PIL import Image
import os 
import torch 
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
from torchvision import transforms
from torchvision.transforms import InterpolationMode
# from data.builders import load_dataset, DatasetZoo

# dataset_zoo = DatasetZoo()
# # coco_vqa_vs
# vqa_vs = load_dataset("coco_vqa_vs")
torch.cuda.empty_cache()

#globally set no gradient tracking
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
#store ans_label_map in json 


#don't forget to put model to device 
model_id = "google/paligemma-3b-ft-vqav2-224"

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#quantize model to reduce size 
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, 
                                                        device_map ="auto",
                                                        torch_dtype=torch.bfloat16,
                                                        revision="bfloat16").eval()

processor = PaliGemmaProcessor.from_pretrained(model_id)

# output = model.generate(**inputs, max_new_tokens=20)
# # print(output)
# print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])

#image_tokens = 256, text_tokens = question length, SEP/EOS token = 2 

#dataloader + processor : 

IMAGE_SIZE = 224
BATCH_SIZE = 4
HIDDEN_SIZE = 2048 
COMP_BATCH_SIZE = 1000 

resize_transform = transforms.Resize(
    (IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC
)

class MeasureOODDataset(Dataset) : 
    
    def __init__(self, data, ds_name):
        self.data = data #list of dictionary elements 
        self.ds_name = ds_name 
        self.vis_root = ds_2_img[ds_name]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = sample["answer"]
        if isinstance(sample["answer"], list) : 
            label = Counter(sample['answer']).most_common(1)[0][0]

        question = sample["question"]

        image_path = os.path.join(self.vis_root , sample["image"]) 
        input_image = Image.open(image_path).convert('RGB')
        resized_image = resize_transform(input_image)
        # inputs = processor(question, resized_image, padding=True)

        #convert label to id for conversion to tensors
        if label not in ans_label_map : 
            n = len(ans_label_map)
            ans_label_map[label] = n

        label = ans_label_map[label]
        
        return question, resized_image, label #must return tensor to parallelize 
    
    @staticmethod 
    def collate_fn(batch):
        questions, images, labels = zip(*batch)

        inputs = processor(text=questions, images=images, padding=True)
        # print(type(labels), labels.size())
        # print(labels)
        labels = torch.tensor(labels)

        #extract inputs if needed 
        return inputs, labels
    
def get_hidden_states(inputs, concept_type="joint", hidden_layer=19) :
    
    with torch.no_grad() : 
        output = model.forward(**inputs, return_dict=True, output_hidden_states=True)
    hidden_states = output.hidden_states #(layer, BATCH SIZE, seq length, hiddensize)
    print(len(hidden_states))

    # output_hidden_state = hidden_states[-1] #average all vectors together  
    # image_hidden_state = hidden_states[0][:, 0:256, :] #first image encoder after projection 

    """
    output_hidden_state : (batch size, seq length, hidden dim)
    image_hidden_state : (batch size, seq length, hidden dim)

    return 
        concept_hidden_states : (batch size, # of concept, hidden dim)
    """



    # image_hidden_state = hidden_states[0][:, 0:256, :] #(batch size, seq length, hidden dim) 
    # u_image_vector = torch.mean(image_hidden_state, dim=1).unsqueeze(1)
    # concept_hidden_vectors = u_image_vector 
    
    # # for i in range(len(hidden_states)) : 
    # u_hidden_vector =  torch.mean(hidden_states[-1], dim=1) 
    # cur_hidden_vectors = u_hidden_vector.unsqueeze(1) # dim 1 = concept
    # concept_hidden_vectors = torch.cat([concept_hidden_vectors, cur_hidden_vectors], dim = 1)

    question_hidden_state = hidden_states[9][:, 256:, :] #(batch size, seq length, hidden dim) #what layer should i do? 1?
    concept_hidden_vectors = torch.mean(question_hidden_state, dim=1).unsqueeze(1)

    last_question_hidden_state = torch.mean(hidden_states[-1][:,256:,:], dim=1) .unsqueeze(1)
    concept_hidden_vectors = torch.cat([concept_hidden_vectors, last_question_hidden_state], dim=1)

   # del hidden_states 
    # del image_hidden_state 
    # del u_image_vector
    # del u_hidden_vector
    # del cur_hidden_vectors 
    del hidden_states
    del question_hidden_state
    del last_question_hidden_state
    torch.cuda.empty_cache()
    

    print(f"concept_hidden_vectors : {concept_hidden_vectors.size()}")
    return concept_hidden_vectors #(batch size, #concepts, hidden size)


def score_func(train_vectors, test_vectors, metric="maha", peranswerType=False) : 
    
    #do them in batches to avoid memory error 

    #perAnswerType Logic
    #train_vectors : (#concepts C , batch size N, hidden size H)
    #test_vectors : (#concepts, batch size, hidden size)
    train_vectors = train_vectors.to(device)
    u_vector = torch.mean(train_vectors, dim=1) #find mean in vector (concept, batch size , hidden size)
    
    c, n, h = train_vectors.size() 
    print(f"size (num_concept, samples, hidden size) : {c, n , h}")

    #store u_vectors for certain things? 

    u_vector = u_vector.expand(c, n, h)
    

    diff =  train_vectors - u_vector  #f - u_c : (C,N ,H)

    diff = diff.permute(0,2,1)  #(C, H, N)
    print("diff size", diff.size())

    del train_vectors

    #covariance matrix : * = element wise multiplication 
    cov = torch.matmul(diff, diff.transpose(0,2,1))  #(C, H, N) x (C, N, H) -> (C, H, H)
    cov = cov / n

    inv_cov = torch.empty(c,h,h) #(C, H, H)
    for i in range(c) : 
        inv_cov[i] = torch.pinverse(cov[i]) # Σ^-1

    torch.cuda.empty_cache()

    #maha metric 
    if metric == "maha" : 
        test_vectors = test_vectors.gpu() 

        #Z_test - μ 
        test_diff = (test_vectors - u_vector) #(concept, BATCH SIZE, hidden size)
        res_1 = torch.matmul(test_diff, inv_cov) #(concept, BATCH SIZE, hidden size)
        res_2 = torch.matmul(res_1, test_diff.permute(0, 2, 1)) #(concept, BATCH SIZE, BATCH SIZE)

        res = [] 
        for i in range(c) : 
            diag = torch.diag(res_2[i]) #(1, d)
            #verify correctness via shape 
            assert diag.size(1) == test_vectors.size(1) , "mismatch shape in diagonal values and n samples" 

            #avg dist across all samples 
            score = torch.sum(diag, dim=1) / test_vectors.size(1)
            score = float(score.item()) 
            res.append(score)

        del res_1
        del res_2 
        del test_diff 

    #next perAnswerType 

    del cov
    del inv_cov 
    del test_vectors 

    
    

    # print("scores shape", res.size()) #(c, )
    print(f"res shape : {len(res)}")
    return res #(concept, 1)


#organize structure : # for each split : {ds_name}_{split} -> store hidden states per ds_split , concept 
#store hidden states : ds_name, split (ds_split), concpept -> store as tensor 
#.pth filename : /nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{ds_name}_{split}_{concept}.pth

#measure ood score : 
"""
train ds name, split 
test ds name, split

concept : joint/image/question etc. 

"""

#how to organize when you get the score 
concept_type = ["question"] #


splits =[
    #ds_name, split, file_name
    #(train, test, concept)
    #vqav2 train with all others
    #train_stuff = sample[0], test_stuff = sample[1]
    [("vqa_v2", "train"), ("vqa_v2","val")],
    [("vqa_rephrasings", "test"), ("vqa_vs", "id_val")], 
    [("vqa_ce","test"), ("vqa_v2","train")],
    [("vqa_v2","test"), ("vqa_vs", "id_test")]

    # [("vqa_vs", "KO"), ("vqa_vs", "KOP")],
    # [("vqa_vs", "KW"), ("vqa_vs", "KW_KO")],
    # [("vqa_vs", "KWP"),("vqa_vs", "QT")],
    # [("vqa_vs", "QT_KO"), ("vqa_vs", "QT_KW")],
    # [("vqa_vs", "QT_KW_KO"), ("vqa_v2","train")]
]


results_file = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/maha_score_dict.json"
if os.path.exists(results_file) : 
    with open(results_file, 'r') as file : 
        results_dict = json.load(file) #read from results dict 
else : 
    results_dict = {} 

hidden_layer_name = ["q_mid", "q_last"]

# for i in range(1,20) : 
#     hidden_layer_name.append(f"joint_l{i}")

if __name__ == "__main__" : 
        
    for measure_instance in splits : 
        
        train_ds_name, train_split = measure_instance[0]
        test_ds_name, test_split = measure_instance[1]

        train_ds_split = f"{train_ds_name}_{train_split}" 
        test_ds_split = f"{test_ds_name}_{test_split}"
        
        train_file = ds_split_2_file[train_ds_split]
        test_file = ds_split_2_file[test_ds_split]
        print(f"Measure Instance {train_ds_split, test_ds_split}")

        if (f"{train_split}" in results_dict): 
            if (f"{test_split}" in results_dict[f"{train_split}"]) : 
                if (concept_type[0] in results_dict[f"{train_split}"][f"{test_split}"]) : 
                    print(f"already measured")
                    continue  
                
        with open(train_file, 'r') as f :
            train_data = json.load(f)


        with open(test_file, 'r') as f : 
            test_data = json.load(f)


        train_hidden_state_file = f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/hidden_states/{train_ds_split}_{'_'.join(concept_type)}.pth"
        if not os.path.exists(train_hidden_state_file) : 

            dataset = MeasureOODDataset(train_data, train_ds_name)
            # concept_type = "joint"

            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn, num_workers=2)

            #VQAV2 -> store results of torch
            train_vectors = []
            co = 1 
            for batch in dataloader : 
                print(f"Current train batch {co}")
                co += 1

                inputs, labels = batch 
                inputs = inputs.to(device) #don't forget to add puts to gpu 
            
                enc_vectors = get_hidden_states(inputs, concept_type)
                enc_vectors_cpu = enc_vectors.cpu() #(batchsize, concept, hidden size)

                #free up memory so next batch can use GPU compute 

                # del inputs, enc_vectors 
                del inputs
                del enc_vectors

                train_vectors.append(enc_vectors_cpu) #(batch size, concepts, hidden size)

            #on cpu now 
            train_vectors = torch.cat(train_vectors, dim=0).to(device) #(n, concepts, hidden size)
            train_vectors = train_vectors.permute(1,0,2) #(concepts, n, hidden size)

            print(f"final train vectors size : {train_vectors.size()}")
            
            torch.save(train_vectors.cpu(), train_hidden_state_file)
            del train_vectors
            torch.cuda.empty_cache()

        test_hidden_state_file = f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/hidden_states/{test_ds_split}_{'_'.join(concept_type)}.pth"
        if not os.path.exists(test_hidden_state_file) : 

            dataset = MeasureOODDataset(test_data, test_ds_name)
            # concept_type = "joint"

            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn, num_workers=2)

            #VQAV2 -> store results of torch
            test_vectors = []
            co = 1 
            for batch in dataloader : 
                print(f"Current test batch {co}")

                co += 1 
                inputs, labels = batch 
                inputs = inputs.to(device)

                enc_vectors = get_hidden_states(inputs, concept_type)
                enc_vectors_cpu = enc_vectors.cpu() #(batchsize, concept, hidden size)

                test_vectors.append(enc_vectors_cpu)
                del inputs
                del enc_vectors 
            test_vectors = torch.cat(test_vectors, dim=0).to(device)
            test_vectors = test_vectors.permute(1, 0, 2)

            print(f"final test vectors size : {test_vectors.size()}")
            
            torch.save(test_vectors.cpu(), test_hidden_state_file)
            del test_vectors 
            torch.cuda.empty_cache()
        
        # #on cpu 
        # train_vectors = torch.load(train_hidden_state_file) #(CONCEPT, batch size, hidden)
        # test_vectors = torch.load(test_hidden_state_file)


        # n_train = train_vectors.size(1)
        # n_test = test_vectors.size(1)
        
        # batch_inc = n_train // COMP_BATCH_SIZE

        # #get mean and s.d. 
        # # for i in range(batch_inc) : 
        # #     train_batch = train_vectors[COMP_BATCH_SIZE * i : min(n_train - 1, (i+1) * COMP_BATCH_SIZE)] #(concept, batch_size, hidden size)
        # #     train_batch = train_batch.to(device)
        # #     torch.sum(train_batch, dim = 1)

        # scores = score_func(train_vectors, test_vectors) #list of score per concept 
        # print(f"Score {scores}")
        # del train_vectors 
        # del test_vectors 
        
        # torch.cuda.empty_cache()

        # #train_vectors, test_vectors : (batch size, concepts, hidden size)
        
        # for concept_name, score in zip(hidden_layer_name, scores) : 

        #     if train_split in results_dict : 
        #         if test_split not in results_dict[train_split] : 
        #             results_dict[train_split][test_split] = {} 
        #         results_dict[train_split][test_split][concept_name] = score
                    
        #     else : 
        #         results_dict[train_split] = {}
        #         results_dict[train_split][test_split] = {} 
        #         results_dict[train_split][test_split][concept_name] = score 


                                        

# with open(results_file, 'w') as file : 
#     json.dump(results_dict, file)




