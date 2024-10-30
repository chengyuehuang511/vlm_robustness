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
from transformers import BertModel, AutoTokenizer 




torch.cuda.empty_cache()

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#global variables 
GLOBAL_CONCEPT = "question"
INCLUDE_ANSWER = False 
COR_SHIFT = False 

concept_type = ["uni_question"]
hidden_layer_name = ["uni_question"]


ds_2_img = json.load(open("/coc/pskynet4/bmaneech3/vlm_robustness/ood_test/other/ds_2_img.json"))

ds_split_2_file = json.load(open("/coc/pskynet4/bmaneech3/vlm_robustness/ood_test/other/ds_split_2_file.json"))
ans_label_map = {} 
id_2_ans_map = {} 


def load_model(model_name="finetune") : 
    
    #image_tokens = 256, text_tokens = question length, SEP/EOS token = 2 


    if model_name == "finetune" : 
        model_id = "google/paligemma-3b-ft-vqav2-224"
        return PaliGemmaProcessor.from_pretrained(model_id), PaliGemmaForConditionalGeneration.from_pretrained(model_id, 
                                                            device_map ="auto",
                                                            torch_dtype=torch.bfloat16,
                                                            revision="bfloat16").eval()

    if model_name == "bert" : 
        model_id = "google-bert/bert-base-uncased" 
        return  AutoTokenizer.from_pretrained("google-bert/bert-base-uncased"), BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa",device_map="auto")



    #quantize model to reduce size 
    # return 

    #pretrained 
    # "google/paligemma-3b-pt-224"

    #DIGRAP 
    """
    ckpt_path = "/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/VQA/spcg_1e-3_mu0.5/20240918154/checkpoint_best.pth"

    # ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model_config = {'arch': 'paligemma_vqa', 'model_type': 'paligemma-3b-pt-224', 'load_pretrained': False, 'pretrained': 'https://huggingface.co/google/paligemma-3b-pt-224', 'load_finetuned': True, 'finetuned': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/VQA/adamh/l2_1e-3_wd0.5_overcap/20240803000/checkpoint_5.pth', 'use_lora': 1, 'target_modules': 'q_proj k_proj v_proj o_proj', 'lora_rank': 8, 'linear_probe': 0}
    #don't forget to put model to device 
    model_id = "google/paligemma-3b-ft-vqav2-224"
    model = PaliGemma_VQA(model_id,
    dtype=torch.bfloat16).from_config(model_config).eval()
    # print(ckpt['config']['model'])
    # model.load_from_pretrained(ckpt_path).eval()
    print("MODEL **", model.base_model.model.model)
    # question_model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)

    # model.base_model.model.model.device_map = "auto"
    model = model.to(device)
    """


processor, model = load_model("bert")


#dataloader + processor : 

IMAGE_SIZE = 224
BATCH_SIZE = 8
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
        if "answer" in sample : 
            label = sample["answer"]
            if isinstance(sample["answer"], list) : 
                label = Counter(sample['answer']).most_common(1)[0][0]

        else : 
            label = 0 

        question = sample["question"]
        if GLOBAL_CONCEPT == "joint" : 
            image_path = os.path.join(self.vis_root , sample["image"]) 
            input_image = Image.open(image_path).convert('RGB')
            resized_image = resize_transform(input_image)
            # inputs = processor(question, resized_image, padding=True)

            #convert label to id for conversion to tensors
            if label not in ans_label_map : 
                n = len(ans_label_map)
                ans_label_map[label] = n #{word: idx}
                id_2_ans_map[n] = label

            # print("label ", label)
            if INCLUDE_ANSWER : 
                question = f"{question} : {label}"
                # print("new question: ", question)

            label = ans_label_map[label]
            
            return question, resized_image, label #must return tensor to parallelize 
    
        return question 
    @staticmethod 
    def collate_fn(batch):
        if GLOBAL_CONCEPT == "joint" : 
            questions, images, labels = zip(*batch)
            inputs = processor(text=questions, images=images, padding=True)

            labels = torch.tensor(labels)
            #extract inputs if needed 
            return inputs, labels
        else : 
            
            questions = processor(batch, return_tensors='pt',padding=True, truncation=True) #didn't do processing with images? 

            return questions 
    
MAX_LENGTH = 2048 
def get_hidden_states(inputs, hidden_layer=19) :
    #prioritize order -> image, joint, question

    #maybe for pretrained models? 

    # if "question" in concept_type: 
    #     emb = question_model.encode(inputs, max_length=MAX_LENGTH)
    #     emb = F.normalize(emb,p=2, dim=1)
    #     concept_hidden_vectors = emb.unsqueeze(1)
    #     print("verify sentence emb size", emb.size()) 


    with torch.no_grad() : 
        output = model.forward(**inputs, return_dict=True, output_hidden_states=True)
    hidden_states = output.hidden_states #(layer, BATCH SIZE, seq length, hiddensize)

    concept_hidden_vectors = None 

    if any(True for concept in concept_type if "question" in concept) : 
        question_hidden_states = hidden_states[-1]
        attention_mask = inputs['attention_mask'].unsqueeze(-1) #(batch size, seq, hidden size)
        # print(type(hidden_states))
        # print(type(attention_mask))
        # print(hidden_states.size())
        # print(attention_mask.size())

        output = torch.sum(hidden_states * attention_mask, dim=1) #(batch size, seq, hidden size) -> (batch size, hidden size)

        total = torch.sum(attention_mask, dim=1) #(batch size, hidden size)

        mean_emb = output / total 


        print("mean ques emb", mean_emb.size()) #(batch size, dim )
        concept_hidden_vectors = mean_emb.unsqueeze(1) #(batch size, 1, dim)



    if "image" in concept_type : 
        image_hidden_states = hidden_states[0][:, :256, :]
        image_hidden_states = torch.mean(image_hidden_states, dim=1).unsqueeze(1)
        concept_hidden_vectors = image_hidden_states


    #new joint 
    if "joint" in concept_type or any(True for concept in concept_type if "joint" in concept): 
        output_hidden_states = hidden_states[-1]
        img_portion = output_hidden_states[:, 0:256, :]
        ques_portion = output_hidden_states[:, 256:, :]

        mean_img = torch.mean(img_portion, dim=1).unsqueeze(1) #(batch size, 1, dim)
        mean_ques = torch.mean(ques_portion, dim=1).unsqueeze(1) #(batch size, 1, dim)

        u_hidden_vector = torch.mean(torch.cat([mean_img, mean_ques], dim=1), dim=1).unsqueeze(1)
        print("dim joint emb", u_hidden_vector.size())
        if concept_hidden_vectors != None : 
            concept_hidden_vectors = torch.cat([concept_hidden_vectors, u_hidden_vectors], dim=1)
            
        else : 
            concept_hidden_vectors = u_hidden_vector

        del mean_img 
        del mean_ques 

    del hidden_states 
    torch.cuda.empty_cache()


    print(f"concept_hidden_vectors : {concept_hidden_vectors.size()}")
    return concept_hidden_vectors # (batch size, #concepts, hidden size)

#store covariance matrix 
cov_path_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/vqa_v2_{'_'.join(concept_type)}_inv_cov.pth"
def score_func(train_vectors, test_vectors, metric="maha", peranswerType=False) : 

    #do them in batches to avoid memory error 
    print("Train vector size", train_vectors.size())
    print("Test vector size", test_vectors.size())

    #train_vectors : (#concepts C , batch size N, hidden size H)
    #test_vectors : (#concepts, batch size, hidden size)

    train_vectors = train_vectors.to(device) 
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
        if not os.path.exists(cov_path_file) : 
            for i in range(n_batch) : 
                start_ind = i * COMP_BATCH_SIZE
                end_ind = min(n - 1, (i+1)*COMP_BATCH_SIZE)
                print(f"Running batch {start_ind} - {end_ind}")
                cur_vectors = train_vectors[:, start_ind : end_ind, :]
                cur_vectors = cur_vectors.to(device)

                mean_vector = u_vector.expand(c, cur_vectors.size(1), h)
                
                diff =  cur_vectors - mean_vector  #f - u_c : (C, N ,H)

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
            torch.save(inv_cov, cov_path_file)
            del cov
        else : 
            inv_cov = torch.load(cov_path_file)
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
            maha_score = -1 * torch.sqrt(diag) #negative mahalanobis

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
    
    
    del inv_cov 
    del test_vectors
    print(f"res shape : {res.size()}")

    # return avg score for each concept + (c, n, 1) 
    return total_res.tolist(), test_results_vectors #(concept, 1) 




answer_list_file = "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/answer_list.json"
splits =[
    #ds_name, split, file_name
    #(train, test, concept)
    #vqav2 train with all others
    #train_stuff = sample[0], test_stuff = sample[1]
    [("vqa_v2","train"), ("vqa_v2","train")],
    [("vqa_v2","val"), ("advqa", "test")],
    [("vqa_v2","train"), ("vqa_v2","val")], 
    [("vqa_v2","train"), ("cvvqa", "test")],
    [("vqa_v2","train"), ("ivvqa", "test")],
    [("vqa_v2","train"), ("okvqa", "test")], 
    [("vqa_v2","train"), ("textvqa", "test")], 
    [("vqa_v2","train"), ("vizwiz", "test")], 
    [("vqa_v2","train"), ("vqa_cp", "test")], 
    [("vqa_v2","train"), ("vqa_ce", "test")], 
    [("vqa_v2","train"), ("vqa_rephrasings", "test")]
]

results_file = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/maha_score_dict.json"
if os.path.exists(results_file) : 
    with open(results_file, 'r') as file : 
        results_dict = json.load(file) #read from results dict 
else : 
    results_dict = {} 


                
#store result vector for each tensor size (n_samples, 1) in order index
if __name__ == "__main__" : 

    for measure_instance in splits : 
        print(torch.cuda.memory_allocated())
        print("Memory summary", torch.cuda.memory_summary(device=None, abbreviated=False))
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

        with open(train_file, 'r') as f :
            train_data = json.load(f)

        with open(test_file, 'r') as f : 
            test_data = json.load(f)

        train_hidden_state_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{train_ds_split}_{'_'.join(concept_type)}.pth"
        if not os.path.exists(train_hidden_state_file) : 

            dataset = MeasureOODDataset(train_data, train_ds_name)

            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn, num_workers=2)


            #VQAV2 -> store results of torch
            train_vectors = []
            co = 1 
            for batch in dataloader : 
                print(f"Current train batch {co}")
                co += 1

                # inputs, labels = batch 
                inputs = batch
                inputs = inputs.to(device) #don't forget to add puts to gpu 
            
                enc_vectors = get_hidden_states(inputs) #(batchsize, concept, hidden size)
                enc_vectors_cpu = enc_vectors.detach().cpu() #(batchsize, concept, hidden size)
              
                #free up memory so next batch can use GPU compute 

                # del inputs, enc_vectors 
                del inputs
                del enc_vector
                train_vectors.append(enc_vectors_cpu) #(batch size, concepts, hidden size, ans dim if )

            #on cpu now 
            train_vectors = torch.cat(train_vectors, dim=0).to(device) #(n, concepts, hidden size)
            train_vectors = train_vectors.permute(1,0,2) #(concepts, n, hidden size)

            print(f"final train vectors size : {train_vectors.size()}")
            
            torch.save(train_vectors.detach().cpu(), train_hidden_state_file)
            del train_vectors
            torch.cuda.empty_cache()

        test_hidden_state_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{test_ds_split}_{'_'.join(concept_type)}.pth"

        if not os.path.exists(test_hidden_state_file) : 
  
            dataset = MeasureOODDataset(test_data, test_ds_name)
            # concept_type = "joint"

            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=dataset.collate_fn, num_workers=2)

            #VQAV2 -> store results of torch
            test_vectors = []
            co = 1 
            for batch in dataloader : 

                co += 1 
                # inputs, labels = batch 
                inputs = batch 
                inputs = inputs.to(device)

                enc_vectors = get_hidden_states(inputs)
                enc_vectors_cpu = enc_vectors.detach().cpu() #(batchsize, concept, hidden size)

        
                test_vectors.append(enc_vectors_cpu)
                del inputs
                del enc_vectors 
 
            test_vectors = torch.cat(test_vectors, dim=0).to(device)
            test_vectors = test_vectors.permute(1, 0, 2)

            print(f"final test vectors size : {test_vectors.size()}")
            
            torch.save(test_vectors.detach().cpu(), test_hidden_state_file)
            del test_vectors 
            torch.cuda.empty_cache()
        
        #on cpu 
        print("test file", test_hidden_state_file)
        train_vectors = torch.load(train_hidden_state_file) #(CONCEPT, batch size, hidden)
        test_vectors = torch.load(test_hidden_state_file)


        n_train = train_vectors.size(1)
        n_test = test_vectors.size(1)
        
        batch_inc = n_train // COMP_BATCH_SIZE

        # scores, test_results_vectors = score_func(train_vectors, test_vectors) #list of score per concept 
        # print(f"Score {scores}")

        # #store test results vectors  
        # indiv_result_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/indiv_result/{test_ds_split}_{'_'.join(concept_type)}.pth"

        # # if not os.path.exists(indiv_result_file) : 
        # torch.save(test_results_vectors.detach().cpu(), indiv_result_file)

        #don't actually because after next iteration - garbage collector takes care 
        del train_vectors 
        del test_vectors 
        # del test_results_vectors
    
        torch.cuda.empty_cache()
        gc.collect()

        #train_vectors, test_vectors : (batch size, concepts, hidden size)
        
#         for concept_name, score in zip(hidden_layer_name, scores) : 
#             if math.isnan(score) : 
#                 score = None

#             if train_ds_split in results_dict : 
#                 if test_ds_split not in results_dict[train_ds_split] : 
#                     results_dict[train_ds_split][test_ds_split] = {} 
#                 results_dict[train_ds_split][test_ds_split][concept_name] = score
                    
#             else : 
#                 results_dict[train_ds_split] = {}
#                 results_dict[train_ds_split][test_ds_split] = {} 
#                 results_dict[train_ds_split][test_ds_split][concept_name] = score 
                

# with open(results_file, 'w') as file : 
#     json.dump(results_dict, file)