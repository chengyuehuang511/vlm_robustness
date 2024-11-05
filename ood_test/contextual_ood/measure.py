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
import argparse
import logging 

torch.cuda.empty_cache()

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


IMAGE_SIZE = 224
BATCH_SIZE = 4
HIDDEN_SIZE = 2048 
COMP_BATCH_SIZE = 1000 


def create_cov_matrix(ft_method, train_dict, concept_type=["image","joint"], metric="maha") : 
    logging.info("getting cov matrix")

    """train_dict : 
    instance_id -> tensor(c, dim)
    """

    
    train_vectors = torch.cat([value.unsqueeze(1) for key, value in sorted(train_dict.items())], dim=1).to(device) #(c, n, dim)
    logging.info("finish sorting")
    logging.info(f"verify train_vector size {train_vectors.size()}") #(c, n, dim)
    
    try : 
        assert train_vectors.size(1) == len(train_dict), "unpack train vectors != len train vectors"
    except AssertionError as e : 
        logging.error(e)


    
    u_vector = torch.mean(train_vectors, dim=1, keepdim=True) #find mean in vector (concept, batch size , hidden size) -> (C, 1, H)
    print("Size of mean vector", u_vector.size())
    # print(torch.cuda.memory_allocated())

    if torch.isinf(u_vector).any() or torch.isnan(u_vector).any() : 
        raise Exception("u vector has NaN values")

    train_vectors = train_vectors.detach().cpu()   
    c, n, h = train_vectors.size() 
    print(f"size (num_concept, samples, hidden size) : {c, n, h}")
    
    #store u_vectors for certain things? 
    n_batch = math.ceil(n / COMP_BATCH_SIZE)
    final_cov = torch.zeros(c,h,h) #(C,H,H)

    print("========================")
    print(f"TRAINING SAMPLE {n} samples")
 
    cov_path_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/cov_matrices/{ft_method}_{'_'.join(concept_type)}_inv_cov.pth"


    if not os.path.exists(cov_path_file) : 
        co = 0
        for i in range(n_batch) : 
            start_ind = i * COMP_BATCH_SIZE
            end_ind = min(n , (i+1)*COMP_BATCH_SIZE)
            print(f"Running batch {start_ind} - {end_ind}")
            cur_vectors = train_vectors[:, start_ind : end_ind, :] #(concept, batch size, dim)
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
            co += end_ind - start_ind 

        assert co == n, "total process samples in inv cov != total samples"
        final_cov = final_cov / (n - 1) #n-1 for bias correction

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
        print("saved inv_cov matrix to : ", cov_path_file)
        del cov

    del train_vectors 
    return u_vector


#new score func 
def score_func(u_vector, test_dict, ft_method, concept_type=["image", "joint"], metric="maha") : 
    #train_dict.values() = list of (c, hidden) vectors  
    #sort the keys into order 
    #verify test_vectors order 
    for idx, (key, value) in enumerate(sorted(test_dict.items())):
        if idx != key : 
            raise Exception("Instance id doesn't correspond to order")

    test_vectors = torch.cat([value.unsqueeze(1) for key, value in sorted(test_dict.items())], dim=1)

    
 
    c, n_test, h = test_vectors.size()
    total_res = torch.zeros(c)
    print("========================")
    print(f"TESTING SAMPLE {n_test} samples")
    n_batch = math.ceil(n_test / COMP_BATCH_SIZE)

    cov_path_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/cov_matrices/{ft_method}_{'_'.join(concept_type)}_inv_cov.pth"
    inv_cov = torch.load(cov_path_file)

    inv_cov = inv_cov.to(dtype=test_vectors.dtype)
    inv_cov = inv_cov.to(device)
    
    test_results_vectors = [] #(c, n, 1)
    #test batches 
    for j in range(n_batch): 
        start_ind = j * COMP_BATCH_SIZE
        end_ind = min(n_test , (j+1) * COMP_BATCH_SIZE)
        print(f"Running batch {start_ind} - {end_ind}")
        cur_vectors = test_vectors[:, start_ind : end_ind, :] #(C, comp_batch_size, h)
        cur_vectors = cur_vectors.to(device)
        mean_vector = u_vector.expand(c, cur_vectors.size(1), h)

        #Z_test - μ 
        test_diff = (cur_vectors - mean_vector)#(concept, BATCH SIZE, hidden size) ; x - u
        print("verify test diff size", test_diff.size())
        res_1 = torch.matmul(test_diff, inv_cov) #(concept, BATCH SIZE, hidden size)
        res_2 = torch.matmul(res_1, test_diff.permute(0, 2, 1)) #(concept, BATCH SIZE, BATCH SIZE)

        res = torch.zeros(c) 
        cur_batch_results = None
        for i in range(c) : 
            diag = torch.diag(res_2[i]) #(1, n) 
            #verify correctness via shape 
            print("diag size",diag.size())

            if (diag < 0).any() : 
                raise Exception("Diagonal values can't be negative")
            
            assert diag.size(0) == cur_vectors.size(1) , "mismatch shape in diagonal values and n samples" 

            #avg dist across all samples 
            maha_score = -1 * torch.sqrt(diag) #negative mahalanobis

            if cur_batch_results == None : 
                cur_batch_results = maha_score.unsqueeze(0) #(1, b)
                print("verify maha score shape", maha_score.size())

            else : 
                cur_batch_results = torch.cat([cur_batch_results, maha_score.unsqueeze(0)], dim = 0) #will be (c, n)
            
            total_score = torch.sum(maha_score) #sum maha score across all batch samples 
            total_score = total_score.item() #would this be on the CPU now 
            res[i] = total_score #(1, c)

        test_results_vectors.append(cur_batch_results) #(c, n)

        total_res = total_res + res  #(c)
        print("Done a sample")
        del res_1 
        del res_2
        torch.cuda.empty_cache()

    test_results_vectors = torch.cat(test_results_vectors, dim=1).permute(1,0) #(n,c)
    total_res = total_res / n_test #average maha score across all test samples

    try : 
        assert test_results_vectors.size(0) == n_test , "indiv results list not equal to total samples"
    except AssertionError as e : 
        logging.error(e)
    
    del inv_cov 
    del test_vectors
    print(f"res shape : {res.size()}")

    # return avg score for each concept + (c, n, 1) 
    return total_res.tolist(), test_results_vectors #(concept, 1) 

#structure : FT -> train & test split : scores 


# ft_methods = ["lora", "digrap", "fft","ftp","lp","lpft","spd"]

hidden_states_path = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states"
train_split = "coco_vqav2_train_val"
test_splits = [
    "coco_vqav2_train_val",
    "coco_advqa_val", 
    "coco_cv-vqa_val",
    "coco_iv-vqa_val",
    "coco_okvqa_val",
    "coco_vqa_ce_val",
    "coco_vqa_cp_val",
    "coco_vqa_raw_val",
    "coco_vqa_rephrasings_val",
    "textvqa_val",
    "vizwiz_val"
]

"""
concept_types : 
- image_ft
- ques_ft 
- joint 
- image (vit)
- question (bert)
"""

# concept_list = ["image_ft", "joint", "ques_ft", "image", "question"]
concept_list = [["image", "joint"], ["ques_ft"]]
if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description="Process a parameter.")
    parser.add_argument("--ft_method", type=str, required=True, help="The parameter to be processed")
    args = parser.parse_args()
    ft_method = args.ft_method

    log_output_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/logs/"
    os.makedirs(os.path.join(log_output_dir, ft_method), exist_ok=True)
    log_file = os.path.join(log_output_dir, "measure_shift.log")
    
    logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()         # Log to console
    ]
    )

    results_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/results_ood/{ft_method}_ood_score_dict.json" #ft -> train -> test dataset 
    if os.path.exists(results_file) : 
        with open(results_file, 'r') as file : 
            results_dict = json.load(file) #read from results dict 
    else : 
        results_dict = {} 


    """for each ft 
        - results_dict (dict) : test_split -> concept -> score (int)
        for each split
            - indiv_results (dict) : concept -> instance_id -> score (int)
    """

    for concept_type in concept_list : 
        #get train covariate matrix 
        train_file_name = f"{train_split}_{'_'.join(concept_type)}_new.pth"
        train_dict_path = os.path.join(hidden_states_path, ft_method, train_file_name) 
        train_dict = torch.load(train_dict_path) 
        print("loading train dict")
        train_dict = {int(key): value for key, value in train_dict.items()}
        print("converted to int")
        
        u_vector = create_cov_matrix(ft_method, train_dict, concept_type=concept_type)
        
        for test_split in test_splits : 
            
            test_file_name = f"{test_split}_{'_'.join(concept_type)}_new.pth"
            test_dict_path = os.path.join(hidden_states_path, ft_method, test_file_name) 
            test_dict = torch.load(test_dict_path) 
            test_dict = {int(key): value for key, value in test_dict.items()}

            shift_scores, test_indiv_results = score_func(u_vector, test_dict, ft_method, concept_type=concept_type)
            logging.info(f"Test split : {test_split}" )
            logging.info(concept_type, shift_scores)

            assert test_indiv_results.size(0) == len(test_dict) and test_indiv_results.size(1) == len(concept_type), "invalid test_indiv_results"

            results_dict[test_split] = {} 
            #store indiv results file 
            logging.info(f"test indiv results size : {test_indiv_results.size}") #(n, c)
            indiv_res_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/{ft_method}/{test_split}_indiv_result.pth"

            if os.path.exists(indiv_res_path) : 
                indiv_results = torch.load(indiv_res_path)
            else : 
                indiv_results = {}

            for concept_idx, concept in enumerate(concept_type) : 
                results_dict[test_split][concept] = shift_scores[concept_idx]

                cur_test_results = test_indiv_results[:, concept_idx] #(n , 1)
                indiv_results_dict = {sample_idx: cur_test_results[sample_idx] for sample_idx in range(len(cur_test_results))}
                indiv_results[concept] = indiv_results_dict

                try : 
                    assert len(indiv_results_dict) == len(test_dict), "length of indiv_results_dict != test_dict"
                except AssertionError as e:
                    logging.error(f"Assertion failed: {e}")

            torch.save(indiv_results, indiv_res_path)

    with open(results_file, 'w') as file : 
        json.dump(results_dict, file)

# #store result vector for each tensor size (n_samples, 1) in order index
# if __name__ == "__main__" : 

#     for measure_instance in splits : 
#         print(torch.cuda.memory_allocated())
#         print("Memory summary", torch.cuda.memory_summary(device=None, abbreviated=False))
#         # results_file = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/maha_score_dict.json"
#         # if os.path.exists(results_file) : 
#         #     with open(results_file, 'r') as file : 
#         #         results_dict = json.load(file) #read from results dict 
#         # else : 
#         #     results_dict = {} 
        
#         train_ds_name, train_split = measure_instance[0]
#         test_ds_name, test_split = measure_instance[1]

#         train_ds_split = f"{train_ds_name}_{train_split}" 
#         test_ds_split = f"{test_ds_name}_{test_split}"
        
#         train_file = ds_split_2_file[train_ds_split]
#         test_file = ds_split_2_file[test_ds_split]
#         print(f"Measure Instance {train_ds_split, test_ds_split}")

#         with open(train_file, 'r') as f :
#             train_data = json.load(f)

#         with open(test_file, 'r') as f : 
#             test_data = json.load(f)

#         train_hidden_state_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{train_ds_split}_new_joint.pth"
#         if not os.path.exists(train_hidden_state_file) : 

#             dataset = MeasureOODDataset(train_data, train_ds_name)

#             dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_fn, num_workers=2)


#             #VQAV2 -> store results of torch
#             train_vectors = []
#             co = 1 
#             for batch in dataloader : 
#                 print(f"Current train batch {co}")
#                 co += 1

#                 inputs, labels = batch 
#                 inputs = inputs.to(device) #don't forget to add puts to gpu 
            
#                 enc_vectors = get_hidden_states(inputs) #(batchsize, concept, hidden size)
#                 enc_vectors_cpu = enc_vectors.detach().cpu() #(batchsize, concept, hidden size)
              
#                 #free up memory so next batch can use GPU compute 

#                 # del inputs, enc_vectors 
#                 del inputs
#                 del enc_vectors
#                 train_vectors.append(enc_vectors_cpu) #(batch size, concepts, hidden size, ans dim if )

#             #on cpu now 
#             train_vectors = torch.cat(train_vectors, dim=0).to(device) #(n, concepts, hidden size)
#             train_vectors = train_vectors.permute(1,0,2) #(concepts, n, hidden size)

#             print(f"final train vectors size : {train_vectors.size()}")
            
#             torch.save(train_vectors.detach().cpu(), train_hidden_state_file)
#             del train_vectors
#             torch.cuda.empty_cache()

#         test_hidden_state_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/{test_ds_split}_new_joint.pth"

#         if not os.path.exists(test_hidden_state_file) : 
  
#             dataset = MeasureOODDataset(test_data, test_ds_name)
#             # concept_type = "joint"

#             dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=dataset.collate_fn, num_workers=2)

#             #VQAV2 -> store results of torch
#             test_vectors = []
#             co = 1 
#             for batch in dataloader : 

#                 co += 1 
#                 inputs, labels = batch 
#                 inputs = inputs.to(device)

#                 enc_vectors = get_hidden_states(inputs)
#                 enc_vectors_cpu = enc_vectors.detach().cpu() #(batchsize, concept, hidden size)

        
#                 test_vectors.append(enc_vectors_cpu)
#                 del inputs
#                 del enc_vectors 
 
#             test_vectors = torch.cat(test_vectors, dim=0).to(device)
#             test_vectors = test_vectors.permute(1, 0, 2)

#             print(f"final test vectors size : {test_vectors.size()}")
            
#             torch.save(test_vectors.detach().cpu(), test_hidden_state_file)
#             del test_vectors 
#             torch.cuda.empty_cache()
        
#         #on cpu 
#         print("test file", test_hidden_state_file)
#         train_vectors = torch.load(train_hidden_state_file) #(CONCEPT, batch size, hidden)
#         test_vectors = torch.load(test_hidden_state_file)


#         n_train = train_vectors.size(1)
#         n_test = test_vectors.size(1)
        
#         batch_inc = n_train // COMP_BATCH_SIZE

#         # scores, test_results_vectors = score_func(train_vectors, test_vectors) #list of score per concept 
#         # print(f"Score {scores}")

#         # #store test results vectors  
#         # indiv_result_file = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/indiv_result/{test_ds_split}_{'_'.join(concept_type)}.pth"

#         # # if not os.path.exists(indiv_result_file) : 
#         # torch.save(test_results_vectors.detach().cpu(), indiv_result_file)

#         #don't actually because after next iteration - garbage collector takes care 
#         del train_vectors 
#         del test_vectors 
#         # del test_results_vectors
    
#         torch.cuda.empty_cache()
#         gc.collect()

#         #train_vectors, test_vectors : (batch size, concepts, hidden size)
        
# #         for concept_name, score in zip(hidden_layer_name, scores) : 
# #             if math.isnan(score) : 
# #                 score = None

# #             if train_ds_split in results_dict : 
# #                 if test_ds_split not in results_dict[train_ds_split] : 
# #                     results_dict[train_ds_split][test_ds_split] = {} 
# #                 results_dict[train_ds_split][test_ds_split][concept_name] = score
                    
# #             else : 
# #                 results_dict[train_ds_split] = {}
# #                 results_dict[train_ds_split][test_ds_split] = {} 
# #                 results_dict[train_ds_split][test_ds_split][concept_name] = score 
                

# # with open(results_file, 'w') as file : 
# #     json.dump(results_dict, file)