import torch 



# random_tensor = torch.randint(low=1,high=5, size= (2, 2, 3))
# print(random_tensor)
# print(torch.matmul(random_tensor, random_tensor.permute(0, 2,1 )))
# print(torch.matmul(random_tensor, random_tensor.permute(0, 2,1 )).size())


tensor1 = torch.randn(32, 10, 256).to(dtype= torch.bfloat16)
tensor2 = torch.randn(32, 10, 256).to(dtype= torch.bfloat16)

cov = torch.matmul(tensor1, tensor2.permute(0,2,1)).to(dtype=torch.bfloat16)  #(C, H, N) x (C, N, H) -> (C, H, H)

# tensor3 = torch.randn(32, 10, 256)

# arr = [tensor1, tensor2, tensor3]
# # print(arr.size())
# print(torch.cat(arr, dim=0))

# print(torch.cat(arr, dim=0).size())

# t = [1,3,4]
# print(torch.tensor(t).size())




# arr = ["image"]
# for i in range(1,20) : 
#     arr.append(f"joint_l{i}")

# print(arr)
# print(len(arr))



# arr = [[("vqa_v2","train"), ("vqa_vs", "id_val")], 
#      [("vqa_v2","train"), ("vqa_vs", "id_val")]
# ]
# for a in arr :
#     traina, trainb = a[0]
#     testa, testb = a[1]



# print('_'.join(["hello", "world"]))



f_train = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hidden_states/vqa_v2_train_joint_image.pth"
f = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/hidden_states/vqa_v2_val_joint_image.pth"
train_vectors = torch.load(f_train)
test_vectors = torch.load(f)

train_max = None
train_min = None 

test_max = None
test_min = None

# for i in range(2048) : 
#     if train_max != None : 
#         train_max =  min(train_max, torch.min(train_vectors[0][:][i]).item())
#     else : 
#         train_max = torch.max(train_vectors[0][:][i]).item() 

#     if test_max != None : 
#         test_max = min(test_max, torch.min(test_vectors[0][:][i]).item())
#     else : 
#         test_max = torch.min(test_vectors[0][:][i])


# print(train_max )
# print(test_max)
# # print(f"train vectors min/max : {torch.min(train_vectors[0][0])}-{torch.max(train_vectors[0][0])} ")
# # print(f"test vectors min/max : {torch.min(test_vectors[0][0])}-{torch.max(test_vectors[0][0])} ")


if torch.isinf(train_vectors).any() or torch.isnan(train_vectors).any() : 
    raise Exception("has NaN values")

if torch.isinf(test_vectors).any() or torch.isnan(test_vectors).any() : 
    raise Exception("has NaN values")



print(torch.sum(train_vectors[0] == 0).item())
print(torch.sum(test_vectors[0] == 0).item())


def score_func(train_vectors, test_vectors, metric="maha", peranswerType=False) : 
    
    #do them in batches to avoid memory error 

    print("Train vector size", train_vectors.size())
    print("Test vector size", test_vectors.size())

    #perAnswerType Logic
    #train_vectors : (#concepts C , batch size N, hidden size H)
    #test_vectors : (#concepts, batch size, hidden size)
   
    # train_vectors = train_vectors
    u_vector = torch.mean(train_vectors, dim=1, keepdim=True).to(dtype=torch.bfloat16) #find mean in vector (concept, batch size , hidden size) -> (C, 1, H)
    print("Size of mean vector", u_vector.size())
   

    if torch.isinf(u_vector).any() or torch.isnan(u_vector).any() : 
        raise Exception("u vector has NaN values")

    train_vectors = train_vectors.detach().cpu()   
    c, n, h = train_vectors.size() 
    print(f"size (num_concept, samples, hidden size) : {c, n , h}")
 
    #store u_vectors for certain things? 

    final_cov = torch.zeros(c,h,h) #(C,H,H)

    print("========================")
    print(f"TRAINING SAMPLE {n} samples")

    mean_vector = u_vector.expand(c, train_vectors.size(1), h)
        
    diff =  train_vectors - mean_vector  #f - u_c : (C,N ,H)

    diff = diff.permute(0,2,1)  #(C, H, N)
    print("diff size", diff.size())

    #covariance matrix : * = element wise multiplication 
    print("diff dtype", diff.dtype)
    cov = torch.matmul(diff, diff.permute(0,2,1))  #(C, H, N) x (C, N, H) -> (C, H, H)
    cov = cov / n
    print("covariance matrix size", cov.size())

    #count number of 0s 
    print("cov 0s : ", torch.sum(cov == 0).item())
    final_cov = cov.to(dtype=torch.float)

    if torch.isinf(final_cov).any() or torch.isnan(final_cov).any() : 
        raise Exception("cov matrix has NaN values")

    inv_cov = torch.empty(c,h,h) #(C, H, H)
    for i in range(c) :
        # reg_matrix = final_cov[i] + torch.eye(final_cov[0].size(1)) 
        print(final_cov[i].size())
        inv_cov[i] = torch.pinverse(final_cov[i]) # Σ^-1


    if torch.isinf(inv_cov).any() or torch.isnan(inv_cov).any() : 
        raise Exception("inv cov matrix has NaN values")
    
    print("inv cov 0s : ", torch.sum(inv_cov == 0).item())

    #maha metric 
    if metric == "maha" : 
        total_res = torch.zeros(c)
        c, n_test, _ = test_vectors.size()
        print("========================")
        print(f"TESTING SAMPLE {n_test} samples")
        inv_cov = inv_cov.to(dtype=test_vectors.dtype)
        
        mean_vector = u_vector.expand(c, test_vectors.size(1),h)

        #Z_test - μ 
        test_diff = (test_vectors - mean_vector)#(concept, BATCH SIZE, hidden size)
        res_1 = torch.matmul(test_diff, inv_cov) #(concept, BATCH SIZE, hidden size)
        res_2 = torch.matmul(res_1, test_diff.permute(0, 2, 1)) #(concept, BATCH SIZE, BATCH SIZE)

        res = torch.zeros(c)
        for i in range(c) : 
            diag = torch.diag(res_2[i]) #(1, d)
            #verify correctness via shape 
            print("diag size",diag.size())
            if (diag < 0).any() : 
                raise Exception("Diagonal values can't be negative")
            assert diag.size(0) == test_vectors.size(1) , "mismatch shape in diagonal values and n samples" 
            #avg dist across all samples 
            maha_score = -1 * torch.sqrt(diag) # (1,d) 
            total_score = torch.sum(maha_score) #sum maha score across all batch samples 
            total_score = total_score.item() #would this be on the CPU now 
            res[i] = total_score #(1, c)
            total_res = total_res + res 
            print("Done a sample")
        total_res = total_res / n_test #average maha score across all test samples
    
    #next perAnswerType 


    # print("scores shape", res.size()) #(c, )
    print(f"res shape : {res.size()}")
    
    return total_res.tolist() #(concept, 1)

res = score_func(train_vectors, test_vectors)

print("score", res)