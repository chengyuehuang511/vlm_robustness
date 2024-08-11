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




