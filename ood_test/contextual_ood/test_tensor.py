import torch 



# random_tensor = torch.randint(low=1,high=5, size= (2, 2, 3))
# print(random_tensor)
# print(torch.matmul(random_tensor, random_tensor.permute(0, 2,1 )))
# print(torch.matmul(random_tensor, random_tensor.permute(0, 2,1 )).size())


# tensor1 = torch.randn(32, 10, 256)
# tensor2 = torch.randn(32, 10, 256)
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



print('_'.join(["hello", "world"]))