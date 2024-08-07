import torch 


t = torch.rand(2, 400000, 2048).to('cuda')
u = torch.rand(2,400000, 2048).to('cuda')

print(t-u)





