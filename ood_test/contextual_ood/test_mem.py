import torch 


# t = torch.rand(2, 400000, 2048).to('cuda')
# u = torch.rand(2,400000, 2048).to('cuda')

# print(t-u)

hist1 = torch.tensor([3,3,5,6])
hist2 = torch.tensor([2,3,2,1])
mi = min(hist1.min().item(), hist2.min().item()) 
ma = min(hist2.max().item(), hist2.max().item())


bin_edges = torch.linspace(mi, ma , 5)

intersection = torch.min(hist1, hist2)
best_inter_idx = torch.argmax(intersection).item()
print("type best intersect idx", best_inter_idx)

intersect_ranges = [(bin_edges[best_inter_idx].item(), bin_edges[best_inter_idx + 1].item())]
start, end = intersect_ranges[0]
print("start end", start, end)




