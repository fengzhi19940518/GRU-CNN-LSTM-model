import torch
hidden=torch.randn(16,48,300)
x=torch.randn(16,48,1)

for i in range(hidden.size(0)):
    if i == 0:
        c= torch.mm((torch.t(hidden[i])),x[i])
        print(c.size())
    else:
        c=torch.cat([c,torch.mm((torch.t(hidden[i])),x[i])],1)
print(c.size())