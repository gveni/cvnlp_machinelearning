import torch

x = torch.rand(5, 3, dtype=torch.float)
print(x)

x_copy = torch.randn_like(x, dtype=torch.float)
print(x_copy)
print("Size of x_copy", x_copy.size())
x_copy.add_(x)
print("Mutated x_copy after adding x to it", x_copy)

x_reshaped = x.view(x.size()[0]*x.size()[1]) 
print("Reshaped x", x_reshaped.size())

"""Moving tensors from one device to other"""
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.float))
else:
    print("CUDA not available")
