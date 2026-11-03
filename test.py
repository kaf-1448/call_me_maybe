import torch


x = torch.tensor([1.0, 2.0, 3.0])

print(x.device)

if torch.cuda.is_available():
    x = x.to("cuda")
    print()
