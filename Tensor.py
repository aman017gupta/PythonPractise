import torch

if torch.cuda.is_available():
    device_name = torch.device("cuda")

print(f"selected device = {device_name}")
tensor = torch.tensor(10,device= device_name, requires_grad=True, dtype= torch.float16)
# print(torch.cuda.is_available())
print(tensor)

y_hat = tensor ** 4
print(y_hat)

y_hat.backward()
print(tensor.grad)

# print(torch.version.cuda)

# gpu_tensor = torch.tensor([10,20,30], device= device_name)

