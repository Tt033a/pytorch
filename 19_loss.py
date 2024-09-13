import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input=torch.tensor([1,2,3],dtype=torch.float32)
targets=torch.tensor([1.0,2,5])
print(input.shape)
input=torch.reshape(input,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))
print(input)#tensor([[[[1, 2, 3]]]])
print(targets)#tensor([[[[1, 2, 5]]]])

#L1Loss 可算均值或者求和
loss1=L1Loss(reduction="sum")
result1=loss1(input,targets)

#MSELoss
loss2=MSELoss()
result2=loss2(input,targets)

print(result1)#tensor(2.)
print(result2)#tensor(1.3333)

#CrossEntropyLoss 交叉熵
x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor(([1]))
x=torch.reshape(x,[1,3])
y=torch.reshape(y,[1])
loss3=CrossEntropyLoss()
result3=loss3(x,y)
print(result3)