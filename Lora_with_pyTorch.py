from numpy.linalg import qr,eig,inv,matrix_rank,inv, norm
from scipy.linalg import null_space
from sympy import Matrix, init_printing,Symbol
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pretrained_model_implementation import (
    TextClassifier,
    model,
    train_model,
    train_loader,
    valid_loader,
    evaluate,
    test_loader,

)
checkpoint = torch.load(
    "my_model.pth",
    map_location=device,
    weights_only=False
)
model=TextClassifier(num_classes=4,freeze=False)#We have to build model again.
state_dict = torch.load("my_model.pth", map_location=device) #then call parameters and load to model
model.load_state_dict(state_dict)


class LoRALayer(nn.Module):
    def __init__(self, in_dim,out_dim, rank:int, alpha) -> None:
        super().__init__()
        #its like Xavier / He initialization, for stabilize to learning and prevent grad explosion.
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        #matris A 
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        #matris B, It began with zeros because designers want not to effect weights from begining.
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        #It will be same size with linear end of do forward (in_dim,out_dim)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B) # matris multipy
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear.to(device)
        #Lora matris size arranged with linear's sizes
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        ).to(device)

    def forward(self, x):
        #lora added Linear
        return self.linear(x) + self.lora(x)





# Here, you freeze all layers:
for parm in model.parameters():
    parm.requires_grad=False
print(checkpoint)

model.fc2=nn.Linear(in_features=128, out_features=2, bias=True).to(device)
model.fc1=LinearWithLoRA(model.fc1,rank=2, alpha=0.1).to(device)

print(f"fc1: {model.fc1}\n fc2: {model.fc2}") # You can see dimension sizes

LR=1
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
model_name="model_lora_final2"
train_model(model,optimizer, criterion, train_loader, valid_loader, epochs=4, model_name=model_name)
evaluate(test_loader , model, device)
