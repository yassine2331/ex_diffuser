from torch import nn 
import torch

class CBM(nn.Module):
    def __init__(self,input_dim,num_concepts):
        super().__init__()
        self.linear =  nn.Linear( input_dim , num_concepts )
        self.m = nn.Softmax(dim=1)

    def forward(self, x:torch.Tensor):
        x=self.linear(x)
        x=self.m(x)
        return x
