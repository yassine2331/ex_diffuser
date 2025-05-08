from multiprocessing import context
from turtle import forward
from torch import nn 
import torch
from torch.onnx.symbolic_opset9 import zero

class CBM(nn.Module):
    def __init__(self,input_dim,num_concepts):
        super().__init__()
        self.linear =  nn.Linear( input_dim , num_concepts )
        self.m = nn.Softmax(dim=1)

    def forward(self, x:torch.Tensor):
        x=self.linear(x)
        x=self.m(x)
        return x


class Context_newtork(nn.Module):
    def __init__(self,input_dim, hidden_dim ):
        super().__init__()
        self.fc_pos = nn.Linear(input_dim, hidden_dim)
        self.fc_neg = nn.Linear(input_dim, hidden_dim)
        self.act_pos = nn.SiLU()
        self.act_neg = nn.SiLU()
        
    
    def forward(self, x):
        context_pos = self.fc_pos(x)
        context_pos = self.act_pos(context_pos)
        
        context_neg = self.fc_pos(x)
        context_neg = self.act_neg(context_neg)


        return context_pos, context_neg

class Probability_network(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.pre_prob = nn.Linear(hidden_dim*2 , hidden_dim)
        self.act = nn.SiLU()
        self.prob = nn.Linear(hidden_dim,2)
        self.act_prob = nn.Softmax(dim= 1)
    def forward(self,x,y):
        z = torch.cat((x,y), dim=1)
        z = self.pre_prob(z)
        z = self.act(z)
        z = self.prob(z)
        z = self.act_prob(z)
        
        return z[:, 0],z[:, 1]
        

class Context(nn.Module):
    def __init__(self, input_dim , context_dim, intervention = None):
        super().__init__()
        self.intervention = intervention
        self.context_newtork = Context_newtork(input_dim=input_dim , hidden_dim=context_dim)
        self.Probability_network = Probability_network(hidden_dim=context_dim)
        self.context_dim = context_dim

    def forward(self,x):
        
        context_pos, context_neg = self.context_newtork(x)
        prob_pos, prob_neg = self.Probability_network(context_pos, context_neg)
       
        if self.intervention == None:
            #context_pos = torch.mul(prob_pos,context_pos )
            #context_neg = torch.mul( prob_neg,context_neg )
            context_pos = prob_pos.view(prob_pos.shape[0],1) * context_pos
            context_neg = prob_neg.view(prob_neg.shape[0],1) * context_neg
            
        else:
            context_pos = self.intevention.view(self.intevention.shape[0],1) * context_pos
            context_neg = (1-self.intervention.view(self.intevention.shape[0],1)) * context_neg

        context = context_pos + context_neg
        return (context , prob_pos) 


class CBM_new(nn.Module):
    def __init__(self,input_dim,num_concepts, context_dim= 5,skip_context = False, intervention=None ):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contexts = nn.ModuleList()
        self.num_concepts = num_concepts
        self.context_dim = context_dim # tempo
        self.skip_context = skip_context
        
        
        if intervention is None:
            for i in range(self.num_concepts):
                self.contexts.append(Context(input_dim,self.context_dim,intervention))
        else:
            for i in range(self.num_concepts):
                self.contexts.append(Context(input_dim,self.context_dim,intervention[:,i]))




        

        if skip_context:
            self.skip_context = nn.Linear(input_dim, self.context_dim)
            self.act_skip = nn.SiLU()
        
    def forward(self, x):
        concepts = []
        probs = []
        for i in range(self.num_concepts):
            c = self.contexts[i](x)[0]
            p = self.contexts[i](x)[1]
            
            concepts.append(c)
            probs.append(p)
        if self.skip_context:
            skip_c = self.skip_context(x)
            skip_act = self.act_skip(skip_c)
            concepts.append(skip_act)
        h = torch.cat(concepts,dim=1)
        probs = torch.stack(probs, dim=1)
        return h , probs



        
