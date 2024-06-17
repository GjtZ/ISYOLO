#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

def gram_matrix(x):
    batch, channel, h, w = x.shape
    before_gram = x.clone()
    before_gram = before_gram.view(batch, channel, h*w)
    gram = torch.bmm(before_gram, before_gram.permute(0, 2, 1))/(h*w)
    return gram

class Implicit_C(nn.Module):              
    def __init__(self, channel, num_imp=4, gram_mode='gram'):
        super(Implicit_C, self).__init__()
        self.channel = channel
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        

        self.imp = nn.Parameter(torch.zeros(1, self.channel, self.num_imp))   
        nn.init.normal_(self.imp, std=.02)

        if self.gram_mode =='gram':
            self.dynamic_layer = nn.Linear(self.channel*self.channel, self.num_imp)        
        elif self.gram_mode == 'gram_4':
            self.dynamic_layer = nn.Linear(4*self.channel, self.num_imp)
            self.linear1 = nn.Linear(self.channel, 1)

    def forward(self,gram):                                                                      
        if self.gram_mode =='gram':
            gram_ = gram.view(-1, self.channel*self.channel)                                             
        elif self.gram_mode == 'gram_4':
            gram_max, _ = torch.max(gram, dim=2)                       
            gram_avg = torch.mean(gram, dim=2)                         
            gram_diag = torch.stack([torch.diag(i) for i in gram], dim=0)           
            gram_linear = self.linear1(gram).squeeze(2)                        
            gram_ = torch.cat([gram_max, gram_avg, gram_diag, gram_linear], dim=1)     
        
        parameters = self.dynamic_layer(gram_).unsqueeze(2)                                   

        parameters_=parameters/(parameters.sum(axis=1,keepdims=True)+1e-12)                           
        # 1.sum    2. max-min   3.sigmoid()
        # parameters_=parameters/(parameters.sum(axis=1,keepdims=True) + 1e-05 )                          
        # parameters_=parameters/(parameters.max(dim=1,keepdim=True)[0] - parameters.min(dim=1,keepdim=True)[0]  + 1e-05 )
        # parameters_=parameters.sigmoid() / ( parameters.sigmoid().sum(axis=1,keepdims=True) + 1e-05 )
        imp1 = torch.matmul(self.imp, parameters_)                         # (B,C,1)
        select_imp = imp1.unsqueeze(3)                     # (B,C,1) -> (B,C,1,1)
        return select_imp

    
class ImplicitAdd_C(nn.Module):                        # add
    def __init__(self, channel, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.channel = channel
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_add = Implicit_C(self.channel, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_add.imp, std=.02)

    def forward(self, x):
        gram = gram_matrix(x)   
        x_ = self.imp_add(gram).expand_as(x) + x
        return x_


class ImplicitMul_C(nn.Module):                         # mul
    def __init__(self, channel, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.channel = channel
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_mul = Implicit_C(self.channel, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        gram = gram_matrix(x)   
        x_ = self.imp_mul(gram).expand_as(x) * x
        return x_


class ImplicitShift_C(nn.Module):                      # shift
    def __init__(self, channel, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.channel = channel
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_add = Implicit_C(self.channel, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_add.imp, std=.02)
        self.imp_mul = Implicit_C(self.channel, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        gram = gram_matrix(x)   
        x_ = self.imp_mul(gram).expand_as(x) * x + self.imp_add(gram).expand_as(x)
        return x_