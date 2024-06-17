import torch
import torch.nn as nn
import torch.nn.functional as F                      
# import math

def gram_matrix(x):   
    batch, channel, h_, w_= x.shape
    before_gram = x.clone()
    before_gram_ = before_gram.view(batch, channel, h_*w_)
    gram = torch.bmm(before_gram_.permute(0, 2, 1), before_gram_)/(channel)
    return gram

class Implicit_S(nn.Module):                
    def __init__(self, h, w,num_imp=4, gram_mode='gram',model_scale='yolox-l'):
        super(Implicit_S, self).__init__()
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.h1 = h
        self.w1 = w
        self.h_gram = round(600/64)  # feature
        self.w_gram = round(960/64)
        
        self.model_scale=model_scale

        self.imp = nn.Parameter(torch.zeros(1, self.h1*self.w1, self.num_imp))      
        #nn.init.normal_(self.imp, std=.02)

        # if model == 'yolox-l'
        if self.model_scale=='yolox-l':
            self.dynamic_layer = nn.Linear((self.h_gram*self.w_gram)**2, 4*self.num_imp) 
            self.dynamic_layer1 = nn.Linear(4*self.num_imp, self.num_imp)  
        else:
        #  if model =='yolox-s' or 'yolox-m'
            self.dynamic_layer = nn.Linear((self.h_gram*self.w_gram)**2, self.num_imp) 


    def forward(self, gram_):  
        if self.model_scale=='yolox-l':
        # if model == 'yolox-l'                                            
            parameters1 = F.relu(self.dynamic_layer(gram_))                  
            parameters = self.dynamic_layer1(parameters1).unsqueeze(2)   
        else:    
        # if model =='yolox-s' or 'yolox-m'
            parameters = self.dynamic_layer(gram_).unsqueeze(2)  

        parameters_=parameters/parameters.sum(axis=1,keepdims=True)                              # (B,4,1)/(B,1,1)
        # parameters_=parameters/(parameters.sum(axis=1,keepdims=True) + 1e-05 )                             # (B,4,1)/(B,1,1)
        # parameters_=parameters/(parameters.max(dim=1,keepdim=True)[0] - parameters.min(dim=1,keepdim=True)[0]  + 1e-05 )
        # parameters_=parameters.sigmoid() / ( parameters.sigmoid().sum(axis=1,keepdims=True) + 1e-05 )

        imp1 = torch.matmul(self.imp, parameters_)                      
        select_imp = imp1.unsqueeze(3).view(-1,1,self.h1,self.w1)        
        return select_imp


    
class ImplicitAdd_S(nn.Module):                    # add
    def __init__(self, h,w, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.h_ = h
        self.w_= w
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_add = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_add.imp, std=.02)

    def forward(self, x):
        h_current,w_current = x.shape[2:4]
        imp_add_scale = F.interpolate(self.imp_add(x),size=(h_current,w_current), mode='bilinear')
        x_ = imp_add_scale.expand_as(x) + x
        return x_


class ImplicitMul_S(nn.Module):                    # mul
    def __init__(self, h, w, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.h_ = h
        self.w_ = w
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_mul = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        h_current,w_current = x.shape[2:4]
        imp_mul_scale = F.interpolate(self.imp_mul(x), size=(h_current,w_current), mode='bilinear')
        x_ = imp_mul_scale.expand_as(x) * x
        return x_
class ImplicitMul_S_8(nn.Module):                  # mul  &  Specify the resolution
    def __init__(self, h, w, num_imp=4, gram_mode='gram'):
        super().__init__()
        self.h_ = round(600/16)
        self.w_ = round(960/16)
        self.num_imp = num_imp
        self.gram_mode = gram_mode
        self.imp_mul = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        x16=F.interpolate(x,size=(self.h_,self.w_),mode='bilinear')
        h_current,w_current = x.shape[2:4]
        imp_mul_scale = F.interpolate(self.imp_mul(x16), size=(h_current,w_current), mode='bilinear')
        x_ = imp_mul_scale.expand_as(x) * x
        return x_


class ImplicitShift_S(nn.Module):                    # shift                      
    def __init__(self, h, w, num_imp=4, gram_mode='gram',model_scale='yolox-l'):
        super().__init__()
        self.h_ = h
        self.w_ = w    
        self.h_gram = round(600/64)  # feature
        self.w_gram = round(960/64)  
        self.num_imp = num_imp
        self.gram_mode = gram_mode

        self.model_scale=model_scale

        self.imp_add = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode,model_scale=self.model_scale)
        nn.init.normal_(self.imp_add.imp,mean=0., std=.02)
        self.imp_mul = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode,model_scale=self.model_scale)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        h_current,w_current = x.shape[2:4]
        x64=F.interpolate(x,size=(self.h_gram,self.w_gram),mode='bilinear')
        gram_scale = gram_matrix(x64).unsqueeze(1).squeeze(1)            # (B ,h*w,h*w)--->(B ,1, h*w,h*w)
        gram_ = gram_scale.view(-1, (self.h_gram*self.w_gram)**2)            # (B,h*w*h*w)   

        imp_mul_scale = F.interpolate(self.imp_mul(gram_), size=(h_current,w_current), mode='bilinear')
        imp_add_scale = F.interpolate(self.imp_add(gram_),size=(h_current,w_current), mode='bilinear')

        x_ = imp_mul_scale.expand_as(x) * x +imp_add_scale.expand_as(x)  
        return x_

    
class ImplicitShift_S_8(nn.Module):          # shift  & Specify the resolution                                 
    def __init__(self, h, w,num_imp=4, gram_mode='gram',model_scale='yolox-l'):
        super().__init__()
        self.h_ = round(600/16)  #  implicit knowledge 
        self.w_ = round(960/16)
        
        self.h_gram = round(600/64)  # gram feature scale
        self.w_gram = round(960/64)
        
        self.num_imp = num_imp
        self.gram_mode = gram_mode

        self.imp_add = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode,model_scale=self.model_scale)
        nn.init.normal_(self.imp_add.imp,mean=0., std=.02)
        self.imp_mul = Implicit_S(self.h_, self.w_, num_imp=self.num_imp, gram_mode=self.gram_mode,model_scale=self.model_scale)
        nn.init.normal_(self.imp_mul.imp, mean=1., std=.02)

    def forward(self, x):
        h_current,w_current = x.shape[2:4]
        x64=F.interpolate(x,size=(self.h_gram,self.w_gram),mode='bilinear')
        gram_scale = gram_matrix(x64).unsqueeze(1).squeeze(1)          
        gram_ = gram_scale.view(-1, (self.h_gram*self.w_gram)**2)        

        imp_mul_scale = F.interpolate(self.imp_mul(gram_), size=(h_current,w_current), mode='bilinear')
        imp_add_scale = F.interpolate(self.imp_add(gram_),size=(h_current,w_current), mode='bilinear')

        x_ = imp_mul_scale.expand_as(x) * x +imp_add_scale.expand_as(x) 
        return x_

    

    
    