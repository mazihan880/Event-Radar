import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Bias_loss(nn.Module):
    def __init__(self, temperature = 0.5):
        super(Bias_loss, self).__init__()
        self.temperature = temperature
        
    def forward(self, bias, features, gcn_feature, emofeature, dctfeature,index):
        features = F.normalize(features, p=2, dim=1)
        gcnfeature = F.normalize(gcn_feature, p=2, dim=1)
        emofeature = F.normalize(emofeature, p=2, dim=1)
        dctfeature = F.normalize(dctfeature, p=2, dim=1)
        
        s_fe = torch.bmm(features.unsqueeze(1), emofeature.unsqueeze(2)).squeeze(-1)/self.temperature
        s_fd= torch.bmm(features.unsqueeze(1), dctfeature.unsqueeze(2)).squeeze(-1)/self.temperature
        s_fg = torch.bmm(features.unsqueeze(1), gcnfeature.unsqueeze(2)).squeeze(-1)/self.temperature
        
        

        s_sum = torch.stack([s_fd, s_fe, s_fg], dim=1)
        maxs, _ = torch.max(s_sum, dim=1, keepdim=True)
        s_sum = torch.exp(s_sum - maxs) 
        bias_matrix = torch.stack([bias, bias, bias], dim=1)
        bias_matrix.scatter_(1, index+1, 1 - bias)

        
        #print(bias_matrix.shape)
        logits = torch.bmm(s_sum.permute(0,2,1), bias_matrix) / (torch.sum(torch.exp(s_sum), dim=1) + 1e-10)
        #print(logits.shape)

        logits = -torch.mean(torch.log(logits))
        
        return logits
        
        
        