import torch.nn as nn
import torch
import numpy as np
from DCFormer import DCTDetectionModel_MOE
from RGCN import multimodal_RGCN
import pdb
from decision import fushion_decision



class feature_align(nn.Module):
    def __init__(self,embedding_dim, align_size):
        super(feature_align,self).__init__()
       
        self.linear_relu_text=nn.Sequential(
            nn.Linear(embedding_dim,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_rgcn=nn.Sequential(
            nn.Linear(embedding_dim,int(align_size)),
            nn.LeakyReLU()
        )
        self.linear_relu_DCFormer=nn.Sequential(
            nn.Linear(embedding_dim,int(align_size)),
            nn.LeakyReLU()
        )
        
    def forward(self,text_tensor,combine_feature,DCT_feature):
        text_tensor,combine_feature, DCT_feature=self.linear_relu_text(text_tensor),self.linear_relu_rgcn(combine_feature),\
                                                        self.linear_relu_DCFormer(DCT_feature)
        return text_tensor,combine_feature,DCT_feature


class encodeMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(encodeMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()
        #self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.soft(out)
        return out

class MLPclassifier(nn.Module):
    def __init__(self,
                 input_dim=728,
                 output_size=2,
                 hidden_dim=128,
                 dropout=0.3):
        super(MLPclassifier, self).__init__()
        self.dropout = dropout
        
        
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, output_size)
    def forward(self,x):
        x = self.linear_relu_tweet(x)
        # x = self.linear_relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
class FixedPooling(nn.Module):
    def __init__(self, fixed_size):
        super().__init__()
        self.fixed_size = fixed_size

    def forward(self, x):
        b, w, h = x.shape
        p_w = self.fixed_size * ((w + self.fixed_size - 1) // self.fixed_size) - w
        p_h = self.fixed_size * ((h + self.fixed_size - 1) // self.fixed_size) - h
        x = nn.functional.pad(x, (0, p_h, 0, p_w))
        pool_size = (((w + self.fixed_size - 1) // self.fixed_size), ((h + self.fixed_size - 1) // self.fixed_size))
        pool = nn.MaxPool2d(pool_size, stride=pool_size)
        return pool(x)



class LModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.1, activation='ReLU',
                 norm_first=True, layer_norm_eps=1e-5,exp=2,k=1):
        super(LModel, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                         dropout=dropout, batch_first=True)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        if activation == 'SELU':
            self.activation = nn.SELU()
       
        #self.moe=MoE(embed_dim*3,embed_dim*3,exp,embed_dim*3,model=MLP,k=k)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, text_src):
        if self.norm_first:
            text, attention_weight = self._sa_block(self.norm1(text_src))
            #text = text_src + text
            #text1,loss=self._ff_block(self.norm2(text))
            #text = text + text1
        else:
            text, attention_weight = self._sa_block(text_src)
            text = self.norm1(text_src + text)
            text1,loss=self._ff_block(self.norm2(text))
            text = self.norm2(text + text1)
        return text, attention_weight

    def _sa_block(self, text):
        text, attention_weight = self.multihead_attention(text, text, text)
        text = self.dropout1(text)
        return text, attention_weight

    def _ff_block(self, text):
        #text = self.linear2(self.dropout(self.activation(self.linear1(text))))
        text_len=text.shape[1]
        text,loss=self.moe(text.reshape((len(text),-1)))
        text = self.dropout2(text)
        return text.reshape((len(text),text_len,-1)),loss

class FakeNewsDetection(nn.Module):
# feature align again in each model
    def __init__(self, args, embedding_dim, align_size=32, encode_model = encodeMLP, feature_out=32):
        super(FakeNewsDetection, self).__init__()
        self.args = args     
        self.feature_out = feature_out
        self.align=feature_align(embedding_dim = embedding_dim,align_size=align_size)
        self.mlp=nn.Sequential(
            nn.Linear(args.text_size, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim), 
            )

        self.text_classifer = nn.Linear(feature_out,2)
        self.fusion=LModel(embed_dim=feature_out)
        self.dct_fea=DCTDetectionModel_MOE(embedding_dim, feature_out = feature_out)
        self.gcn_fea=multimodal_RGCN(self.args, embedding_dim, feature_out = feature_out)
        self.text_fea = encode_model(embedding_dim, feature_out, 64) 
        self.dropout=nn.Dropout(0.1)
        self.fixed_pooling=FixedPooling(fixed_size=6)
        self.bn1=nn.BatchNorm1d(args.text_size)
        
        
        self.decision = fushion_decision(3, feature_out)
        self.ln  = nn.LayerNorm(feature_out)
        self.bn2=nn.BatchNorm1d(feature_out*3+6*6)
        self.mlp1 = nn.Linear(feature_out*3+6*6, feature_out)
        self.mlp_classifier= nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(feature_out, feature_out), 
            nn.LeakyReLU(),
            nn.Linear(feature_out, 2), 
            )
      
    
    def forward(self,text_tensor, img, graphs_all, graphs_post, graphs_image, type2nidxs, num_nodes, label, global_step):
        text_tensor = self.bn1(text_tensor)
        text_tensor = self.mlp(text_tensor)
        gcn_out, _=self.gcn_fea(graphs_all, graphs_post, graphs_image, type2nidxs, num_nodes)
        #print(gcn_out.shape)
        dct_out, _=self.dct_fea(img)
        bertemo_out = self.text_fea(text_tensor)
        #prob_text = self.text_classifer(bertemo_out)
        outputs=[gcn_out,dct_out,bertemo_out]
        
        evn_single, uncertain, loss = self.decision(outputs, label, global_step)
        
        

        out_tensor=torch.stack(outputs,dim=1)
        out_tensor = torch.mul(out_tensor,1 - uncertain)
        out_tensor= self.ln(out_tensor)  #self.bn1(out_tensor.permute(0,2,1)).permute(0,2,1)
        
        
        y=self.dropout(out_tensor)
        y,attention=self.fusion(out_tensor)
        attention=self.fixed_pooling(attention)
        y=torch.cat([y.reshape(len(y),self.feature_out*3),attention.reshape(len(attention),-1)],dim=1)
        y_embed = self.mlp1(y)
        y_out=self.mlp_classifier(y_embed)
        
        
        
        return y_out , y_embed, evn_single, uncertain, loss, gcn_out, dct_out, bertemo_out
        
        