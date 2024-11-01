import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

class class_token_pos_embed(nn.Module):
    def __init__(self, embed_dim):
        super(class_token_pos_embed, self).__init__()
        
        num_patches = patchembed().num_patches
        
        self.num_tokens = 1  
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+self.num_tokens, embed_dim))
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x): 
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return x

class patchembed(nn.Module):
    def __init__(self, img_size=int(224 * 0.75), patch_size=24, in_c=3, embed_dim=64):
        super(patchembed, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size//patch_size, img_size//patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, 
                              kernel_size=patch_size, stride=patch_size)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        

    
    def forward(self, inputs):
        B, C, H, W = inputs.shape
        assert H==self.img_size[0] and W==self.img_size[1], 

        x = self.proj(inputs)
        x = x.flatten(start_dim=2, end_dim=-1)  
        x = x.transpose(1, 2)  
        x = self.norm(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(drop)
    

    def forward(self, inputs):

        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x)
        
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class encoder_block(nn.Module):
    def __init__(self, dim, num_heads = 8, mlp_ratio=4., drop_ratio=0.5):
        super(encoder_block, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.atten = nn.MultiheadAttention(dim, num_heads, dropout=drop_ratio)
        self.drop = nn.Dropout()
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features)
        

    def forward(self, inputs):
        

        x = self.norm1(inputs)
        x, _ = self.atten(x, x, x)
        x = self.drop(x)
        feat1 = x + inputs  

        x = self.norm2(feat1)
        x = self.mlp(x)
        x = self.drop(x)
        feat2 = x + feat1  
        
        return feat2

class encodeMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(encodeMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class DCTDetectionModel_MOE(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, encode_model = encodeMLP, feature_out=32):
        super(DCTDetectionModel_MOE, self).__init__()
        self.dct_stem = patchembed(embed_dim=embedding_dim,)
        self.positional_encoding = class_token_pos_embed(embed_dim=embedding_dim)
        self.encoder_layer = nn.Sequential(*[encoder_block(dim = embedding_dim) for _ in range(3)]) 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.feature_layer = encode_model(embedding_dim, feature_out, 64)
        self.output =nn.Linear(feature_out, 2)
        self.dropout = dropout

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, img):
        patched_img = self.dct_stem(img)
        patched_img = self.positional_encoding(patched_img)
        attention_output = self.encoder_layer(patched_img)
        attention_output = self.norm(attention_output)
        
        attention_output = F.dropout(attention_output, p=self.dropout, training=self.training)
        out= self.feature_layer(torch.mean(attention_output, dim=1))
        prob = self.output(out)
        return out , prob
    
    