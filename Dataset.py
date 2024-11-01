import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

class NewsDataset(Dataset):
    def __init__(
        self,
        graph_data,
        dct_data
        ):
        super(NewsDataset, self).__init__()
        self.data = dct_data
        self.nodes_features, self.edges2postgraph, self.edges2imagegraph, self.edges2others, self.edges2fcnodes, self.type2nidxs = graph_data
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        instance = self.data[index]
        label = instance["label"]
        Id = instance["Id"]
        dct_img = instance["dct_img"]
        bertemo = instance["post"]
        nodes = self.nodes_features[Id]
        edges2postgraph = self.edges2postgraph[Id]
        edges2imagegraph = self.edges2imagegraph[Id]
        edges2fcnodes = self.edges2fcnodes[Id]
        type2nidxs = self.type2nidxs[Id]
        
        postgraph =  self.get_graph_data(nodes, edges2postgraph, 'post_subgraph')
        image_graph =  self.get_graph_data(nodes, edges2imagegraph, 'image_subgraph')
        all_graph = self.get_graph_data(nodes, edges2fcnodes, 'all_nodes')
        num_nodes = len(nodes)
        types = type2nidxs
        
        

        return  (label, dct_img, bertemo, postgraph, image_graph, all_graph, types,num_nodes, Id, index)
       
    
        
    def collate_fn(self, samples) :   
        batch={}
        label_list=[]
            
        for s in samples:
            label_list.append(int(s[0]))
        label_list=torch.tensor(label_list)
        batch["label_list"]=label_list

        # ========== news ==============
        
        ids=[s[1] for s in samples]
        batch['dct_img'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        
        ids=[s[2] for s in samples]
        batch['bertemo'] = torch.Tensor(np.array([item.cpu().numpy() for item in ids]))
        
        ids=[s[3] for s in samples]
        batch['post_graph'] = ids
        
        ids=[s[4] for s in samples]
        batch['image_graph'] = ids
        
        ids=[s[5] for s in samples]
        batch['all_graph'] = ids
        
        ids=[s[6] for s in samples]
        batch['type2idx'] = ids
        
        ids=[s[7] for s in samples]
        
        batch['num_nodes'] = ids
        
        
        
        # ========== ID ============== 
        Id = [s[8] for s in samples]      
        batch['Id'] = Id
        return batch
    
    def get_graph_data(self, nodes, edges, node_type):
        return Data(x=nodes, edge_index=edges['index'], edge_attr=edges['weight'], y=node_type)
    

            
            
        
        
        

    