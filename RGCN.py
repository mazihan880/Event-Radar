import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv, inits
from utils import ZERO, normalized_correlation_matrix
import pdb


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

class multimodal_RGCN(nn.Module):
    def __init__(self, args, embedding_dim, dropout=0.1,encode_model = encodeMLP, feature_out=32):
        super(multimodal_RGCN, self).__init__()
        self.args = args
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.gnn_layers = []
        self.gnn_dynamic_update_weights = []
        for _ in range(args.num_gnn_layers):
            fc_conv = GCNConv(args.dim_node_features,
                                  args.dim_node_features, add_self_loops=False, normalize=False)
            post_conv = GCNConv(
                args.dim_node_features, args.dim_node_features, add_self_loops=False, normalize=False)
            image_conv = GCNConv(args.dim_node_features,
                                  args.dim_node_features, add_self_loops=False, normalize=False)
            self.gnn_layers.append(nn.ModuleDict(
                {'fc': fc_conv, 'post': post_conv, 'image': image_conv}))

            t = nn.Parameter(torch.Tensor(
                args.dim_node_features, args.dim_node_features))
            inits.glorot(t)

            self.gnn_dynamic_update_weights.append(t)

        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.gnn_dynamic_update_weights = nn.ParameterList(
            self.gnn_dynamic_update_weights)
        
        self.Hbn = nn.BatchNorm1d(args.dim_node_features)
        self.generate_feature = nn.Sequential(
            nn.Linear(in_features=args.dim_node_features * 4, out_features=args.dim_node_features),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(in_features=args.dim_node_features, out_features=self.embedding_dim), 
            )
        
        self.bn = nn.BatchNorm1d(self.embedding_dim * 5)
        
        self.ambigious_cal = nn.Sequential(
            nn.Linear(self.embedding_dim * 5, self.embedding_dim * 2),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim), 
            )
        
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)
        
        
        
        self.feature_layer = encode_model(embedding_dim, feature_out, 64)
        
        self.output =nn.Linear(feature_out, 2)

    def forward_GCN(self, GCN, x, graphs, A, layer_num):
        if layer_num == 0:
            edge_index, edge_weight = graphs.edge_index, graphs.edge_attr
        else:
            # --- Update edge_weights in graphs_* ---
            try:
                # (2, E)
                edge_index = graphs.edge_index
                E, N = len(graphs.edge_attr), len(A)
                # (E, N)
                start = F.one_hot(edge_index[0], num_classes=N)
                # (N, E)
                end = F.one_hot(edge_index[1], num_classes=N).t()

                # (E)
                edge_weight = torch.diag(start.float() @ A @ end.float())
                del start, end

            except:
                print('\n[Out of Memory] There are too much edges in this batch (num = {}), so it executes as a for-loop for this batch.\n'.format(len(graphs.edge_attr)))
                # (2, E)
                edge_index = graphs.edge_index
                edges_num = len(graphs.edge_attr)
                edge_weight = torch.zeros(
                    edges_num, device=self.args.device, dtype=torch.float)

                for e in tqdm(range(edges_num)):
                    a, b = graphs.edge_index[:, e]
                    edge_weight[e] = A[a, b]

        # （num_nodes_of_batch, 768)
        out = GCN(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return out

    def forward(self, graphs_all, graphs_post, graphs_image, type2nidxs, num_nodes):
        H = torch.clone(graphs_all.x)
        A = normalized_correlation_matrix(H)

        for i, gnn in enumerate(self.gnn_layers):
            H_all = self.forward_GCN(
                gnn['fc'], x=H, graphs=graphs_all, A=A, layer_num=i)
            H_post = self.forward_GCN(
                gnn['post'], x=H_all, graphs=graphs_post, A=A, layer_num=i)
            H_image = self.forward_GCN(
                gnn['image'], x=H_all, graphs=graphs_image, A=A, layer_num=i)

            # (num_nodes_in_batches, 768)
            H = F.leaky_relu(H_all + H_post + H_image)
            
            # --- Update adjacency_matrix ---
            A_hat = torch.sigmoid(
                H @ self.gnn_dynamic_update_weights[i] @ H.t())
            A = (1 - self.args.updated_weights_for_A) * \
                A + self.args.updated_weights_for_A * A_hat
                
        
        #post_index_lists = [torch.tensor(type_index["post_subgraph"], dtype=torch.long) for type_index in type2nidxs]

        H = self.Hbn(H)
        curr = 0
        post_subgraph = []
        image_subgraph = []
        for j, num in enumerate(num_nodes):
            post_nodes_idxs = type2nidxs[j]['post_subgraph']
            image_nodes_idxs = type2nidxs[j]['image_subgraph']
            curr_nodes = H[curr:curr+num]
            post_nodes = curr_nodes[list(post_nodes_idxs)]
            image_nodes = curr_nodes[list(image_nodes_idxs)]
            post_subgraph.append(post_nodes)
            image_subgraph.append(image_nodes)
            curr += num
        
        post_subgraph = torch.stack(post_subgraph, dim = 0)
        image_subgraph = torch.stack(image_subgraph, dim = 0)
        #print(post_subgraph.shape)
        combine_feature = torch.cat([post_subgraph, image_subgraph, post_subgraph - image_subgraph, post_subgraph * image_subgraph],  dim=-1)
        combine_feature = self.generate_feature(combine_feature)
        
        combine_feature = combine_feature.reshape(-1, 5 * self.embedding_dim)
        
        combine_feature = self.bn(combine_feature)

        combine_feature = self.ambigious_cal(combine_feature)
        combine_feature = self.bn2(combine_feature)
        out = self.feature_layer(combine_feature)

        prob = self.output(out)

        return out , prob


