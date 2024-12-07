import torch
import csv
import json
import os
from sklearn.metrics import classification_report,auc,precision_recall_curve
from tqdm import tqdm
from torch_geometric.data import Batch
from torch.nn import functional as F
import pickle
def evaluation(outputs, labels):
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct
class Tester:
    def __init__(self, args, DetectionModel, test_set, test_size):
        self.args = args
        self.classifier = DetectionModel
        self.test_set = test_set
        self.test_size = test_size
        
        
    def test(self):
        self.classifier = torch.load(os.path.join(self.args.test_path ,f"{self.args.name}ckpt.classifier"), map_location= self.args.device)
        self.classifier.eval()
        preds = []
        ans_list = []
        id_list = []
        amb_preds = []
        emo_preds = []
        dct_preds = []
        all_label = []
        am_label = []
        emo_label = []
        dct_label = []
        y_embed_list= {}
        gcn_list = {}
        with tqdm(self.test_set) as pbar:
            for batches in pbar:
                
                #######Load Data#########
                ans = batches["label_list"].to(self.args.device)
                bert_emo = batches["bertemo"].to(self.args.device)
                dct_img = batches["dct_img"].to(self.args.device)

                post_graph, image_graph,  all_graph, type2nidx, num_nodes = batches["post_graph"], batches["image_graph"], batches["all_graph"], batches["type2idx"], batches["num_nodes"]
                post_graph = Batch.from_data_list(post_graph).to(self.args.device)
                image_graph = Batch.from_data_list(image_graph).to(self.args.device)
                all_graph = Batch.from_data_list(all_graph).to(self.args.device)
                Id = batches["Id"]
                y = ans.to(torch.long)   

                
                for ans_label in ans:
                    ans_label = int(ans_label)
                    ans_list.append(ans_label)
                
                for index in Id:
                    id_list.append(index)
                    
                with torch.no_grad():
                    
                    pred,  y_embed, env_single,  uncertain, certainloss, gcn_out, dct_embed, emo_embed = self.classifier(
                        bert_emo,
                        dct_img,
                        all_graph,
                        post_graph,
                        image_graph, 
                        type2nidx,
                        num_nodes,
                        y,
                        72,
                        )
                    for i, y_em in enumerate(y_embed):
                        
                        y_embed_list[Id[i]] = {
                            "label":int(ans[i]),
                            "embedding": y_em.cpu()
                        }
                        
                        
                    for j, g_em in enumerate(gcn_out):
                        gcn_list[Id[j]] = {
                            "label":int(ans[i]),
                            "embedding": g_em.cpu()
                        }
                        
                    
                    _, label= torch.max(pred,1)
                    _, a_label = torch.max(env_single[0],1)
                    _, e_label = torch.max(env_single[2],1)
                    _, d_label = torch.max(env_single[1],1)
                    fpred = F.softmax(pred, dim=-1)[:,1]
                    ambigious_pred = F.softmax(env_single[0], dim=-1)[:,1]
                    emo_pred= F.softmax(env_single[2], dim=-1)[:,1]
                    dct_pred= F.softmax(env_single[1], dim=-1)[:,1]
                    
                    for lall in label.cpu().numpy():
                       all_label.append(lall)
                    for la in a_label.cpu().numpy():
                       am_label.append(la)
                    for le in e_label.cpu().numpy():
                       emo_label.append(le)
                    for ld in d_label.cpu().numpy():
                       dct_label.append(ld)
                    
                    
                    for pp in fpred.cpu().numpy():
                        preds.append(pp)
                    
                        
                    for aa in ambigious_pred.cpu().numpy():
                        amb_preds.append(aa)
                    for ee in emo_pred.cpu().numpy():
                        emo_preds.append(ee)
                    for dd in dct_pred.cpu().numpy():
                        dct_preds.append(dd)
        
        print(classification_report(ans_list, all_label, digits=4))
        pickle.dump(y_embed_list, open("y_embed_twitter.pkl", "wb"))
        pickle.dump(gcn_list, open("gcn_embed_twitter.pkl", "wb"))
        with open(os.path.join(self.args.output_dir, f"{self.args.name}report.txt"), mode="w") as f:
            f.write("Total:\n")
            f.write(classification_report(ans_list,all_label,digits=4))
            f.write("Ambigious:\n")
            f.write(classification_report(ans_list,am_label,digits=4))
            f.write("Emotion:\n")
            f.write(classification_report(ans_list,emo_label,digits=4))
            f.write("DCT:\n")
            f.write(classification_report(ans_list,dct_label,digits=4))
            
        with open(os.path.join(self.args.output_dir, f"{self.args.name}result.txt"), mode="w") as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'label','result', 'pred', "gcn_pred", "emo_pred", "dct_pred", "u1", "u2", "u3"])
            print(len(id_list), len(ans_list), len(all_label), len(preds), len(amb_preds), len(emo_preds), len(dct_preds))
            for i,p in enumerate(preds):
                writer.writerow([id_list[i],ans_list[i], all_label[i], p , amb_preds[i], emo_preds[i], dct_preds[i]])