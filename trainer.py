import torch
from torch import nn
import os
import json
import torch.optim as optim
from tqdm import trange, tqdm
from sklearn.metrics import classification_report
from torch.nn import functional as F
from numpy import mean
from tabulate import tabulate
from torch_geometric.data import Batch
import numpy as np
from loss import Bias_loss
torch.autograd.set_detect_anomaly(True)




def evaluation(outputs, labels):
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct




class Trainer:
    def __init__(self, 
                 args,
                 DetectionModel,
                 tr_set,
                 tr_size, 
                 dev_set, 
                 dev_size):
        self.args = args 
        self.tr_set = tr_set
        self.tr_size = tr_size
        self.dev_set = dev_set
        self.dev_size = dev_size
        self.total_loss = nn.CrossEntropyLoss()
        self.bertemo_loss = nn.CrossEntropyLoss()
        self.dct_loss = nn.CrossEntropyLoss()
        self.ambigious = nn.CrossEntropyLoss()
        self.classifier = DetectionModel
        self.bias_loss = Bias_loss()
    
    def train(self):
        
        
        NET_Classifier = optim.Adam(self.classifier.parameters(), lr = self.args.lr,  weight_decay = 1e-6, eps = 1e-4)
        train_acc_values = []
        train_loss_values = []
        bias_list = []
        test_precision_values = []
        test_recall_values = []
        test_f1_values = []
        test_acc_values = []
        epoch_pbar = trange(self.args.num_epoch, desc="Epoch")
        best_acc=0
        for epoch in epoch_pbar:
            self.classifier.train()
            cls_loss = []
            total_train_acc = []
            certain_losses = []
            dct_acc = []
            emo_acc = []
            ambigious_acc = []
            
            
            with tqdm(self.tr_set) as pbar:
                for batches in pbar:
                    pbar.set_description(desc = f"Epoch{epoch}")
                    #######Load Data#########
                    y = batches["label_list"].to(self.args.device)
                    bert_emo = batches["bertemo"].to(self.args.device)
                    dct_img = batches["dct_img"].to(self.args.device)
                    post_graph, image_graph,  all_graph, type2nidx, num_nodes = batches["post_graph"], batches["image_graph"], batches["all_graph"], batches["type2idx"], batches["num_nodes"]
                    post_graph = Batch.from_data_list(post_graph).to(self.args.device)
                    image_graph = Batch.from_data_list(image_graph).to(self.args.device)
                    all_graph = Batch.from_data_list(all_graph).to(self.args.device)

                    y = y.to(torch.long)   
                    
                    pred,  y_embed, env_single,  uncertain, certainloss, gcn_out, dct_embed, emo_embed = self.classifier(
                        bert_emo,
                        dct_img,
                        all_graph,
                        post_graph,
                        image_graph, 
                        type2nidx,
                        num_nodes,
                        y,
                        epoch+1)

                    
                    _, label = torch.max(pred, dim = 1)
                    

                                  
                    total_loss = self.total_loss(pred, y) 
                    
                

                    #bias = uncertain[:, 0] 
                    bias, bias_index = torch.min(uncertain, dim=1, keepdim=True)
                    bias = bias.unsqueeze(1)
                    
                    biasloss = self.bias_loss(bias, y_embed, gcn_out, emo_embed, dct_embed,bia_index)

                    correct = evaluation(label, y)/len(label)
                    
                    
                    _, ambigious_label = torch.max(env_single[0], dim = 1)
                    _, emo_label = torch.max(env_single[2], dim = 1)
                    _, dct_label = torch.max(env_single[1], dim = 1)
                    ambigious_correct = evaluation(ambigious_label, y)/len(label)
                    dct_correct = evaluation(dct_label, y)/len(label)
                    emo_correct = evaluation(emo_label, y)/len(label)
                    
                    class_loss = total_loss+ self.args.alpha * certainloss +self.args.beta * biasloss
                    NET_Classifier.zero_grad()
                    class_loss.backward()
                    NET_Classifier.step()
                    
                    
                    cls_loss.append(class_loss.item())
                    certain_losses.append(certainloss.item())
                    
                    
                    dct_acc.append(dct_correct)
                    ambigious_acc.append(ambigious_correct)
                    emo_acc.append(emo_correct)
                    total_train_acc.append(correct)
                    
                    bias_list.append(biasloss.item())
                    pbar.set_postfix(loss = class_loss.item())
                    
                    
                    
                    

            train_loss_info_json = {"epoch": epoch,"Class_loss": mean(cls_loss), "certain loss":mean(certain_losses)}    
            train_acc_info_json = {"epoch": epoch,"train Acc": mean(total_train_acc), "ambigious acc":mean(ambigious_acc), "emo acc":mean(emo_acc), "dct acc":mean(dct_acc), "emo acc":mean(emo_acc), "bias": mean(bias_list)} 
            train_acc_values.append(mean(total_train_acc))
            train_loss_values.append(mean(cls_loss))
            print(f"{'#' * 10} TRAIN LOSSES: {str(train_loss_info_json)} {'#' * 10}")
            print(f"{'#' * 10} TRAIN ACCURACY: {str(train_acc_info_json)} {'#' * 10}")
            
            
            with open(os.path.join(self.args.output_dir, f"log{self.args.name}.txt"), mode="a") as fout:
                fout.write(json.dumps(train_loss_info_json) + "\n")
                fout.write(json.dumps(train_acc_info_json) + "\n")
            
            
                
            
            self.classifier.eval()  
            valid_acc = []
            ans_list = []
            preds = []
            
            
            
            with torch.no_grad():
                for batches in self.dev_set:

                    #######Load Data#########
                    y = batches["label_list"].to(self.args.device)
                    bert_emo = batches["bertemo"].to(self.args.device)
                    dct_img = batches["dct_img"].to(self.args.device)
                    
                    post_graph, image_graph,  all_graph, type2nidx, num_nodes = batches["post_graph"], batches["image_graph"], batches["all_graph"], batches["type2idx"], batches["num_nodes"]
                    post_graph = Batch.from_data_list(post_graph).to(self.args.device)
                    image_graph = Batch.from_data_list(image_graph).to(self.args.device)
                    all_graph = Batch.from_data_list(all_graph).to(self.args.device)
                    
                    for ans_label in y:
                        ans_label = int(ans_label)
                        ans_list.append(ans_label)
                    
                    
                    y=y.to(torch.long)
                    pred,  y_embed, env_single,  uncertain, certainloss, gcn_out, dct_embed, emo_embed = self.classifier(
                        bert_emo,
                        dct_img,
                        all_graph,
                        post_graph,
                        image_graph, 
                        type2nidx,
                        num_nodes,
                        y,
                        epoch+1)
                    
                    _, label= torch.max(pred,1)
                    
                    
                    correct = evaluation(label, y)/len(label)
                
                    valid_acc.append(correct)
                    
                    for p in label.cpu().numpy():
                        preds.append(p)
                        
                
                report = classification_report(ans_list, preds, digits=4, output_dict = True)
                
                #print(report)
                
                test_precision_values.append(float(report["macro avg"]["precision"]))
                test_recall_values.append(float(report["macro avg"]["recall"]))
                test_f1_values.append(float(report["macro avg"]["f1-score"]))

                            
                
                with open(os.path.join(self.args.output_dir, f"log{self.args.name}.txt"), mode="a") as f:
                    f.write(classification_report(ans_list,preds, digits=4))
                
                valid_info_json = {"epoch": epoch,"valid_Acc":mean(valid_acc)}
                test_acc_values.append(mean(valid_acc))
                print(f"{'#' * 10} VALID: {str(valid_info_json)} {'#' * 10}")
                self.print_result_table_handler(train_loss_values, train_acc_values,  test_acc_values, test_precision_values,test_recall_values,test_f1_values, report, print_type='tabel', table_type='pretty')
                
                with open(os.path.join(self.args.output_dir, f"log{self.args.name}.txt"), mode="a") as fout:
                    fout.write(json.dumps(valid_info_json) + "\n")
            
                if mean(valid_acc) > best_acc:
                    best_acc = mean(valid_acc)
                    torch.save(self.classifier, f"{self.args.ckpt_dir}/{self.args.name}ckpt.classifier")
                    print('saving model with acc {:.3f}\n'.format(mean(valid_acc)))
                    with open(os.path.join(self.args.output_dir, f"{self.args.name}best_valid_log.txt"), mode="a") as fout:
                        fout.write(json.dumps(valid_info_json) + "\n")
                        
                        
    def print_result_table_handler(self, loss_values, acc_values, 
                                   test_acc_values, 
                                   test_precision_values,test_recall_values,
                                   test_f1_values, report, print_type='tabel',
                                   table_type='pretty'):
        
        def trend(values_list):
            if len(values_list) == 1:
                diff_value = values_list[-1]
                return '↑ ({:+.6f})'.format(diff_value)
            else:
                diff_value = values_list[-1] - values_list[-2]
                if values_list[-1] > values_list[-2]:
                    return '↑ ({:+.6f})'.format(diff_value)
                elif values_list[-1] == values_list[-2]:
                    return '~'
                else:
                    return '↓ ({:+.6f})'.format(diff_value)
        
        if print_type == 'tabel':
            avg_table = [["train loss",loss_values[-1],trend(loss_values)],
                     ["train acc",acc_values[-1],trend(acc_values)],
                     ["test acc",test_acc_values[-1],trend(test_acc_values)],
                     ["test pre", test_precision_values[-1],trend(test_precision_values)],
                     ['test rec',test_recall_values[-1],trend(test_recall_values)],
                     ['test F1',test_f1_values[-1],trend(test_f1_values)]]


            avg_header = ['metric','value','trend']
            print((tabulate(avg_table, avg_header, floatfmt=".6f", tablefmt=table_type)))

            class_table = [['0', report["0"]["precision"], report["0"]["recall"], report["0"]["f1-score"], '{}/{}'.format(report["0"]["support"], report['macro avg']["support"])],
                          ['1', report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"], '{}/{}'.format(report["1"]["support"], report['macro avg']["support"])]]

            class_header = ['class', 'precision', 'recall', 'f1', 'support']
            print((tabulate(class_table, class_header, floatfmt=".6f", tablefmt=table_type)))
        else:
            print(("Average train loss: {}".format(loss_values[-1])))
            print(("Average train acc: {}".format(acc_values[-1])))
            print(("Average test acc: {}".format(test_acc_values[-1])))
            print(report)
                
                
                
         
         
 
        



