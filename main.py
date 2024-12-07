import random
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os
import torch
from transformers import  logging
from typing import Dict
from Dataset import NewsDataset
from Model import FakeNewsDetection
from torch.utils.data import DataLoader
import copy
from trainer import Trainer
from tester import Tester


import warnings


random.seed(1107)

warnings.filterwarnings("ignore")
logging.set_verbosity_warning()
logging.set_verbosity_error()

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]


def pickle_reader(path):
    return pickle.load(open(path, "rb"))


def main(args):
    
    
    ####initialize DataLoader
    data_paths = {split: args.data_dir / f"{split}.pkl" for split in SPLITS}
    data = {split: pickle_reader(path) for split, path in data_paths.items()}
    graph_data_path = {split: args.data_dir / f"graph_max_nodes_{split}.pkl" for split in SPLITS}
    graph_data = {split: pickle_reader(path) for split, path in graph_data_path.items()}
    
    datasets : Dict[str, NewsDataset] = {
        split: NewsDataset(graph_data[split], split_data)
        for split, split_data in data.items()
    }
    
    for split, split_dataset in datasets.items():
        if split == "train" and args.mode==0:
            tr_size = len(split_dataset)
            print("tr_size:",tr_size)
            tr_set = DataLoader(
                split_dataset,  batch_size = args.batch_size,collate_fn = split_dataset.collate_fn,
                shuffle = True, drop_last = True,
                num_workers = 0, pin_memory = False)
        elif split == "eval" and args.mode == 0:
            dev_size=len(split_dataset)
            print("dev_size:",dev_size)
            dev_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last=True,
                num_workers=0, pin_memory=False)
        elif args.mode == 1:
            test_size=len(split_dataset)
            print("test_size:",test_size)
            test_set=DataLoader(
                split_dataset,  batch_size=args.batch_size,collate_fn= split_dataset.collate_fn,
                shuffle=True, drop_last = True,
                num_workers=0, pin_memory=False)
            
    classifier = FakeNewsDetection(args, embedding_dim = args.embedding_dim)
    classifier.to(args.device)

        
    ifexist=os.path.exists(args.output_dir)
    if not ifexist:
        os.makedirs(args.output_dir)
    
    
    if args.mode==0: #train/dev
        args_dict_tmp = vars(args)
        args_dict = copy.deepcopy(args_dict_tmp)
        with open(os.path.join(args.output_dir, f"param_{args.name}.txt"), mode="w") as f:
            f.write("============ parameters ============\n")
            print("============ parameters =============")
            for k, v in args_dict.items():
                f.write("{}: {}\n".format(k, v))
                print("{}: {}".format(k, v))
        trainer=Trainer(args, classifier, tr_set, tr_size, dev_set, dev_size)
        trainer.train()
    else: #test
        tester=Tester(args, classifier, test_set,test_size,datasets["eval"])
        tester.test()
    
        
        
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type = Path,
        help = "Directory to the dataset",
        default = "../dataset/",
    )
    
    parser.add_argument(
        "--cache_dir",
        type = Path,
        help = "Directory to the preprocessed caches.",
        default = "./cache/",
    )
    
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./output/",
    )
    
    
    
    parser.add_argument(
        "--test_path",
        type=Path,
        help="Directory to load the test model.",
        default="./ckpt/",
    )
  
    parser.add_argument("--embedding_dim", type=float, default=64)
    parser.add_argument("--text_size", type=float, default=822)
    parser.add_argument('--dim_node_features', type=int, default=512)
    
    parser.add_argument("--num_dct", type=int, default=2)
    parser.add_argument("--num_gcn", type=int, default=3)
    parser.add_argument("--num_text", type=int, default=2)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument('--updated_weights_for_A', type=float,
                    default=0.5, help='1 - alpha')
    
    
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    
    
    parser.add_argument(
            "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:2"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--name", type=str, default="ex0")
    parser.add_argument("--mode", type=int, help="train:0, test:1", default=0)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

    
        
        
