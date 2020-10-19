import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from argparse import ArgumentParser

from utils import *
from model import BasicDistanceSearch, DistanceSearchSingleMove
from train_helpers import run_train
import torch
import pickle
import json

from torch import optim

parser = ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=200)
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument("--data_name", type=str, default="FB15K237_TransE")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--bs", type=int, default=512)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--use_cuda", action="store_true", default=True)
parser.add_argument("--log_dir", type=str, default="./log/")
parser.add_argument("--model_dir", type=str, default="./ckpt/")
parser.add_argument("--opt", type=str, default="adam")
parser.add_argument("--search_times", type=int, default=5)
args = parser.parse_args()

print("loading graph...")
ent_embeddings = pickle.load(open(args.data_dir + args.data_name + "/ent_embeddings.pkl", "rb"))
neighbor_info = pickle.load(open(args.data_dir + args.data_name + "/neighbor_info_padding.pkl", "rb"))
type_info = json.load(open(args.data_dir + args.data_name + "new_entity_type.json", "r"))
node_num = 14951
relation_num = 1345
type_num = 571
graph = dict()
graph["node_embedding"] = ent_embeddings
graph["neighbor"] = neighbor_info
graph["node_num"] = node_num
graph["rel_num"] = relation_num
graph["type_num"] = type_num
graph["type_info"] = type_info

print("loading queries...")
train_queries = json.load(open(args.data_dir + args.data_name + "/train_set.json", "r"))
val_queries = json.load(open(args.data_dir + args.data_name + "/dev_set.json", "r"))
test_queries = json.load(open(args.data_dir + args.data_name + "/test_set.json", "r"))


model = DistanceSearchSingleMove(graph=graph, k=None, embedding_dim=args.embed_dim, search_times=args.search_times)

if args.use_cuda:
    model.cuda()

if args.opt == "adam":
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr)

log_file = args.log_dir + "temp.log"
model_file = args.model_dir + "a.ckpt"

logger = setup_logging(log_file)

print("training len=3...")
run_train(model=model, optimizer=optimizer, train_queries=train_queries["3chain"], val_queries=val_queries["3chain"],
          test_queries=test_queries["3chain"], logger=logger, batch_size=args.bs, epochs=args.epoch,
          log_every=100, use_cuda=args.use_cuda)

# torch.save(model.state_dict(), model_file)