import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class BasicDistanceSearch(nn.Module):
    """
    A basic search way by distance
    """
    def __init__(self, graph, k, embedding_dim, search_times, strategy=None):
        super(BasicDistanceSearch, self).__init__()
        self.graph = graph
        # get node embedding to tensor
        node_num = graph["node_embedding"].num_embeddings
        emb = graph["node_embedding"](torch.LongTensor([i for i in range(node_num)]))
        self.node_embedding = torch.nn.Parameter(torch.cat((
            emb, torch.zeros(embedding_dim).unsqueeze(0)), 0), requires_grad=False)

        self.node_neighbors = torch.nn.Parameter(torch.LongTensor(
            [graph["neighbor"][i][0] for i in graph["neighbor"]]), requires_grad=False)
        self.rel_neighbors = torch.nn.Parameter(torch.LongTensor(
            [graph["neighbor"][i][1] for i in graph["neighbor"]]), requires_grad=False)
        self.node_num = graph["node_num"]
        self.rel_num = graph["rel_num"]
        self.type_num = graph["type_num"]
        # self.node_type = graph["node_type"]
        self.dim = embedding_dim
        self.k = k
        self.st = search_times

        self.node_weight = torch.nn.Parameter(torch.Tensor(self.node_num+1).uniform_(0, 1), requires_grad=True)
        self.rel_weight = torch.nn.Parameter(torch.Tensor(self.rel_num+1).uniform_(0, 1), requires_grad=True)
        self.type_weight = torch.nn.Parameter(torch.Tensor(self.type_num+1).uniform_(0, 1), requires_grad=True)
        self.rel_eye = torch.nn.Parameter(torch.eye(self.rel_num+1), requires_grad=False)






    def forward(self, queries):
        '''

        :param queries: np.ndarray len * bs, query = (e1, r1, e2, r2, e3), len(query) = 5;
        query = (e1, r1, e2, r2, e3, r3, e4), len(query) = 7;
        :return: vector of a distance score
        '''
        # losses = []
        e1s = (queries["e1s"])
        r1s = (queries["r1s"])
        e2s = (queries["e2s"])
        r2s = (queries["r2s"])
        e3s = (queries["e3s"])
        e1_embeddings = self.node_embedding[e1s]
        e2_embeddings = self.node_embedding[e2s]
        e3_embeddings = self.node_embedding[e3s]
        e1_neighbors = self.node_neighbors[e1s]
        e2_neighbors = self.node_neighbors[e2s]

        e1_neighbor_embeddings = self.node_embedding[e1_neighbors] 
        e2_neighbor_embeddings = self.node_embedding[e2_neighbors]

        # find e2
        e1e2_dist = self.euclid_dist(e1_embeddings, e2_embeddings).unsqueeze(-1) # bs * 1
        e1neighbor_e2_dist = self.euclid_dist(e1_neighbor_embeddings, e2_embeddings.unsqueeze(1)) # bs * neighbor_num
        e1e2_neighbor_dist = e1e2_dist - e1neighbor_e2_dist # bs * neighbor_num
        e1e2_neighbor_calculate = e1e2_neighbor_dist - e1e2_neighbor_dist.mean(dim=1).unsqueeze(1) # bs * neighbor_num


        current_embeddings = e1_embeddings # bs * dim
        for i in range(self.st):
            e1_neighbors_type_weight = self.get_type_weight([e1s], e1_neighbors)


            if e1_neighbors_type_weight != None:
                e1_neighbor_weight = self.node_weight[e1_neighbors] + e1_neighbors_type_weight
            else:
                e1_neighbor_weight = self.node_weight[e1_neighbors]
            e1_neighbor_weight = F.softmax(e1_neighbor_weight)
            current_neighbor_vec = current_embeddings.unsqueeze(1) - e1_neighbor_embeddings
            # neighbor_weight * neighbor_calculate--0~1 * current_neighbor_vec
            # e1_neighbor_weight = e1e2_neighbor_calculate * e1_neighbor_weight
            moves = e1_neighbor_weight.unsqueeze(2) * current_neighbor_vec
            moves = moves.mean(1).div(self.st)
            current_embeddings = current_embeddings + moves

        losses = self.euclid_dist(current_embeddings, e2_embeddings)

        # find e3
        e2e3_dist = self.euclid_dist(e2_embeddings, e3_embeddings).unsqueeze(-1) # bs * 1
        e2neighbor_e3_dist = self.euclid_dist(e2_neighbor_embeddings, e3_embeddings.unsqueeze(1))
        e2e3_neighbor_dist = e2e3_dist - e2neighbor_e3_dist  # bs * neighbor_num
        e2e3_neighbor_calculate = e2e3_neighbor_dist - e2e3_neighbor_dist.mean(dim=1).unsqueeze(1)
        for i in range(self.st):
            e2_neighbors_type_weight = self.get_type_weight([e1s, e2s], e2_neighbors)
            e2_neighbors_edge_weight = self.get_edge_weight(r1s.unsqueeze(1), self.rel_neighbors[e2s])
            if e2_neighbors_type_weight != None:
                e2_neighbor_weight = self.node_weight[e2_neighbors] + e2_neighbors_type_weight + e2_neighbors_edge_weight
            else:
                e2_neighbor_weight = self.node_weight[e2_neighbors] + e2_neighbors_edge_weight
            e2_neighbor_weight = F.softmax(e2_neighbor_weight)
            current_neighbor_vec = current_embeddings.unsqueeze(1) - e2_neighbor_embeddings

            moves = e2_neighbor_weight.unsqueeze(2) * current_neighbor_vec
            moves = moves.mean(1).div(self.st)
            current_embeddings = current_embeddings + moves

        losses = losses + self.euclid_dist(current_embeddings, e3_embeddings)

        if len(queries) == 7:
            # find e4
            r3s = torch.LongTensor(queries["r3s"])
            e4s = torch.LongTensor(queries["e4s"])
            e4_embeddings = self.node_embedding[e4s]
            e3_neighbors = self.node_neighbors[e3s]
            e3_neighbor_embeddings = self.node_embedding[torch.LongTensor(e3_neighbors)]

            e3e4_dist = self.euclid_dist(e3_embeddings, e4_embeddings).unsqueeze(-1)
            e3neighbor_e4_dist = self.euclid_dist(e3_neighbor_embeddings,
                                                  e4_embeddings.unsqueeze(1))
            e3e4_neighbor_dist = e3e4_dist - e3neighbor_e4_dist
            e3e4_neighbor_calculate = e3e4_neighbor_dist - e3e4_neighbor_dist.mean(dim=1).unsqueeze(1)
            for i in range(self.st):
                e3_neighbors_type_weight = self.get_type_weight([e1s, e2s, e3s], e3_neighbors)
                e3_neighbors_edge_weight = self.get_edge_weight(torch.cat((r1s.unsqueeze(1), r2s.unsqueeze(1)), dim=1), self.rel_neighbors[e3s])
                neighbor_weight = self.node_weight[e3_neighbors] + e3_neighbors_type_weight + e3_neighbors_edge_weight
                neighbor_weight = F.softmax(neighbor_weight)
                current_neighbor_vec = current_embeddings.unsqueeze(1) - e3_neighbor_embeddings

                neighbor_weight = e3e4_neighbor_calculate * neighbor_weight
                moves = neighbor_weight.unsqueeze(2) * current_neighbor_vec
                moves = moves.mean(1)
                current_embeddings = current_embeddings + moves
            losses = losses + self.euclid_dist(current_embeddings, e4_embeddings)

        loss = losses.mean()

        return loss



    def node2type(self, nodes):
        return [self.node_type[i] for i in nodes]

    def euclid_dist(self, a, b):
        return torch.sqrt(torch.sum((a-b) ** 2, dim=-1)).cuda()

    def get_type_weight(self, history_nodes, neighbor_nodes):
        '''

        :param history_nodes: a list. nodes appeared in the history
        :param neighbor_nodes: current neighbors
        :return: type weight (bs * nnum)
        '''
        if self.type_num == 0:
            return None
        # else:
        #     hn_t = torch.LongTensor(history_nodes).t()  # bs * nnum * size(1 or 2)
        #     rs = self.rel_eye[he_t].sum(dim=1)  # bs * rel_num
        #     lst = [he_t[i][rs[i]].unsqueeze(0) for i in range(rs.shape[0])]
        #     edge_weight = lst[0]
        #     for i in lst[1:]:
        #         torch.cat(edge_weight, i)
        #     edge_weight = self.rel_weight[edge_weight]
        #
        #     return edge_weight

    def get_edge_weight(self, history_edges, neighbor_rels):
        '''

        :param history_edges:
        :param neighbor_nodes:
        :return: (bs * nnum)
        '''

        he_t = history_edges.t() # bs * size(1 or 2)
        rs = self.rel_eye[he_t].sum(dim=1).squeeze() # bs * rel_num
        edge_weight = self.rel_weight[neighbor_rels] + (self.rel_weight * rs)[neighbor_rels]

        return edge_weight


class DistanceSearchSingleMove(nn.Module):
    def __init__(self, graph, k, embedding_dim, search_times, strategy=None):
        super(DistanceSearchSingleMove, self).__init__()
        self.graph = graph
        # get node embedding to tensor
        node_num = graph["node_embedding"].num_embeddings
        emb = graph["node_embedding"](torch.LongTensor([i for i in range(node_num)]))

        typ = graph["type_info"](torch.LongTensor([i for i in range(node_num)]))

        self.node_embedding = torch.nn.Parameter(torch.cat((
            emb, torch.zeros(embedding_dim).unsqueeze(0)), 0), requires_grad=False)
        self.node_type = torch.nn.Parameter(typ, requires_grad=False)
        self.node_neighbors = torch.nn.Parameter(torch.LongTensor(
            [graph["neighbor"][i][0] for i in graph["neighbor"]]), requires_grad=False)
        self.rel_neighbors = torch.nn.Parameter(torch.LongTensor(
            [graph["neighbor"][i][1] for i in graph["neighbor"]]), requires_grad=False)
        self.node_num = graph["node_num"]
        self.rel_num = graph["rel_num"]
        self.type_num = graph["type_num"]
        # self.node_type = graph["node_type"]
        self.dim = embedding_dim
        self.k = k
        self.st = search_times

        self.node_weight = torch.nn.Parameter(torch.Tensor(self.node_num+1).uniform_(0, 1), requires_grad=True)
        self.rel_weight = torch.nn.Parameter(torch.Tensor(self.rel_num+1).uniform_(0, 1), requires_grad=True)
        self.type_weight = torch.nn.Parameter(torch.Tensor(self.type_num+1).uniform_(0, 1), requires_grad=True)
        self.rel_eye = torch.nn.Parameter(torch.eye(self.rel_num+1), requires_grad=False)

    def forward(self, queries, p=False):
        # find e3
        e1s = queries["e1s"]
        e2s = queries["e2s"]
        e3s = queries["e3s"]
        r1s = queries["r1s"]
        r2s = queries["r2s"]
        e2_embeddings = self.node_embedding[e2s]

        current_embeddings = e2_embeddings
        e2_neighbors = self.node_neighbors[e2s]
        e2_neighbors_type_weight = self.get_type_weight([e1s, e2s], e2_neighbors)
        e2_neighbors_edge_weight = self.get_edge_weight(r1s.unsqueeze(1), self.rel_neighbors[e2s])
        e2_neighbor_weight = self.node_weight[e2_neighbors] + e2_neighbors_edge_weight
        if e2_neighbors_type_weight:
            e2_neighbor_weight += e2_neighbors_type_weight
        e2_neighbor_weight = F.softmax(e2_neighbor_weight)


        # distance
        e2_e3_dist = self.euclid_dist(self.node_embedding[e2s], self.node_embedding[e3s])
        e2neighbor_e3_dist = self.euclid_dist(self.node_embedding[e2_neighbors], self.node_embedding[e3s].unsqueeze(1))
        neighbor_sign = torch.sign(e2_e3_dist.unsqueeze(-1) - e2neighbor_e3_dist)
        current_neighbor_vec = self.node_embedding[e2_neighbors] - current_embeddings.unsqueeze(1)
        current_neighbor_vec = current_neighbor_vec * e2_neighbor_weight.unsqueeze(-1)
        moves = current_neighbor_vec * neighbor_sign.unsqueeze(-1)
        moves = moves.sum(1)
        current_embeddings = current_embeddings + moves

        losses = self.euclid_dist(current_embeddings, self.node_embedding[e3s])

        loss = losses.mean()
        if p:
            print("e2_neighbor_weight:", e2_neighbor_weight)
            print("moves:", moves)
            print("losses:", losses)
            print("loss:", losses.mean())
        return loss


    def euclid_dist(self, a, b):
        return torch.sqrt(torch.sum((a-b) ** 2, dim=-1))

    def get_type_weight(self, history_nodes, neighbor_nodes):
        nt = self.node_type[neighbor_nodes]
        type_weight = self.type_weight[nt]
        return type_weight


    def get_edge_weight(self, history_edges, neighbor_rels):
        '''

        :param history_edges:
        :param neighbor_nodes:
        :return: (bs * nnum)
        '''

        he_t = history_edges.t() # bs * size(1 or 2)
        rs = self.rel_eye[he_t].sum(dim=1).squeeze() # bs * rel_num
        edge_weight = self.rel_weight[neighbor_rels] + (self.rel_weight * rs)[neighbor_rels]

        return edge_weight