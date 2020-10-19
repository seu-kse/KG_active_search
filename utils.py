import logging
import numpy as np

def setup_logging(log_file, console=True):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')
    if console:
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging


def get_edge_weight(model, history_edge, neighbor_rels):
    rel_weight = model.rel_weight.detach().numpy()
    weight_result = [0 for _ in neighbor_rels]
    for i in range(len(neighbor_rels)):
        weight_result[i] = rel_weight[neighbor_rels[i]]
        if neighbor_rels[i] == history_edge:
            weight_result[i] *= 2
    return weight_result

def get_node_weight(model, neighbors):
    node_weight = model.node_weight.detach().numpy()
    weight_result = node_weight[neighbors]
    return weight_result

def softmax(array):
    '''

    :param array: np.array
    :return:
    '''
    return np.exp(array) / sum(np.exp(array))

def euclid_dist(a, b):
    '''

    :param a: np.array
    :param b:
    :return:
    '''
    return np.linalg.norm(b-a)