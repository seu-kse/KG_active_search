import random
import numpy as np
import sys
from tqdm import tqdm
import torch

def update_loss(loss, losses, ema_loss, ema_alpha=0.01):
    losses.append(loss)
    if ema_loss is None:
        ema_loss = loss
    else:
        ema_loss = (1 - ema_alpha) * ema_loss + ema_alpha * loss
    return losses, ema_loss

def run_train(model, optimizer, train_queries, val_queries, test_queries, logger, batch_size,
              epochs, log_every, use_cuda):
    losses = []
    ema_loss = None
    num_batch = len(train_queries) // batch_size
    model.train()
    for e in range(epochs):
        total_loss = 0
        with tqdm(total=num_batch, desc='epoch {}'.format(e+1)) as pbar:
            for i in range(num_batch):

                batch = train_queries[i*batch_size:(i+1)*batch_size]
                standard_queries = {}
                standard_queries["e1s"] = torch.LongTensor([int(q["e1"]) for q in batch])
                standard_queries["e2s"] = torch.LongTensor([int(q["e2"]) for q in batch])
                standard_queries["e3s"] = torch.LongTensor([int(q["e3"]) for q in batch])
                standard_queries["r1s"] = torch.LongTensor([int(q["r1"]) for q in batch])
                standard_queries["r2s"] = torch.LongTensor([int(q["r2"]) for q in batch])
                if use_cuda:
                    for par in standard_queries:
                        standard_queries[par].cuda()
                if i == 5:
                    loss = model.forward(standard_queries, p=True)
                else:
                    loss = model.forward(standard_queries)
                # losses, ema_loss = update_loss(loss.item(), losses, ema_loss)
                total_loss += loss.item()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (i+1e-5))})
                loss.backward()
                optimizer.step()
                model.zero_grad()
                if i % log_every == 0:
                    # logger.info("Epoch:{:d} Iter: {:d}; ema_loss: {:f}".format(e, i, ema_loss))
                    print("node_weight:", model.node_weight)
                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss / (i + 1e-5))})
                pbar.update(1)

        torch.save(model, "./ckpt/FB16K237_200k_epoch{}.ckpt".format(e))

        # run eval
        num_eval_batch = len(val_queries) // batch_size
        all_loss = 0
        for i in range(num_eval_batch):
            standard_queries["e1s"] = torch.LongTensor([int(q["e1"]) for q in batch])
            standard_queries["e2s"] = torch.LongTensor([int(q["e2"]) for q in batch])
            standard_queries["e3s"] = torch.LongTensor([int(q["e3"]) for q in batch])
            standard_queries["r1s"] = torch.LongTensor([int(q["r1"]) for q in batch])
            standard_queries["r2s"] = torch.LongTensor([int(q["r2"]) for q in batch])
            if use_cuda:
                for par in standard_queries:
                    standard_queries[par].cuda()
            loss = model.forward(standard_queries)
            all_loss += loss.item()
        eval_result = run_eval(model, val_queries, batch_size)
        logger.info("Epoch:{:d} val result: {:f}".format(e, eval_result))

    logger.info("finish training.")



def run_eval(model, queries, batch_size):
    model.eval()
    num_batch = len(queries) // (batch_size)
    all_loss = 0
    for i in range(num_batch):
        batch = queries[i*batch_size : (i+1)*batch_size]
        test_queries = {}
        test_queries["e1s"] = torch.LongTensor([int(q["e1"]) for q in batch])
        test_queries["e2s"] = torch.LongTensor([int(q["e2"]) for q in batch])
        test_queries["e3s"] = torch.LongTensor([int(q["e3"]) for q in batch])
        test_queries["r1s"] = torch.LongTensor([int(q["r1"]) for q in batch])
        test_queries["r2s"] = torch.LongTensor([int(q["r2"]) for q in batch])
        for par in test_queries:
            test_queries[par].cuda()
        all_loss += model.forward(test_queries)

    return all_loss