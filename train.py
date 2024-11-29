# from __future__ import division, print_function

import argparse
import json
import os
import random
import time
import log_help as log
import logging
import torch
import torch.nn as nn
from sklearn import metrics
import deepdish as dd
from tqdm import trange
from models_pytorch import TGCN
from utils import *


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--no_sparse', action='store_true')
    parser.add_argument("--load_ckpt", action='store_true')
    parser.add_argument('--featureless', action='store_true')
    parser.add_argument("--save_path", type=str, default='./saved_model', help="the path of saved model")
    parser.add_argument('--dataset', type=str, default='post', help='dataset name, default to post')
    parser.add_argument('--model', type=str, default='gcn', help='model name, default to gcn')
    parser.add_argument('--lr', '--learning_rate', default=0.0002, type=float)   # 0.002/0.0002
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--hidden", default=400, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)   # 0.5/0.3/0.1
    parser.add_argument("--weight_decay", default=0.000001, type=float)
    parser.add_argument("--early_stop", default=2000, type=int)
    parser.add_argument("--num_graph", default=3, type=int)
    parser.add_argument("--save_log", default='./record_lg/', type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args(args)


def save_model(model, optimizer, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}
    torch.save(save_dict, os.path.join(args.save_path, 'model.bin'))


def train(args, features, train_label, train_mask, val_label, val_mask, test_label, test_mask, model, indice_list, weight_list,logging):
    cost_valid = []
    acc_valid = []
    max_acc = 0.0
    min_cost = 10.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fct = nn.CrossEntropyLoss()

    best_FS = 0
    for epoch in range(args.epochs):
        model.train()
        t = time.time()
        outs = model(features, indice_list, weight_list, 1-args.dropout)
        pre_loss = loss_function(outs, train_label,train_mask, expt_type=5, scale=2)
        train_pred = torch.argmax(outs, dim=-1)
        ce_loss = (pre_loss * train_mask/train_mask.mean()).mean()
        train_acc = ((train_pred == train_label).float() * train_mask/train_mask.mean()).mean()
        loss = ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        valid_cost, valid_acc, pred, labels, duration,M = evaluate(args,features, val_label, val_mask, model, indice_list, weight_list)
        if valid_acc > best_FS and epoch > 10:
            best_FS = valid_acc
            save_model(model, optimizer, args)
            min_cost = cost_valid[-1]
            logging.info('model saved')

        test_cost, test_acc, pred, labels, test_duration,_ = evaluate(args,features, test_label, test_mask, model, indice_list, weight_list)
        model.train()
        cost_valid.append(valid_cost)
        acc_valid.append(valid_acc)

        logging.info(f"Epoch: {epoch + 1:04d} "
                    f"train_loss={loss.item():.5f} "
                    f"train_acc={train_acc.item():.5f} "
                    f"val_loss={valid_cost:.5f} "
                    f"val_acc={valid_acc:.5f} "
                    f"test_loss={test_cost:.5f} "
                    f"test_acc={test_acc:.5f} "
                    f"time={time.time() - t:.5f}")

        if epoch > args.early_stop and cost_valid[-1] > np.mean(cost_valid[-(args.early_stop + 1):-1]):
            print("Early stopping...")
            break


def evaluate(args, features, label, mask, model, indice_list, weight_list):
    model.eval()
    t_test = time.time()
    with torch.no_grad():
        outs = model(features, indice_list, weight_list, 1)
        pre_loss = loss_function(outs, label,mask, expt_type=5, scale=2)
        pred = torch.argmax(outs, dim=-1)
        ce_loss = (pre_loss * mask/mask.mean()).mean()
        loss = ce_loss
        acc = ((pred == label).float() * mask/mask.mean()).mean()
        M = gr_metrics(pred, label, mask)
    return loss.item(), acc.item(), pred.cpu().numpy(), label.cpu().numpy(), (time.time() - t_test),M

def load_ckpt(model):
    model_dict = model.state_dict()
    model.load_state_dict(model_dict)

def get_edge_tensor(adj):
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    data = torch.tensor(adj.data, dtype=torch.float)
    indice = torch.stack((row,col),dim=0)
    return indice, data


def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def main(args):


    ids_log = log.create_log_id(args.save_log)
    log.logging_config(folder=args.save_log, name='log{:d}'.format(ids_log), no_console=False)
    logging.info(f'time: {time.asctime(time.localtime(time.time()))}')
    logging.info(args)
        
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    f_file = os.sep.join(['..', 'data_tgcn', args.dataset, 'build_train'])
    if torch.cuda.is_available():
        device = 'cuda'
    set_seed(args)
    adj, adj1, adj2, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels,vocab = load_corpus_torch(args.dataset, device)
    adj = adj.tocoo()
    adj1 = adj1.tocoo()
    adj2 = adj2.tocoo()
    support_mix = [adj, adj1, adj2]
    indice_list, weight_list = [] , []
    for adjacency in support_mix:
        ind, dat = get_edge_tensor(adjacency)
        indice_list.append(ind.to(device))
        weight_list.append(dat.to(device))
        
    in_dim = adj.shape[0]
    model = TGCN(in_dim=in_dim, hidden_dim=args.hidden, out_dim=5,
                    num_graphs=args.num_graph, dropout=args.dropout, n_layers=args.layers, bias=False, featureless=args.featureless)
    features = torch.tensor(list(range(in_dim)), dtype=torch.long).to(device)
    

    model.to(device)
    
    if args.do_train:
        logging.info("Start training...")
        train(args, features, y_train, train_mask, y_val, val_mask, y_test, test_mask, model, indice_list, weight_list,logging)

    if args.do_valid:
        # FLAGS.dropout = 1.0
        save_dict = torch.load(os.path.join(args.save_path, 'model.bin'))
        if args.load_ckpt:
            load_ckpt(model)
        else:
            model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        val_cost, val_acc, pred, labels, val_duration,M = evaluate(args,
            features, y_val, val_mask, model, indice_list, weight_list)

        logging.info(f"Val set results:: cost={val_cost:.5f}, accuracy={val_acc:.5f}, time={val_duration:.5f}")

        val_pred = []
        val_labels = []
        print(len(val_mask))
        for i in range(len(val_mask)):
            if val_mask[i] == 1:
                val_pred.append(pred[i])
                val_labels.append(labels[i])

        logging.info("Val Precision, Recall and F1-Score...")
        logging.info(metrics.classification_report(val_labels, val_pred, digits=4))
        logging.info("Macro average Val Precision, Recall and F1-Score...")
        logging.info(metrics.precision_recall_fscore_support(val_labels, val_pred, average='macro'))
        logging.info("Micro average Val Precision, Recall and F1-Score...")
        logging.info(metrics.precision_recall_fscore_support(val_labels, val_pred, average='micro'))
        logging.info(f"GP: {M[0]}, GR: {M[1]}, FS: {M[2]}")


    if args.do_test:
        save_dict = torch.load(os.path.join(args.save_path, 'model.bin'))
        if args.load_ckpt:
            load_ckpt(model)
        else:
            model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        
        test_cost, test_acc, pred, labels, test_duration,M = evaluate(args,
            features, y_test, test_mask, model, indice_list, weight_list)        
        logging.info(f"Test set results: cost={test_cost:.5f}, accuracy={test_acc:.5f}, time={test_duration:.5f}")


        test_pred = []
        test_labels = []
        print(len(test_mask))
        for i in range(len(test_mask)):
            if test_mask[i] == 1:
                test_pred.append(pred[i])
                test_labels.append(labels[i])

        logging.info("Test Precision, Recall and F1-Score...")
        logging.info(metrics.classification_report(test_labels, test_pred, digits=4))
        logging.info("Macro average Test Precision, Recall and F1-Score...")
        logging.info(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
        logging.info("Micro average Test Precision, Recall and F1-Score...")
        logging.info(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))
        logging.info(f"GP: {M[0]}, GR: {M[1]}, FS: {M[2]}")
if __name__ == '__main__':
    main(parse_args())
