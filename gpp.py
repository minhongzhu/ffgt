"""
    IMPORTING LIBS
"""

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import namedtuple

"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from loader.dataset import GraphDataModule
from model.grpe.network import Grpe_FFGT
from model.vanilla.network import Vanilla_FFGT
from utils.lr import PolynomialDecayLR
from loss.ap import AP
from loss.mae import MAE
from loss.acc import accuracy_SBM
from loss.weighted_cross_entropy import weighted_cross_entropy

from ogb.graphproppred import Evaluator



"""
    GPU Setup
"""

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    Training epoch
"""
def train(
    model, 
    train_loader,
    config,
    optimizer,
    lr_scheduler,
    device=None
):
    model.train()
    loss_fn = config.loss_fn
    
    losses = []

    for iter, (batch, extra_batch) in enumerate(train_loader):
        # 将数据放入GPU中
        batch = batch.to(device)
        extra_batch = extra_batch.to(device)
        optimizer.zero_grad()

        y_hat = model(batch, extra_batch)
        y_gt = batch.y.view(y_hat.shape).float()

        if not torch.isnan(y_hat).any():

            mask = ~torch.isnan(y_gt)

            if config.name in ['PATTERN']:
                loss, _ = loss_fn(y_hat[mask], y_gt[mask])
            else:
                loss = loss_fn(y_hat[mask], y_gt[mask])
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()
            
            losses.append(loss.detach())

    return torch.stack(losses).mean().item()

"""
    valid epoch
"""
@torch.no_grad()
def evaluate(model, test_loader, config, device=None):
    model.eval()
    evaluator = config.evaluator

    y_pred = []
    y_true = []
    
    for iter, (batch, extra_batch) in enumerate(test_loader):
        batch = batch.to(device)
        extra_batch = extra_batch.to(device)
           
        y_hat = model(batch, extra_batch)
        y_gt = batch.y.view(y_hat.shape).float()

        if not torch.isnan(y_hat).any():
            y_pred.append(y_hat)
            y_true.append(y_gt)
        
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
        
    if isinstance(evaluator, accuracy_SBM):
        y_true = y_true.long().squeeze(-1)
        y_pred = y_pred.squeeze(-1)
    
    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return result

"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, data_module, params, net_params, dirs):

    t0 = time.time()
    per_epoch_time = []
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
        

    if data_module.dataset_name.startswith('PATTERN'):
        DATASET_NAME = 'PATTERN'
    else:
        DATASET_NAME = data_module.dataset_name
    
    if MODEL_NAME == 'grpe_ffgt':
        model = Grpe_FFGT(
            num_task=net_params['n_task'],
            num_layer=net_params['n_layers'],
            d_model=net_params['hidden_dim'],
            d_attn=net_params['attn_dim'],
            nhead=net_params['n_heads'],
            khead=net_params['k_heads'],
            dim_feedforward=net_params['ffn_dim'],
            dropout=net_params['dropout_rate'],
            attention_dropout=net_params['attention_dropout_rate'],
            max_hop=net_params['max_hop'],
            num_node_type=net_params['num_node_type'],
            num_edge_type=net_params['num_edge_type'],
            use_independent_token=net_params['use_independent_token'],
            num_last_mlp=net_params['num_last_mlp'],
            task=net_params['task_type']
        )
    elif MODEL_NAME == 'vanilla_ffgt':
        model = Vanilla_FFGT(
            num_task=net_params['n_task'],
            num_layer=net_params['n_layers'],
            d_model=net_params['hidden_dim'],
            d_pe=net_params['pe_dim'],
            d_attn=net_params['attn_dim'],
            nhead=net_params['n_heads'],
            khead=net_params['k_heads'],
            dim_feedforward=net_params['ffn_dim'],
            dropout=net_params['dropout_rate'],
            attention_dropout=net_params['attention_dropout_rate'],
            num_node_type=net_params['num_node_type'],
            num_edge_type=net_params['num_edge_type'],
            use_independent_token=net_params['use_independent_token'],
            num_last_mlp=net_params['num_last_mlp'],
            task=net_params['task_type'],
            add_edge=net_params['add_edge']
        )


    model = model.to(device)

    # view parameters
    from utils.metrics import view_model_param
    net_params['total_param'] = view_model_param(model, MODEL_NAME)

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""
                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)
    
    # prepare loader
    print("Training Graphs: ", len(data_module.dataset_train))
    print("Validation Graphs: ", len(data_module.dataset_val))
    print("Test Graphs: ", len(data_module.dataset_test))

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # prepare for evaluator
    ExpConfig = namedtuple("ExpConfig", ["name","evaluator", "metric", "loss_fn"])
    experiments = {
        'pcba':ExpConfig(
            'pdba',
            Evaluator(name="ogbg-molpcba"), 
            "ap", 
            F.binary_cross_entropy_with_logits
        ),
        'pep-func':ExpConfig(
            'pep-func',
            AP(name="peptide-functional", num_tasks=10),
            "ap",
            F.binary_cross_entropy_with_logits
        ),
        'pep-struc':ExpConfig(
            'pep-struc',
            MAE(name="peptide-structural", num_tasks=11),
            "mae",
            F.l1_loss
        ),
        'ZINC': ExpConfig(
            'ZINC',
            MAE(name="zinc", num_tasks=1),
            "mae",
            F.l1_loss
        ),
        'PATTERN': ExpConfig(
            'PATTERN',
            accuracy_SBM(name='PATTERN', num_tasks=1),
            'acc_SBM',
            weighted_cross_entropy
        )
    }
    
    if params['dataset'].startswith('PATTERN'):
        config = experiments['PATTERN']
    else:
        config = experiments[params['dataset']]

    # prepare for schedular & optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params['peak_lr'], weight_decay=params['weight_decay'])
    scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=params['warmup_epochs'] * len(train_loader),
        tot_updates=params['tot_epochs'] * len(train_loader),
        lr=params['peak_lr'],
        end_lr=params['end_lr'],
        power=1.0
    )

    # training start here
    # At any point you can hit Ctrl + C to break out of training early.
    max_val_result, max_val_best = 0.0, 0.0
    min_val_result, min_val_best = 100, 100
    val_best_epoch = 0
    print("Start training!")
    try:
        with tqdm(range(1, params['epochs'] + 1)) as t:    # tqdm:python进度条
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss = train(model, train_loader, config, optimizer, scheduler, device)
                
                epoch_val_result = evaluate(model, val_loader, config, device)
                epoch_test_result = evaluate(model, test_loader, config, device)

                # write to logs
                if DATASET_NAME in ['PATTERN']:
                    if epoch_val_result['acc_SBM'] > max_val_result:
                        max_val_result = epoch_val_result['acc_SBM']
                        max_val_best = epoch_test_result['acc_SBM']
                        val_best_epoch = epoch
                    
                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_acc_SBM', epoch_val_result['acc_SBM'], epoch)
                    writer.add_scalar('test/_acc_SBM', epoch_test_result['acc_SBM'], epoch)

                    t.set_postfix(
                        time=time.time()-start, 
                        lr=optimizer.param_groups[0]['lr'],
                        train_loss=epoch_train_loss,
                        val_acc_SBM=epoch_val_result['acc_SBM'],
                        test_acc_SBM=epoch_test_result['acc_SBM']
                    )
                elif DATASET_NAME in ['pcba', 'pep-func']:
                    if epoch_val_result['ap'] > max_val_result:
                        max_val_result = epoch_val_result['ap']
                        max_val_best = epoch_test_result['ap']
                        val_best_epoch = epoch
                    
                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_ap', epoch_val_result['ap'], epoch)
                    writer.add_scalar('test/_ap', epoch_test_result['ap'], epoch)

                    t.set_postfix(
                        time=time.time()-start, 
                        lr=optimizer.param_groups[0]['lr'],
                        train_loss=epoch_train_loss,
                        val_ap=epoch_val_result['ap'],
                        test_ap=epoch_test_result['ap']
                    )
                elif DATASET_NAME in ['pep-struc', 'ZINC', 'lsc-v2']:
                    if epoch_val_result['mae'] < min_val_result:
                        min_val_result = epoch_val_result['mae']
                        min_val_best = epoch_test_result['mae']
                        val_best_epoch = epoch

                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_mae', epoch_val_result['mae'], epoch)
                    writer.add_scalar('test/_mae', epoch_test_result['mae'], epoch)

                    t.set_postfix(
                        time=time.time()-start, 
                        lr=optimizer.param_groups[0]['lr'],
                        train_loss=epoch_train_loss,
                        val_mae=epoch_val_result['mae'],
                        test_mae=epoch_test_result['mae']
                    )

                per_epoch_time.append(time.time()-start)
                             
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    # Saving model parameters
    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))
    
    # evaluate result
    if DATASET_NAME in ['PATTERN']:
        test_result = evaluate(model, test_loader, config, device)['acc_SBM']
        val_result = evaluate(model, val_loader, config, device)['acc_SBM']
        val_best_test = max_val_best
    elif DATASET_NAME in ['pcba', 'pep-func']:
        test_result = evaluate(model, test_loader, config, device)['ap']
        val_result = evaluate(model, val_loader, config, device)['ap']
        val_best_test = max_val_best
    elif DATASET_NAME in ['pep-struc', 'ZINC', 'lsc-v2']:
        test_result = evaluate(model, test_loader, config, device)['mae']
        val_result = evaluate(model, val_loader, config, device)['mae']
        val_best_test = min_val_best

    print("Val result: {:.6f}".format(val_result))
    print("Test result: {:.6f}".format(test_result))
    print("Val Best Test result: {:.6f}".format(val_best_test))
    print("TOTAL TIME TAKEN: {:.6f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.6f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nVal : {:.6f}\nVal Best Test: {:.6f}\nTest: {:.6f}\n\nVal Best Test epoch:{}\n
    Convergence Time (Epochs): {:.6f}\nTotal Time Taken: {:.6f} hrs\nAverage Time Per Epoch: {:.6f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  val_result, val_best_test, test_result, val_best_epoch, epoch, 
                  (time.time()-t0)/3600, np.mean(per_epoch_time)))

def main():    
    """
        USER CONTROLS
    """
    # argparse模块：命令行选项、参数和子命令解析器。
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="gpu id")
    parser.add_argument('--model', help="model name")
    parser.add_argument('--dataset', help="dataset name")
    parser.add_argument('--out_dir', help="output directory")
    # parameters
    parser.add_argument('--seed', help="random seed")
    parser.add_argument('--num_workers', help="number of parallel workers")
    parser.add_argument('--epochs', help="training epochs")
    parser.add_argument('--batch_size', help="training batch size")
    parser.add_argument('--peak_lr', help="peak learning rate")
    parser.add_argument('--end_lr', help="end learning rate")
    parser.add_argument('--warmup_epochs', help="warm up epochs")
    parser.add_argument('--tot_epochs', help="total number of epochs for param update")
    parser.add_argument('--weight_decay', help="weight decay rate")
    parser.add_argument('--max_dist', help="focal length")
    parser.add_argument('--max_freq', help="maximal frequency in LapPE")
    parser.add_argument('--max_node', help="maximum node per graph")
    # net parameters
    parser.add_argument('--n_task', help="num of prediction tasks")
    parser.add_argument('--task_type', help="task level, node/link/graph")
    parser.add_argument('--n_layers', help="num of model layers")    
    parser.add_argument('--hidden_dim', help="model hidden dimension")
    parser.add_argument('--ffn_dim', help="feed forward dim")
    parser.add_argument('--pe_dim', help="output dim of pe")
    parser.add_argument('--attn_dim', help="attention dim of each heads")
    parser.add_argument('--n_heads', help="total num of heads")
    parser.add_argument('--k_heads', help="num of focal attn heads")
    parser.add_argument('--dropout_rate', help="dropout")
    parser.add_argument('--attention_dropout_rate', help="attn dropout")
    parser.add_argument('--use_independent_token', help="whether to use independent bias each layer")
    parser.add_argument('--num_last_mlp', help="num of output mlp layers")
    parser.add_argument('--max_hop', help="maximal spd in grpe")
    parser.add_argument('--num_node_type', help="num node type (atom)")
    parser.add_argument('--num_edge_type', help="num edge type (bond)")
    parser.add_argument('--add_edge', help="whether to use edge feat in vanilla")
    args = parser.parse_args()

    # use json file to save and modify parameters
    with open(args.config) as f:
        config = json.load(f)
        
    # device(gpu)
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']

    # training parameters
    params = config['params']
    params['dataset'] = config['dataset']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.num_workers is not None:
        params['num_workers'] = int(args.num_workers)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.peak_lr is not None:
        params['peak_lr'] = float(args.peak_lr)
    if args.end_lr is not None:
        params['end_lr'] = float(args.end_lr)
    if args.warmup_epochs is not None:
        params['warmup_updates'] = int(args.warmup_updates)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)    
    if args.tot_epochs is not None:
        params['tot_updates'] = int(args.tot_updates)
    if args.max_dist is not None:
        params['max_dist'] = int(args.max_dist)
    if args.max_freq is not None:
        params['max_freq'] = int(args.max_freq)
    if args.max_node is not None:
        params['max_node'] = int(args.max_node)
    if MODEL_NAME == 'grpe_ffgt':
        params['max_freq'] = None 

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.n_task is not None:
        net_params['n_task'] = int(args.n_task)
    if args.task_type is not None:
        net_params['task_type'] = str(args.task_type)
    if args.n_layers is not None:
        net_params['n_layers'] = int(args.n_layers)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.k_heads is not None:
        net_params['k_heads'] = int(args.k_heads)    
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.attn_dim is not None:
        net_params['attn_dim'] = int(args.attn_dim)    
    if args.ffn_dim is not None:
        net_params['ffn_dim'] = int(args.ffn_dim)    
    if args.dropout_rate is not None:
        net_params['dropout_rate'] = float(args.dropout_rate)
    if args.attention_dropout_rate is not None:
        net_params['attention_dropout_rate'] = float(args.attention_dropout_rate)
    if args.use_independent_token is not None:
        net_params['use_independent_token'] = bool(args.use_independent_token)
    if args.num_last_mlp is not None:
        net_params['num_last_mlp'] = int(args.num_last_mlp)
    if args.max_hop is not None:
        net_params['max_hop'] = int(args.max_hop)
    if args.num_node_type is not None:
        net_params['num_node_type'] = int(args.num_node_type)
    if args.num_edge_type is not None:
        net_params['num_edge_type'] = int(args.num_edge_type)
    if args.add_edge is not None:
        net_params['add_edge'] = bool(args.add_edge)

    # data_module 
    data_module = GraphDataModule(
        dataset_name=DATASET_NAME,
        model=MODEL_NAME,
        task=net_params['task_type'],
        batch_size=params['batch_size'],
        seed=params['seed'],
        max_dist=params['max_dist'],
        max_node=params['max_node'],
        max_freq=params['max_freq'],
        num_workers=params['num_workers']
    )

    seed = params['seed']

    root_log_dir = out_dir + 'Seed{}/'.format(seed) + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + \
                    str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'Seed{}/'.format(seed) + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + \
                    str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'Seed{}/'.format(seed) + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + \
                    str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'Seed{}/'.format(seed) + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + \
                    str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'Seed{}/results'.format(seed)):
        os.makedirs(out_dir + 'Seed{}/results'.format(seed))
        
    if not os.path.exists(out_dir + 'Seed{}/configs'.format(seed)):
        os.makedirs(out_dir + 'Seed{}/configs'.format(seed))

    train_val_pipeline(MODEL_NAME, data_module, params, net_params, dirs)


if __name__ == '__main__':
    main()
