import numpy as np
import dgl
import os
import time
import random
import glob
import argparse, json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from tensorboardX import SummaryWriter
from tqdm import tqdm


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
from data.data import LoadData  # import dataset

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
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
    t_avg_test_acc, t_avg_train_acc, t_avg_convergence_epochs = [], [], []
    s_avg_test_acc, s_avg_train_acc, s_avg_convergence_epochs = [], [], []

    t0 = time.time()
    per_epoch_time = []

    dataset = LoadData(DATASET_NAME)

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    # Read Train Val Test Data
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    print("With Size: {}====={}====={}".format(len(trainset), len(valset), len(testset)))
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    # At any point you can hit Ctrl + C to break out of training early.
    # try:
    # for split_number in range(10):
    split_number = random.randint(0, 9)
    t0_split = time.time()
    log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    print("RUN NUMBER: ", split_number)

    print("Number of Classes: ", net_params['n_classes'])

    # Init Target Model
    t_model = gnn_model(MODEL_NAME, net_params)
    t_model = t_model.to(device)
    t_optimizer = optim.Adam(t_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    t_scheduler = optim.lr_scheduler.ReduceLROnPlateau(t_optimizer, mode='min',
                                                       factor=params['lr_reduce_factor'],
                                                       patience=params['lr_schedule_patience'],
                                                       verbose=True)
    print("Target Model:\n{}".format(t_model))
    # Init Shadow Model
    s_model = gnn_model(MODEL_NAME, net_params)
    s_model = s_model.to(device)
    s_optimizer = optim.Adam(s_model.parameters(), lr=params['init_lr'],
                             weight_decay=params['weight_decay'])
    s_scheduler = optim.lr_scheduler.ReduceLROnPlateau(s_optimizer, mode='min',
                                                       factor=params['lr_reduce_factor'],
                                                       patience=params['lr_schedule_patience'],
                                                       verbose=True)
    print("Shadow Model:\n{}".format(s_model))
    t_epoch_train_losses, t_epoch_val_losses, t_epoch_train_accs, t_epoch_val_accs = [], [], [], []
    s_epoch_train_losses, s_epoch_val_losses, s_epoch_train_accs, s_epoch_val_accs = [], [], [], []


    # Set train, val and test data size
    train_size = params['train_size']
    val_size = params['val_size']
    test_size = params['test_size']
    # Load Train, Val and Test Dataset
    trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
    print("Size of Trainset:{}, Valset:{}, Testset:{}".format(len(trainset), len(valset), len(testset)))
    print("Needed Train size:{} , Val Size:{} and Test Size：{}".format(train_size, val_size, test_size))

    # In order to flexible manage the size of Train, Val, Test data,
    # Here we resize the size
    dataset_all = trainset + testset + valset
    trainset, valset, testset = random_split(dataset_all,
                                                      [len(dataset_all) - val_size * 2 - test_size * 2, val_size * 2,
                                                       test_size * 2])
    print("Adjust Size:", len(trainset), len(valset), len(testset))

    # Split Data into Target and Shadow
    target_train_set, shadow_train_set = random_split(trainset,[len(trainset) // 2, len(trainset) - len(trainset) // 2])
    target_val_set, shadow_val_set = random_split(valset, [len(valset) // 2, len(valset) - len(valset) // 2])
    target_test_set, shadow_test_set = random_split(testset, [len(testset) // 2, len(testset) - len(testset) // 2])

    print("Target train set with size:{} and Shadow train set with size:{}".format(len(target_train_set),
                                                                                   len(shadow_train_set)))
    print("Target Val set with size:{} and Shadow Val set with size:{}".format(len(target_val_set),
                                                                               len(shadow_val_set)))
    print("Target Test set with size:{} and Shadow Test set with size:{}".format(len(target_test_set),
                                                                                 len(shadow_test_set)))

     # sample defined size of graphs
    selected_T_train_set, _ = random_split(target_train_set, [train_size, len(target_train_set) - train_size])
    selected_T_val_set, _ = random_split(target_val_set, [val_size, len(target_val_set) - val_size])
    selected_T_test_set, _ = random_split(target_test_set, [test_size, len(target_test_set) - test_size])
    print('Selected Training Size:{}, Validation Size: {} and Testing Size:{}'.format(len(selected_T_train_set),
                                                                                      len(selected_T_val_set),
                                                                                      len(selected_T_test_set)))

    selected_S_train_set, _ = random_split(shadow_train_set, [train_size, len(shadow_train_set) - train_size])
    selected_S_val_set, _ = random_split(shadow_val_set, [val_size, len(shadow_val_set) - val_size])
    selected_S_test_set, _ = random_split(shadow_test_set, [test_size, len(shadow_test_set) - test_size])
    print('Selected Shadow Size:{}, Validation Size: {} and Testing Size:{}'.format(len(selected_S_train_set),
                                                                                    len(selected_S_val_set),
                                                                                    len(selected_S_test_set)))

    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False

    # import train functions for all other GCNs
    from train.train_TUs_graph_classification import train_epoch_sparse as train_epoch, \
        evaluate_network_sparse as evaluate_network

    # Load data
    target_train_loader = DataLoader(selected_T_train_set, batch_size=params['batch_size'], shuffle=True,
                                     drop_last=drop_last,
                                     collate_fn=dataset.collate)
    target_val_loader = DataLoader(selected_T_val_set, batch_size=params['batch_size'], shuffle=False,
                                   drop_last=drop_last, collate_fn=dataset.collate)
    target_test_loader = DataLoader(selected_T_test_set, batch_size=params['batch_size'], shuffle=False,
                                    drop_last=drop_last, collate_fn=dataset.collate)

    shadow_train_loader = DataLoader(selected_S_train_set, batch_size=params['batch_size'], shuffle=True,
                                     drop_last=drop_last, collate_fn=dataset.collate)
    shadow_val_loader = DataLoader(selected_S_val_set, batch_size=params['batch_size'], shuffle=False,
                                   drop_last=drop_last, collate_fn=dataset.collate)
    shadow_test_loader = DataLoader(selected_S_test_set, batch_size=params['batch_size'], shuffle=False,
                                    drop_last=drop_last, collate_fn=dataset.collate)

    print('Start Training Target Model...')
    print("target_train_loader:", len(target_train_loader))
    t_ckpt_dir, s_ckpt_dir = '', ''
    try:
        with tqdm(range(params['epochs'])) as t1:
            for epoch in t1:

                t1.set_description('Epoch %d' % epoch)

                start = time.time()
                # else:   # for all other models common train function
                t_epoch_train_loss, t_epoch_train_acc, t_optimizer = train_epoch(t_model,
                                                                                 t_optimizer,
                                                                                 device,
                                                                                 target_train_loader, epoch)

                t_epoch_val_loss, t_epoch_val_acc = evaluate_network(t_model, device, target_val_loader, epoch)
                _, t_epoch_test_acc = evaluate_network(t_model, device, target_test_loader, epoch)

                t_epoch_train_losses.append(t_epoch_train_loss)
                t_epoch_val_losses.append(t_epoch_val_loss)
                t_epoch_train_accs.append(t_epoch_train_acc)
                t_epoch_val_accs.append(t_epoch_val_acc)

                writer.add_scalar('train/_loss', t_epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', t_epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', t_epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', t_epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', t_epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', t_optimizer.param_groups[0]['lr'], epoch)

                _, t_epoch_test_acc = evaluate_network(t_model, device, target_test_loader, epoch)
                t1.set_postfix(time=time.time() - start, lr=t_optimizer.param_groups[0]['lr'],
                               train_loss=t_epoch_train_loss, val_loss=t_epoch_val_loss,
                               train_acc=t_epoch_train_acc, val_acc=t_epoch_val_acc,
                               test_acc=t_epoch_test_acc)

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint
                t_ckpt_dir = os.path.join(root_ckpt_dir, "T_RUN_")
                if not os.path.exists(t_ckpt_dir):
                    os.makedirs(t_ckpt_dir)
                torch.save(t_model.state_dict(), '{}.pkl'.format(t_ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(t_ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)
                '''
                  Update Params
                '''
                t_scheduler.step(t_epoch_val_loss)

                if t_optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0_split > params[
                    'max_time'] * 3600:  # Dividing max_time by 10, since there are 10 runs in TUs
                    print('-' * 89)
                    print("Max_time for one train experiment elapsed {:.3f} hours, so stopping".format(
                        params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Target Model Training --- Exiting from training early because of KeyboardInterrupt')

    try:
        # Start training Shadow Model
        with tqdm(range(params['epochs'])) as t2:
            for epoch in t2:
                t2.set_description('Epoch %d' % epoch)
                start = time.time()
                s_epoch_train_loss, s_epoch_train_acc, s_optimizer = train_epoch(s_model, s_optimizer, device,
                                                                                 shadow_train_loader, epoch)

                s_epoch_val_loss, s_epoch_val_acc = evaluate_network(s_model, device, shadow_val_loader, epoch)
                _, s_epoch_test_acc = evaluate_network(s_model, device, shadow_test_loader, epoch)

                s_epoch_train_losses.append(s_epoch_train_loss)
                s_epoch_val_losses.append(s_epoch_val_loss)
                s_epoch_train_accs.append(s_epoch_train_acc)
                s_epoch_val_accs.append(s_epoch_val_acc)

                writer.add_scalar('train/_loss', s_epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', s_epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', s_epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', s_epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', s_epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', s_optimizer.param_groups[0]['lr'], epoch)

                _, s_epoch_test_acc = evaluate_network(s_model, device, shadow_test_loader, epoch)
                t2.set_postfix(time=time.time() - start, lr=s_optimizer.param_groups[0]['lr'],
                               train_loss=s_epoch_train_loss, val_loss=s_epoch_val_loss,
                               train_acc=s_epoch_train_acc, val_acc=s_epoch_val_acc,
                               test_acc=s_epoch_test_acc)

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint
                s_ckpt_dir = os.path.join(root_ckpt_dir, "S_RUN_")
                if not os.path.exists(s_ckpt_dir):
                    os.makedirs(s_ckpt_dir)
                torch.save(s_model.state_dict(), '{}.pkl'.format(s_ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(s_ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                s_scheduler.step(s_epoch_val_loss)

                if s_optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0_split > params[
                    'max_time'] * 3600:  # Dividing max_time by 10, since there are 10 runs in TUs
                    print('-' * 89)
                    print("Max_time for one train experiment elapsed {:.3f} hours, so stopping".format(
                        params['max_time']))
                    break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Shadow Model Training --- Exiting from training early because of KeyboardInterrupt')

    print("=================Evaluate Target Model Start=================")
    _, t_test_acc = evaluate_network(t_model, device, target_test_loader, '0|T|' + t_ckpt_dir)
    _, t_train_acc = evaluate_network(t_model, device, target_train_loader, '1|T|' + t_ckpt_dir)
    t_avg_test_acc.append(t_test_acc)
    t_avg_train_acc.append(t_train_acc)
    t_avg_convergence_epochs.append(epoch)
    print("Target Test Accuracy [LAST EPOCH]: {:.4f}".format(t_test_acc))
    print("Target Train Accuracy [LAST EPOCH]: {:.4f}".format(t_train_acc))
    print("Target Convergence Time (Epochs): {:.4f}".format(epoch))
    _, s_test_acc = evaluate_network(s_model, device, shadow_test_loader, '0|S|' + s_ckpt_dir)
    _, s_train_acc = evaluate_network(s_model, device, shadow_train_loader, '1|S|' + s_ckpt_dir)
    s_avg_test_acc.append(s_test_acc)
    s_avg_train_acc.append(s_train_acc)
    s_avg_convergence_epochs.append(epoch)
    print("Shadow Test Accuracy [LAST EPOCH]: {:.4f}".format(s_test_acc))
    print("Shadow Train Accuracy [LAST EPOCH]: {:.4f}".format(s_train_acc))
    print("Shadow Convergence Time (Epochs): {:.4f}".format(epoch))

    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time() - t0) / 3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    print("AVG CONVERGENCE Time (Epochs): {:.4f}".format(np.mean(np.array(s_avg_convergence_epochs))))
    # Final test accuracy value averaged over 10-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}"""
          .format(np.mean(np.array(s_avg_test_acc)) * 100, np.std(s_avg_test_acc) * 100))
    print("\nAll splits Test Accuracies:\n", s_avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}"""
          .format(np.mean(np.array(s_avg_train_acc)) * 100, np.std(s_avg_train_acc) * 100))
    print("\nAll splits Train Accuracies:\n", s_avg_train_acc)

    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL TARGET RESULTS\nTARGET TEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\nTARGET TRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
    FINAL SHADOW RESULTS\nSHADOW TEST ACCURACY averaged: {:.4f} with s.d. {:.4f}\nSHADOW TRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}\n\n
    Average Convergence Time (Epochs): {:.4f} with s.d. {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\nAll Splits Test Accuracies: {}""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, s_model, net_params['total_param'],
                        np.mean(np.array(t_avg_test_acc)) * 100, np.std(t_avg_test_acc) * 100,
                        np.mean(np.array(t_avg_train_acc)) * 100, np.std(t_avg_train_acc) * 100,
                        np.mean(np.array(s_avg_test_acc)) * 100, np.std(s_avg_test_acc) * 100,
                        np.mean(np.array(s_avg_train_acc)) * 100, np.std(s_avg_train_acc) * 100,
                        np.mean(t_avg_convergence_epochs), np.std(t_avg_convergence_epochs),
                        (time.time() - t0) / 3600, np.mean(per_epoch_time), t_avg_test_acc))


def main():
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
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
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred == 'True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False

    # TUs
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = len(np.unique(dataset.all.graph_labels))
    net_params['n_classes'] = num_classes

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        max_num_node = max(num_nodes)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

    if MODEL_NAME == 'RingGNN':
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)


main()
