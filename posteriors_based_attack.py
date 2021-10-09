import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
from torch.utils.data import DataLoader
import warnings
import statistics as st
from attack_models import MLP
from utils import load_pickled_data, trainData, binary_acc, testData

warnings.simplefilter("ignore")


def apply_attack(epochs, attack_base_path, target_base_path, attack_times,model):
    # Shadow Dataset Used as Attack Dataset
    S_X_train_in = load_pickled_data(attack_base_path + 'S_X_train_Label_1.pickle')
    S_y_train_in = load_pickled_data(attack_base_path + 'S_y_train_Label_1.pickle')
    S_X_train_out = load_pickled_data(attack_base_path + 'S_X_train_Label_0.pickle')
    S_y_train_out = load_pickled_data(attack_base_path + 'S_y_train_Label_0.pickle')
    print("S_X_train_in Size:{} and S_X_train_out Size:{}".format(len(S_X_train_in), len(S_X_train_out)))
    S_Label_0_num_nodes = load_pickled_data(attack_base_path + 'S_num_node_0.pickle')
    S_Label_1_num_nodes = load_pickled_data(attack_base_path + 'S_num_node_1.pickle')
    S_Label_0_num_edges = load_pickled_data(attack_base_path + 'S_num_edge_0.pickle')
    S_Label_1_num_edges = load_pickled_data(attack_base_path + 'S_num_edge_1.pickle')
    # Target Dataset used as Attack Evaluation Dataset
    T_X_train_in = load_pickled_data(target_base_path + 'T_X_train_Label_1.pickle')
    T_y_train_in = load_pickled_data(target_base_path + 'T_y_train_Label_1.pickle')
    T_X_train_out = load_pickled_data(target_base_path + 'T_X_train_Label_0.pickle')
    T_y_train_out = load_pickled_data(target_base_path + 'T_y_train_Label_0.pickle')
    print("T_X_train_in Size:{} and T_X_train_out Size:{}".format(len(T_X_train_in), len(T_X_train_out)))
    T_Label_0_num_nodes = load_pickled_data(target_base_path + 'T_num_node_0.pickle')
    T_Label_1_num_nodes = load_pickled_data(target_base_path + 'T_num_node_1.pickle')
    T_Label_0_num_edges = load_pickled_data(target_base_path + 'T_num_edge_0.pickle')
    T_Label_1_num_edges = load_pickled_data(target_base_path + 'T_num_edge_1.pickle')
    # Prepare Dataset
    X_attack = torch.FloatTensor(np.concatenate((S_X_train_in, S_X_train_out), axis=0))
    y_target = torch.FloatTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0))
    y_attack = torch.FloatTensor(np.concatenate((S_y_train_in, S_y_train_out), axis=0))
    X_target = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0))

    X_attack_nodes = torch.FloatTensor(np.concatenate((S_Label_1_num_nodes, S_Label_0_num_nodes), axis=0))
    X_attack_edges = torch.FloatTensor(np.concatenate((S_Label_1_num_edges, S_Label_0_num_edges), axis=0))
    X_target_nodes = torch.FloatTensor(np.concatenate((T_Label_1_num_nodes, T_Label_0_num_nodes), axis=0))
    X_target_edges = torch.FloatTensor(np.concatenate((T_Label_1_num_edges, T_Label_0_num_edges), axis=0))

    n_in_size = X_attack.shape[1]
    attack_precision, attack_recall, attack_fscore = [], [], []
    for attack in range(attack_times):
        # Init Attack Model
        attack_model = MLP(in_size=n_in_size, out_size=1, hidden_1=64, hidden_2=64)
        attack_criterion = nn.BCEWithLogitsLoss()
        attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)
        attack_train_data = trainData(X_attack, y_attack)
        # Prepare Attack Model Training Data
        attack_train_loader = DataLoader(dataset=attack_train_data, batch_size=64, shuffle=True)
        for i in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in attack_train_loader:
                attack_optimizer.zero_grad()
                y_pred = attack_model(X_batch)
                loss = attack_criterion(y_pred, y_batch.unsqueeze(1))
                acc = binary_acc(y_pred, y_batch.unsqueeze(1))
                loss.backward()
                attack_optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += acc.item()
            # print(f'Epoch {i + 0:03}: | Loss: {epoch_loss / len(attack_train_loader):.5f} |'
            #       f' Acc: {epoch_acc / len(attack_train_loader):.3f}')

        # Load Target Evaluation Data
        target_evaluate_data = testData(X_target)
        target_evaluate_loader = DataLoader(dataset=target_evaluate_data, batch_size=1)
        # Eval Attack Model

        y_pred_list = []
        attack_model.eval()
        # print("Attack {}.".format(attack))
        correct_node_list, correct_edge_list = [], []
        incorrect_node_list, incorrect_edge_list = [], []
        num_nodes, num_edges = [], []
        with torch.no_grad():
            for X_batch, num_node, num_edge, y in zip(target_evaluate_loader, X_target_nodes, X_target_edges, y_target):
                y_test_pred = attack_model(X_batch)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                num_nodes.append(num_node.detach().item())
                num_edges.append(num_edge.detach().item())
                if y == y_pred_tag.detach().item():
                    correct_node_list.append(num_node.detach().item())
                    correct_edge_list.append(num_edge.detach().item())
                else:
                    incorrect_node_list.append(num_node.detach().item())
                    incorrect_edge_list.append(num_edge.detach().item())
                y_pred_list.append(y_pred_tag.cpu().numpy())
            # for X_batch in target_evaluate_loader:
            #     y_test_pred = attack_model(X_batch)
            #     y_test_pred = torch.sigmoid(y_test_pred)
            #     y_pred_tag = torch.round(y_test_pred)
            #     y_pred_list.append(y_pred_tag.cpu().numpy())
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        precision, recall, fscore, support = precision_recall_fscore_support(y_target, y_pred_list, average='macro')
        attack_precision.append(precision)
        attack_recall.append(recall)
        attack_fscore.append(fscore)
        # data_array = np.hstack([np.asarray(num_nodes),np.asarray(num_edges),np.asarray(y_target),np.asarray(y_pred_list)])
        y = [y_t.detach().item() for y_t in y_target]

        writecsv = pd.DataFrame({'num_node': num_nodes, 'num_edge': num_edges, 'label': y, 'predict': y_pred_list})
        print(writecsv)
        writecsv.to_csv("data/statis/test_results_" + model + "_attack" + str(attack) + ".csv", index=False)
        print("Attack Precision:{}, Recall:{} and F-Score:{}".format(precision, recall, fscore))
    print("Attack Precision:\n\t{}".format(attack_precision))
    print("Attack Recall:\n\t{}".format(attack_recall))
    print("Attack F-Score:\n\t{}".format(attack_fscore))
    print("Average attack precision:{}, Recall:{} and F-Score:{}".format(st.mean(attack_precision),
                                                                         st.mean(attack_recall),
                                                                         st.mean(attack_fscore)))
    print("Attack precision stdev:{}, Recall stdev:{} and F-Score stdev:{}".format(st.stdev(attack_precision),
                                                                                   st.stdev(attack_recall),
                                                                                   st.stdev(attack_fscore)))


if __name__ == '__main__':
    # target_path = 'out/superpixels_graph_classification/checkpoints/GCN_CIFAR10_GPU1_21h02m43s_on_Jan_25_2021/T_RUN_/'
    # attack_path = 'out/superpixels_graph_classification/checkpoints/GCN_CIFAR10_GPU1_21h02m43s_on_Jan_25_2021/S_RUN_/'
    result_path = 'data/statis/GCN/'
    folders = os.listdir(result_path)
    sorted(folders)
    models = ['GCN', 'GIN', 'GAT', 'GatedGCN', 'MLP']
    for folder in folders:
        target_path = result_path + folder + '/T_RUN_/'
        attack_path = result_path + folder + '/S_RUN_/'
        print(target_path)
        files = os.listdir(target_path)
        model_name = [f for f in files if f.startswith('epoch')][0]
        epoch = int(model_name.split('.')[0].split('_')[1]) + 1
        print(epoch, target_path)
        apply_attack(epochs=300, attack_base_path=attack_path, target_base_path=target_path, attack_times=10, model=folder)
