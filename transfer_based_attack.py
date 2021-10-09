import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
import os
from attack_models import MLP
from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData
warnings.simplefilter("ignore")


def transfer_based_attack(epochs):
    # GCN_DD_GPU1_12h53m37s_on_Jan_28_2021 0.7900280269	0.6378787879
    # GCN_DD_GPU0_19h36m32s_on_Jan_27_2021  0.822117084 0.7315151515

    # GCN_PROTEINS_full_GPU0_03h11m51s_on_Jan_28_2021 0.7707677769	0.5766666667
    attack_base_path = 'data/statis/GCN/GCN_ENZYMES_GPU0_16h40m29s_on_Jun_08_2021/'
    target_base_path = 'data/statis/GCN/GCN_DD_GPU1_16h26m32s_on_Jun_08_2021/'
    # GCN_ENZYMES_GPU0_16h40m29s_on_Jun_08_2021 -> GCN_DD_GPU1_16h26m32s_on_Jun_08_2021
    # For attack dataset
    if os.listdir(attack_base_path).__contains__("S_RUN_"):
        S_X_train_in = load_pickled_data(attack_base_path + 'S_RUN_/S_X_train_Label_1.pickle')
        S_y_train_in = load_pickled_data(attack_base_path + 'S_RUN_/S_y_train_Label_1.pickle')
        S_X_train_out = load_pickled_data(attack_base_path + 'S_RUN_/S_X_train_Label_0.pickle')
        S_y_train_out = load_pickled_data(attack_base_path + 'S_RUN_/S_y_train_Label_0.pickle')
        S_Label_0_num_nodes = load_pickled_data(attack_base_path + 'S_RUN_/S_num_node_0.pickle')
        S_Label_1_num_nodes = load_pickled_data(attack_base_path + 'S_RUN_/S_num_node_1.pickle')
        S_Label_0_num_edges = load_pickled_data(attack_base_path + 'S_RUN_/S_num_edge_0.pickle')
        S_Label_1_num_edges = load_pickled_data(attack_base_path + 'S_RUN_/S_num_edge_1.pickle')
    else:
        S_X_train_in = load_pickled_data(attack_base_path + 'X_train_Label_1.pickle')
        S_y_train_in = load_pickled_data(attack_base_path + 'y_train_Label_1.pickle')
        S_X_train_out = load_pickled_data(attack_base_path + 'X_train_Label_0.pickle')
        S_y_train_out = load_pickled_data(attack_base_path + 'y_train_Label_0.pickle')
    # For target Dataset
    if os.listdir(target_base_path).__contains__("T_RUN_"):
        T_X_train_in = load_pickled_data(target_base_path + 'T_RUN_/T_X_train_Label_1.pickle')
        T_y_train_in = load_pickled_data(target_base_path + 'T_RUN_/T_y_train_Label_1.pickle')
        T_X_train_out = load_pickled_data(target_base_path + 'T_RUN_/T_X_train_Label_0.pickle')
        T_y_train_out = load_pickled_data(target_base_path + 'T_RUN_/T_y_train_Label_0.pickle')
        T_Label_0_num_nodes = load_pickled_data(target_base_path + 'T_RUN_/T_num_node_0.pickle')
        T_Label_1_num_nodes = load_pickled_data(target_base_path + 'T_RUN_/T_num_node_1.pickle')
        T_Label_0_num_edges = load_pickled_data(target_base_path + 'T_RUN_/T_num_edge_0.pickle')
        T_Label_1_num_edges = load_pickled_data(target_base_path + 'T_RUN_/T_num_edge_1.pickle')
    else:
        T_X_train_in = load_pickled_data(target_base_path + 'X_train_Label_1.pickle')
        T_y_train_in = load_pickled_data(target_base_path + 'y_train_Label_1.pickle')
        T_X_train_out = load_pickled_data(target_base_path + 'X_train_Label_0.pickle')
        T_y_train_out = load_pickled_data(target_base_path + 'y_train_Label_0.pickle')

    # print("T_X_train_in Size:{} and T_X_train_out Size:{}".format(len(T_X_train_in), len(T_X_train_out)))
    # Prepare Dataset
    X_attack = torch.FloatTensor(np.concatenate((S_X_train_in, S_X_train_out), axis=0))
    X_attack_nodes = torch.FloatTensor(np.concatenate((S_Label_1_num_nodes, S_Label_0_num_nodes), axis=0))
    X_attack_edges = torch.FloatTensor(np.concatenate((S_Label_1_num_edges, S_Label_0_num_edges), axis=0))

    y_target = torch.FloatTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0))
    y_attack = torch.FloatTensor(np.concatenate((S_y_train_in, S_y_train_out), axis=0))
    X_target = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0))
    X_target_nodes = torch.FloatTensor(np.concatenate((T_Label_1_num_nodes, T_Label_0_num_nodes), axis=0))
    X_target_edges = torch.FloatTensor(np.concatenate((T_Label_1_num_edges, T_Label_0_num_edges), axis=0))

    feature_nums = min(X_attack.shape[1],X_target.shape[1])
    # print("feature_nums:{}".format(feature_nums))
    selected_X_target = select_top_k(X_target, feature_nums)
    selected_X_attack = select_top_k(X_attack, feature_nums)

    # selected_X_attack, selected_X_target = X_attack,X_target
    n_in = selected_X_attack.shape[1]
    attack_model = MLP(in_size=n_in, out_size=1, hidden_1=64, hidden_2=64)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)
    attack_data = trainData(selected_X_attack, y_attack)
    target_data = testData(selected_X_target)
    train_loader = DataLoader(dataset=attack_data, batch_size=64, shuffle=True)
    target_loader = DataLoader(dataset=target_data, batch_size=1)
    all_acc = []
    for i in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = attack_model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        all_acc.append(epoch_acc)
    y_pred_list = []
    attack_model.eval()
    correct_node_list, correct_edge_list = [],[]
    incorrect_node_list, incorrect_edge_list = [],[]
    with torch.no_grad():
        for X_batch, num_node, num_edge,y in zip(target_loader, X_target_nodes, X_target_edges, y_target):
            y_test_pred = attack_model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            if y == y_pred_tag.detach().item():
                correct_node_list.append(num_node.detach().item())
                correct_edge_list.append(num_edge.detach().item())
            else:
                incorrect_node_list.append(num_node.detach().item())
                incorrect_edge_list.append(num_edge.detach().item())
            y_pred_list.append(y_pred_tag.cpu().numpy()[0])
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    report = classification_report(y_target, y_pred_list)
    precision, recall, fscore, support = precision_recall_fscore_support(y_target,
                                                                         y_pred_list, average='macro')
    print(precision, recall)

    # print("correct_node_list:",correct_node_list)
    # print("correct_edge_list:",correct_edge_list)
    # print("incorrect_node_list:",incorrect_node_list)
    # print("incorrect_edge_list:",incorrect_edge_list)

    print(np.mean(correct_node_list), np.mean(correct_edge_list))
    print(np.mean(incorrect_node_list), np.mean(incorrect_edge_list))


if __name__ == '__main__':
    transfer_based_attack(300)