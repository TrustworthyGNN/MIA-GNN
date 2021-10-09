import os
import pickle
import torch
from scipy.spatial import distance
import numpy as np
from torch import nn
import pandas as pd
from sklearn.metrics import accuracy_score

def load_pickled_data(path):
    with open(path, 'rb') as f:
        unPickler = pickle.load(f)
        return unPickler


def load_data(m_path, nm_path):
    data_in = load_pickled_data(m_path)
    data_out = load_pickled_data(nm_path)
    return data_in, data_out


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def cal_distances(data):
    # print('calculate distances...')
    distance_matrix = []
    for raw in data:
        label = np.argmax(raw)
        # cosine_dis = distance.cosine(label, raw)
        euclid_dis = distance.euclidean(label, raw)
        # corr_dis = distance.correlation(np.argmax(raw), raw) # nan
        cheby_dis = distance.chebyshev(label, raw)
        bray_dis = distance.braycurtis(label, raw)
        canber_dis = distance.canberra(label, raw)
        mahal_dis = distance.cityblock(label, raw)
        sqeuclid_dis = distance.sqeuclidean(label, raw)
        v = [euclid_dis, cheby_dis, bray_dis, canber_dis, mahal_dis, sqeuclid_dis]
        distance_matrix.append(v)
    return distance_matrix


def cal_distance(data):
    label = np.argmax(data)
    # cosine_dis = distance.cosine(label, raw)
    euclid_dis = distance.euclidean(label, data)
    # corr_dis = distance.correlation(np.argmax(raw), raw) # nan
    cheby_dis = distance.chebyshev(label, data)
    bray_dis = distance.braycurtis(label, data)
    canber_dis = distance.canberra(label, data)
    mahal_dis = distance.cityblock(label, data)
    sqeuclid_dis = distance.sqeuclidean(label, data)
    v = [euclid_dis, cheby_dis, bray_dis, canber_dis, mahal_dis, sqeuclid_dis]
    return v


# Setup a plot such that only the bottom spine is shown
def get_all_probabilities(data, factor):
    return_list = []
    for d in data:
        return_list.append(d[np.argmax(d)] * factor)
    return return_list


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def get_all_exps(path):
    dataset_list = ['CIFAR10', 'MNIST', 'DD', 'ENZYMES', 'PROTEINS_full', 'OGBG']
    folders = os.listdir(path)
    assert len(folders) > 0, "No dataset folder exist!"
    exp_path_list = []
    for folder in folders:
        if dataset_list.__contains__(folder):
            dataset_folder = os.path.join(path, folder)
            exps = os.listdir(dataset_folder)
            for exp in exps:
                exp_path = os.path.join(dataset_folder, exp)
                exp_path_list.append(exp_path)
    return exp_path_list


if __name__ == '__main__':
    base_path = 'data/statis/GCN'
    folders = os.listdir(base_path)
    # print(sorted(folders))
    for folder in folders:
        all_exps = os.listdir(base_path + '/' + folder)
        print(all_exps)
        # for exp_path in all_exps:
        #     exp_path = os.path.join(base_path + '/' + folder, exp_path)
        # print(exp_path)
        flag = 2
        if all_exps.__contains__('S_RUN_'):
            exp_path = os.path.join(base_path + '/' + folder, 'S_RUN_')
            m_data_path = os.path.join(exp_path, 'S_X_train_Label_1.pickle')
            nm_data_path = os.path.join(exp_path, 'S_X_train_Label_0.pickle')
            S_Label_0_num_nodes = load_pickled_data(exp_path + '/S_num_node_0.pickle')
            S_Label_1_num_nodes = load_pickled_data(exp_path + '/S_num_node_1.pickle')
            S_Label_0_num_edges = load_pickled_data(exp_path + '/S_num_edge_0.pickle')
            S_Label_1_num_edges = load_pickled_data(exp_path + '/S_num_edge_1.pickle')
            X_attack_nodes = torch.FloatTensor(np.concatenate((S_Label_1_num_nodes, S_Label_0_num_nodes), axis=0))
            X_attack_edges = torch.FloatTensor(np.concatenate((S_Label_1_num_edges, S_Label_0_num_edges), axis=0))
            data_in, data_out = load_data(m_data_path, nm_data_path)
            # LOSS function based attack
            # ce_criterion = nn.CrossEntropyLoss()
            # nl_criterion = nn.NLLLoss()
            if flag == 1:
                mse_criterion = nn.MSELoss()
                ce_criterion = nn.CrossEntropyLoss()
                mse_in_loss_list, mse_out_loss_list, loss_diff_list = [], [], []
                ce_in_loss_list, ce_out_loss_list, ce_loss_diff_list = [], [], []
                with open('out/dd_loss_difference_calculation_single_instance.txt', 'a+') as writer:
                    writer.write("For Experiment:{} \n".format(exp_path))
                    for i in range(min(len(data_in), len(data_out))):
                        x_in, x_in_label = data_in[i], np.argmax(data_in[i])
                        x_out, x_out_label = data_out[i], np.argmax(data_out[i])

                        ce_in_loss = ce_criterion(torch.FloatTensor([x_in]), torch.LongTensor([x_in_label]))
                        ce_out_loss = ce_criterion(torch.FloatTensor([x_out]), torch.LongTensor([x_out_label]))

                        mse_in_loss = mse_criterion(torch.FloatTensor([x_in]), torch.LongTensor([x_in_label]))
                        mse_out_loss = mse_criterion(torch.FloatTensor([x_out]), torch.LongTensor([x_out_label]))

                        mse_in_loss_list.append(float(mse_in_loss.numpy()))
                        mse_out_loss_list.append(float(mse_out_loss.numpy()))

                        ce_in_loss_list.append(float(ce_in_loss.numpy()))
                        ce_out_loss_list.append(float(ce_out_loss.numpy()))

                    loss_diff_list.append(np.mean(mse_in_loss_list) - np.mean(mse_out_loss_list))
                    # writer.write("MSE Difference:{}\n".format(loss_diff_list))
                    print(np.mean(ce_in_loss_list), np.mean(ce_out_loss_list))
                    writer.write(
                        "\t\tMSELLoss for Member:\n\t{} and Non-Member：\n\t{}\n".format(ce_in_loss_list,
                                                                                        ce_out_loss_list))
            elif flag == 2:
                print('plot all probabilities of {}'.format(exp_path))
                data_in_list = get_all_probabilities(data_in, 1)
                data_out_list = get_all_probabilities(data_out, 1)
                count_m = 0
                count_nm = 0
                t = np.array(range(9900, 10000)) / 10000
                with open('out/threshold_based_attack_result.txt', 'a+') as writer:
                    writer.write("For Experiment:{} \n".format(exp_path))
                    for t_ in t:
                        correct_node_list, correct_edge_list = [], []
                        incorrect_node_list, incorrect_edge_list = [], []
                        y_true, y_pred_list = [],[]
                        num_nodes, num_edges = [], []
                        a = [x for x in data_in_list if x > t_]
                        b = [x for x in data_out_list if x > t_]
                        s_data = np.concatenate((data_in, data_out), axis=0)
                        s_data_label = np.concatenate(([1 for i in data_in], [0 for j in data_in]), axis=0)
                        max_prob_list = []
                        for i in range(min(len(s_data), len(s_data_label))):
                            # x_in, x_in_label = data_in[i], np.argmax(data_in[i])
                            # x_out, x_out_label = data_out[i], np.argmax(data_out[i])
                            max_prob_list.append(np.max(s_data[i]))
                            num_nodes.append(X_attack_nodes[i].detach().item())
                            num_edges.append(X_attack_edges[i].detach().item())
                            # print(np.max(x_in), x_in_label)
                        print(max_prob_list)
                        # print(len(s_data),len(s_data_label), len(X_attack_nodes))
                        writecsv = pd.DataFrame(
                            {'num_node': num_nodes, 'num_edge': num_edges, 'max_prob': max_prob_list, 'label': s_data_label})
                        print(writecsv)
                        writecsv.to_csv("data/statis/test_results_with_threshold_" + str(t_) + ".csv",
                                        index=False)
                        # if x_in > t_:
                        #     correct_node_list.append(n)
                        #     correct_edge_list.append(e)
                        #     y_pred_list.append(0)
                        # else:
                        #     incorrect_node_list.append(n)
                        #     incorrect_edge_list.append(e)
                        #     y_pred_list.append(1)
                        # y_true.append(np.argmax(d))

                    # for c, d, n, e in zip(s_data,s_data_value, X_attack_nodes, X_attack_edges):
                    #     num_nodes.append(n.detach().item())
                    #     num_edges.append(e.detach().item())
                    #     if c > t_:
                    #         correct_node_list.append(n)
                    #         correct_edge_list.append(e)
                    #         y_pred_list.append(0)
                    #     else:
                    #         incorrect_node_list.append(n)
                    #         incorrect_edge_list.append(e)
                    #         y_pred_list.append(1)
                    #     y_true.append(np.argmax(d))
                    # y = [y_t.detach().item() for y_t in y_target]
                    # writecsv = pd.DataFrame(
                    #     {'num_node': num_nodes, 'num_edge': num_edges, 'label': y_true, 'predict': y_pred_list})
                    # print(writecsv)
                    # writecsv.to_csv("data/statis/test_results_with_threshold_" + str(t_) + ".csv",
                    #                 index=False)
                    # writer.write(
                    #     '\t\t With threshold:{}, Percentage for member is:{}, and for non-member is:{}, '
                    #     'and num of correct node mean:{}, and num of correct edges:{}, '
                    #     'and num of incorrect node mean:{}, '
                    #     'and num of incorrect edge mean:{}'.format(
                    #         t_,
                    #         len(
                    #             a) / len(
                    #             data_in_list),
                    #         len(
                    #             b) / len(
                    #             data_out_list), np.mean(correct_node_list), np.mean(correct_edge_list),
                    #     np.mean(incorrect_node_list), np.mean(incorrect_edge_list)))
                    # writer.write('\n')
                    #
                    # print(accuracy_score(y_true,y_pred_list))