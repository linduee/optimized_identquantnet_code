"""Training workflow for the standalone Optimized IdentQuantNet bundle."""

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
import math
import copy
import random
import argparse
import torch.utils
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker
import os
import sys

import torch.utils.data

if "__file__" in globals():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    _cwd = os.getcwd()
    if os.path.basename(_cwd) == "optimized_identquantnet_code":
        CURRENT_DIR = _cwd
    elif os.path.isdir(os.path.join(_cwd, "optimized_identquantnet_code")):
        CURRENT_DIR = os.path.join(_cwd, "optimized_identquantnet_code")
    else:
        CURRENT_DIR = _cwd
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import identification_model
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import gaussian_kde
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


def _first_existing_path(candidates):
    normalized = [candidate for candidate in candidates if candidate]
    for candidate in normalized:
        if os.path.exists(candidate):
            return candidate
    if normalized:
        return normalized[0]
    return None


BUNDLE_ROOT = CURRENT_DIR
LOCAL_DATASET_INPUT_DIR = os.path.join(BUNDLE_ROOT, "data_input")
LOCAL_DATASET_DIR = os.path.join(BUNDLE_ROOT, "data")
DATASET_ROOT = _first_existing_path(
    [
        LOCAL_DATASET_INPUT_DIR if os.path.isdir(LOCAL_DATASET_INPUT_DIR) else None,
        LOCAL_DATASET_DIR if os.path.isdir(LOCAL_DATASET_DIR) else None,
        os.environ.get("OPTIMIZED_IDENTQUANTNET_DATASET_DIR"),
        LOCAL_DATASET_INPUT_DIR,
        LOCAL_DATASET_DIR,
    ]
)
TRAINED_RESULTS_SUBDIR = "model_save_trial"
TRAINED_RESULTS_DIR = os.path.join(BUNDLE_ROOT, TRAINED_RESULTS_SUBDIR)

if DATASET_ROOT is None:
    raise FileNotFoundError(
        "No dataset directory could be resolved.\n"
        f"Place your dataset under '{LOCAL_DATASET_INPUT_DIR}' or '{LOCAL_DATASET_DIR}', or set "
        "OPTIMIZED_IDENTQUANTNET_DATASET_DIR to your dataset directory."
    )


def _require_existing_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}\\n"
            f"Current dataset root: {DATASET_ROOT}\\n"
            f"You can either place the expected files under "
            f"'{os.path.join(BUNDLE_ROOT, 'data_input')}' or '{os.path.join(BUNDLE_ROOT, 'data')}', or set "
            f"OPTIMIZED_IDENTQUANTNET_DATASET_DIR to your dataset directory."
        )
    return path


plt.rc('font', family = 'Times New Roman')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_path = CURRENT_DIR
results_subdir = TRAINED_RESULTS_SUBDIR
results_dir = TRAINED_RESULTS_DIR
os.makedirs(results_dir, exist_ok=True)

random.seed(1021)
torch.manual_seed(1021)
num_data = 10000

parser = argparse.ArgumentParser()
parser.add_argument('--fontsize_fig', type = int, default = 35)
parser.add_argument('--dataset_ratio', type = list, default = [0.6, 0.2, 0.2])
parser.add_argument('--batch_size', type = int, default = 256)
parser.add_argument('--lr', type = float, default = 0.0001)
parser.add_argument('--num', type = int, default = num_data)
parser.add_argument('--num_epochs', type = int, default = 0)

HyperParameters = parser.parse_args(args = [])


parser = argparse.ArgumentParser()
parser.add_argument('--num', type = int, default = num_data)
parser.add_argument('--fontsize_fig', type = int, default = 35)
parser.add_argument('--dataset_ratio', type = list, default = [0.6, 0.2, 0.2])
parser.add_argument('--batch_size_id', type = int, default = 256)
parser.add_argument('--batch_size_qu', type = int, default = 128)
parser.add_argument('--batch_size_error', type = int, default = 32)
parser.add_argument('--lr_id', type = float, default = 0.0001)
parser.add_argument('--lr_qu_mse', type = float, default = 0.0001)
parser.add_argument('--lr_qu_weightedmse', type = float, default = 0.0005)
parser.add_argument('--weighted_threshold', type = float, default = 0.0)
parser.add_argument('--penalty_alpha', type = float, default = 100.0)
parser.add_argument('--num_epochs_id', type = int, default = 100)
parser.add_argument('--num_epochs_mix_id', type = int, default = 1000)
parser.add_argument('--num_epochs_qu_mse', type = int, default = 2000)
parser.add_argument('--num_epochs_qu_weightedmse', type = int, default = 10000)
parser.add_argument('--identification_training_flag', type = bool, default = False)
parser.add_argument('--quantification_regression_training_index_mse', type = list, default = [])
parser.add_argument('--quantification_regression_training_index_weightedmse', type = list, default = [])
parser.add_argument('--group_identification_training_flag', type = bool, default = False)
parser.add_argument('--group_quantification_regression_training_index_mse', type = list, default = [])
parser.add_argument('--group_quantification_regression_training_index_weightedmse', type = list, default = [])
parser.add_argument('--group_mix_classification_index', type = list, default = [])
parser.add_argument('--group_errorprediction_index', type = list, default = [])
parser.add_argument('--error_tolerance', type = list, default = [0, 0, 0, 0, 0])
parser.add_argument('--group_mlp_mix_classification_index', type = list, default = [])

IdentQuantParameters = parser.parse_args(args = [])

def mape(actual, predicted):
    return np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100


def mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))


def mse(actual, predicted):
    return np.mean((np.array(actual) - np.array(predicted)) ** 2)

def accurcay(actual, predicted):
    residual = np.abs(np.array(actual) - np.array(predicted)).tolist()
    num = len(residual)
    per1 = sum(1 for x in residual if x <= 1) / num
    per2 = sum(1 for x in residual if (x <= 2 and x > 1)) / num
    per3 = sum(1 for x in residual if (x <= 3 and x > 2)) / num
    per4 = sum(1 for x in residual if (x <= 4 and x > 3)) / num
    per5 = sum(1 for x in residual if (x <= 5 and x > 4)) / num
    per5more = sum(1 for x in residual if x > 5) / num


    return [per1, per1 + per2, per1 + per2 + per3, per1 + per2 + per3 + per4, per1 + per2 + per3 + per4 + per5, per5more]

def overestimation_percentage(actual, predicted):
    num_over = 0
    for i in range(len(actual)):
        if predicted[i] > actual[i]:
            num_over += 1
    return num_over / len(actual) * 100

def is_subset(list1, list2):
    return all(any(d1 == d2 for d2 in list2) for d1 in list1)


class TwoWayANN(nn.Module):
    def __init__(self, device, net_layer_size_list, feature_size, label_size, activation_func):
        super(TwoWayANN, self).__init__()
        self.device = device
        self.input_channels = feature_size
        self.output_channels = label_size
        self.activation_func = activation_func
        self.feature_1 = self.make_layers_seq(net_layer_size_list[0][:-1], self.input_channels[0], net_layer_size_list[0][-1])
        self.feature_2 = self.make_layers_seq(net_layer_size_list[0][:-1], self.input_channels[1], net_layer_size_list[0][-1])
        self.qu_1 = self.make_layers_seq(net_layer_size_list[1], net_layer_size_list[0][-1], self.output_channels[0])
        self.qu_2 = self.make_layers_seq(net_layer_size_list[2], net_layer_size_list[0][-1] * 2, self.output_channels[1])

    def make_layers_seq(self, size_list, input_channels, output_channels):
        layers = []
        for size in size_list:
            layers += [nn.Linear(input_channels, size)]
            if self.activation_func == 'relu':
                layers += [nn.ReLU()]
            elif self.activation_func == 'leaky_relu':
                layers += [nn.LeakyReLU(0.1)]
            elif self.activation_func == 'elu':
                layers += [nn.ELU()]
            input_channels = size
        layers += [nn.Linear(input_channels, output_channels)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x_1 = x[:,:, :self.input_channels[0]]
        x_2 = x[:,:, self.input_channels[0]:]
        feature_1 = self.feature_1(x_1)
        feature_2 = self.feature_2(x_2)
        y_1 = self.qu_1(feature_1)
        y_2 = self.qu_2(torch.cat((feature_1, feature_2), dim = 2))

        return [y_1, y_2]


ETO = {
    'name': 'ETO',
    'index': 0,
    'resolution': 1,
    'Therapeutic range': [0, 60]
}

MTX = {
    'name': 'MTX',
    'index': 1,
    'resolution': 1,
    'Therapeutic range': [0, 130]
}

IFO = {
    'name': 'IFO',
    'index': 2,
    'resolution': 1,
    'Therapeutic range': [50, 150]
}

CP = {
    'name': 'CP',
    'index': 3,
    'resolution': 10,
    'Therapeutic range': [50, 150]
}

_5FU_CV = {
    'name': '5_FU_CV',
    'index': 4,
    'resolution': 1,
    'Therapeutic range': [40, 310]
}


drugs = [ETO, MTX, IFO, CP, _5FU_CV]
drugs_name = [drugs[j]['name'] for j in range(len(drugs))]
peaks = [[2, 0],[1, 0], [0, 1], [0, 1], [1, 0]]

quantification_label_list = [[] for _ in drugs]
quantification_label_concentration_list =[[] for _ in drugs]
for i in range(len(drugs)):
    start_con = drugs[i]['Therapeutic range'][0]
    end_con = drugs[i]['Therapeutic range'][1]
    resolution_now = drugs[i]['resolution']
    while start_con + resolution_now < end_con:
        quantification_label_list[i].append([start_con, start_con + resolution_now])
        quantification_label_concentration_list[i].append(sum(quantification_label_list[i][-1]) / 2)
        start_con += resolution_now
    quantification_label_list[i].append([start_con, end_con])
    quantification_label_concentration_list[i].append(sum(quantification_label_list[i][-1]) / 2)

feature_name = ['Mean', 'Variance', 'Amplitude', 'Peak']
selected_feature_index = [0, 1, 2, 3]

drugs_group = [[CP, ETO], [ETO, IFO, MTX], [CP, MTX, _5FU_CV], [ETO, IFO], [ETO, IFO]]
drugs_group_name = ['EPOCH', 'MIED', 'CMF', 'ICE', 'AVI']


base_simulated_data_path = os.path.join(DATASET_ROOT, 'identifiaction_and_quantification_')
base_measured_data_path = os.path.join(DATASET_ROOT, 'measured_identifiaction_and_quantification_')

drugs_single = [ETO, MTX, IFO, CP, _5FU_CV]
drugs_single_name = [drugs_single[j]['name'] for j in range(len(drugs_single))]

single_simulated_data_path = [base_simulated_data_path + drugs_single_name[i] + '.pickle' for i in range(len(drugs_single_name))]
single_measured_data_path = [base_measured_data_path + drugs_single_name[i] + '.pickle' for i in range(len(drugs_single_name))]


drugs_mix = [[ETO, MTX]]
peaks_mix = [[[ETO, ETO, MTX],[]]]
peaks_select_index_oxidation_mix = [[0, 2]]
peaks_select_index_reduction_mix = [[]]
drugs_mix_name = [[drugs_mix[i][j]['name'] for j in range(len(drugs_mix[i]))] for i in range(len(drugs_mix))]


mix_simulated_data_path = []
mix_measured_data_path = []
for i in range(len(drugs_mix_name)):
    new_simulated_path = copy.deepcopy(base_simulated_data_path)
    new_measured_path = copy.deepcopy(base_measured_data_path)
    for j in range(len(drugs_mix_name[i])):
        new_simulated_path += drugs_mix_name[i][j]
        new_measured_path += drugs_mix_name[i][j]
        if j != (len(drugs_mix_name[i]) - 1):
            new_simulated_path += '_'
            new_measured_path += '_'
    new_simulated_path += '.pickle'
    new_measured_path += '.pickle'
    mix_simulated_data_path.append(new_simulated_path)
    mix_measured_data_path.append(new_measured_path)

class WeightedMSE_Loss(nn.Module):
    def __init__(self, threshold=0, alpha=1.0, error_weight=100, dimension=5, avoid="underestimation"):
        super(WeightedMSE_Loss, self).__init__()
        self.threshold = threshold
        self.alpha = alpha
        self.error_weight = error_weight
        self.dimension = dimension
        self.avoid = avoid

    def forward(self, output, label):
        raw_residual = output - label
        if self.avoid == "underestimation":
            boundary_residual = raw_residual + self.threshold
            abs_res = torch.abs(boundary_residual)
            loss_weight = torch.where(
                boundary_residual < 0,
                self.error_weight * torch.log(abs_res + self.error_weight).pow(1 / self.dimension),
                0.1 * torch.log(abs_res + 10).pow(-1),
            )
        elif self.avoid == "overestimation":
            boundary_residual = raw_residual - self.threshold
            abs_res = torch.abs(boundary_residual)
            loss_weight = torch.where(
                boundary_residual > 0,
                self.error_weight * torch.log(abs_res + self.error_weight).pow(1 / self.dimension),
                0.1 * torch.log(abs_res + 10).pow(-1),
            )
        else:
            raise ValueError("Invalid value for 'avoid'. Use 'underestimation' or 'overestimation'.")

        weighted_mse = loss_weight * (boundary_residual ** 2)
        return weighted_mse.mean()


IdentQuantParameters.group_identification_training_flag = True
IdentQuantParameters.group_quantification_regression_training_index_mse = [0, 1, 2, 3]
IdentQuantParameters.group_quantification_regression_training_index_weightedmse = [0, 1, 2, 3]
IdentQuantParameters.group_mix_classification_index = [0, 1, 2, 3, 4]
IdentQuantParameters.group_mlp_mix_classification_index = [0, 1, 2, 3, 4]
IdentQuantParameters.group_errorprediction_index = [0, 1, 2, 3, 4]

predict_drugs = [ETO, MTX, IFO, CP, _5FU_CV]
mul_drugs_group = [[CP, ETO], [ETO, IFO], [ETO, IFO, MTX], [CP, MTX, _5FU_CV]]
predict_drugs_branches = [[ETO['index'], MTX['index']], [IFO['index']], [CP['index']], [_5FU_CV['index']]]
predict_drugs_names = [ETO['name'] + MTX['name'], IFO['name'], CP['name'], _5FU_CV['name']]
predict_drug_flag = [0 for _ in range(len(drugs))]
identification_feature_index = [0, 2, 3]
quantification_feature_index = [[[2, 3], [2, 3]], [1, 2, 3], [1, 2, 3], [2, 3]]
quantification_feature_index_single_drug = [[2, 3], [2, 3], [1, 2, 3], [1, 2, 3], [2, 3]]

for i in range(len(predict_drugs)):
    index_now = predict_drugs[i]['index']
    predict_drug_flag[index_now] = 1

simulated_C_output_dataset = []
simulated_drug_output_index_dataset = []
simulated_Gaussian_output_dataset = []

measured_C_output_dataset = []
measured_drug_output_index_dataset = []
measured_Gaussian_output_dataset = []

for i in range(len(drugs_mix)):
    if is_subset(drugs_mix[i], predict_drugs):
        print(drugs_mix_name[i])
        list_file = open(_require_existing_file(mix_simulated_data_path[i]), 'rb')
        (C_output_dataset, drug_output_index_dataset, Gaussian_output_dataset_oxidation, Gaussian_output_dataset_reduction) = pickle.load(list_file)
        drug_output_index_dataset = [[drugs_mix[i][j]['index'] for j in range(len(drugs_mix[i]))] for _ in range(len(drug_output_index_dataset))]
        for j in range(IdentQuantParameters.num):
            simulated_C_output_dataset.append(C_output_dataset[j])
            simulated_drug_output_index_dataset.append(drug_output_index_dataset[j])
            if len(peaks_select_index_oxidation_mix[i]) != 0:
                simulated_Gaussian_output_dataset.append(np.array(Gaussian_output_dataset_oxidation[j])[peaks_select_index_oxidation_mix[i], :].tolist()) 
            if len(peaks_select_index_reduction_mix[i]) != 0:
                simulated_Gaussian_output_dataset.append(np.array(Gaussian_output_dataset_reduction[j])[peaks_select_index_reduction_mix[i], :].tolist())

        for j in range(len(drugs_mix[i])):
            print(np.max(np.array(drug_output_index_dataset)[:, j]))
            print(np.max(np.array(C_output_dataset)[:, j]))
            print(np.min(np.array(C_output_dataset)[:, j]))

        list_file = open(_require_existing_file(mix_measured_data_path[i]), 'rb')
        (C_output_dataset, drug_output_index_dataset, Gaussian_output_dataset_oxidation, Gaussian_output_dataset_reduction) = pickle.load(list_file)
        drug_output_index_dataset = [[drugs_mix[i][j]['index'] for j in range(len(drugs_mix[i]))] for _ in range(len(drug_output_index_dataset))]

        print(C_output_dataset)


        for j in range(len(C_output_dataset)):
            measured_C_output_dataset.append(C_output_dataset[j])
            measured_drug_output_index_dataset.append(drug_output_index_dataset[j])
            if len(peaks_select_index_oxidation_mix[i]) != 0:
                measured_Gaussian_output_dataset.append(np.array(Gaussian_output_dataset_oxidation[j])[peaks_select_index_oxidation_mix[i], :].tolist())
            if len(peaks_select_index_reduction_mix[i]) != 0:
                measured_Gaussian_output_dataset.append(np.array(Gaussian_output_dataset_reduction[j])[peaks_select_index_reduction_mix[i], :].tolist())

        for j in range(len(drugs_mix[i])):
            predict_drug_flag[drugs_mix[i][j]['index']] = 0

for i in range(len(drugs_single)):
    if predict_drug_flag[i] == 1:
        print(drugs_single_name[i])
        list_file = open(_require_existing_file(single_simulated_data_path[i]), 'rb')
        (C_output_dataset, drug_output_index_dataset, Gaussian_output_dataset_oxidation, Gaussian_output_dataset_reduction) = pickle.load(list_file)
        drug_output_index_dataset = [drugs_single[i]['index'] for _ in range(len(drug_output_index_dataset))]

        simulated_C_output_dataset += [[C_output_dataset[:IdentQuantParameters.num][k]] for k in range(IdentQuantParameters.num)]
        simulated_drug_output_index_dataset += [[drug_output_index_dataset[:IdentQuantParameters.num][k]] for k in range(IdentQuantParameters.num)]
        for j in range(IdentQuantParameters.num):
            if len(Gaussian_output_dataset_oxidation[0]) != 0:
                simulated_Gaussian_output_dataset += [Gaussian_output_dataset_oxidation[j]]
            if len(Gaussian_output_dataset_reduction[0]) != 0:
                simulated_Gaussian_output_dataset += [Gaussian_output_dataset_reduction[j]]

        list_file = open(_require_existing_file(single_measured_data_path[i]), 'rb')
        (C_output_dataset, drug_output_index_dataset, Gaussian_output_dataset_oxidation, Gaussian_output_dataset_reduction) = pickle.load(list_file)
        print(C_output_dataset)

        drug_output_index_dataset = [drugs_single[i]['index'] for _ in range(len(drug_output_index_dataset))]
        measured_C_output_dataset += [[k] for k in C_output_dataset]
        measured_drug_output_index_dataset += [[k] for k in drug_output_index_dataset]
        for j in range(len(C_output_dataset)):
            if len(Gaussian_output_dataset_oxidation) != 0:
                measured_Gaussian_output_dataset += [Gaussian_output_dataset_oxidation[j]]
            if len(Gaussian_output_dataset_reduction) != 0:
                measured_Gaussian_output_dataset += [Gaussian_output_dataset_reduction[j]]

print("Simulated:")
print(len(simulated_C_output_dataset))
print(len(simulated_drug_output_index_dataset))
print(len(simulated_Gaussian_output_dataset))

print("Measured:")
print(len(measured_C_output_dataset))
print(len(measured_drug_output_index_dataset))
print(len(measured_Gaussian_output_dataset))

index_random_simulated = [i for i in range(len(simulated_C_output_dataset))]
random.shuffle(index_random_simulated)
simulated_C_output_dataset_shuffle = [simulated_C_output_dataset[index_now] for index_now in index_random_simulated]
simulated_drug_output_index_dataset_shuffle = [simulated_drug_output_index_dataset[index_now] for index_now in index_random_simulated]
simulated_feature_input_shuffle = [simulated_Gaussian_output_dataset[index_now] for index_now in index_random_simulated]

n_train = int(len(simulated_feature_input_shuffle) * IdentQuantParameters.dataset_ratio[0])
n_val = int(len(simulated_feature_input_shuffle) * IdentQuantParameters.dataset_ratio[1])
n_test = len(simulated_feature_input_shuffle) - n_train - n_val
n_train_val = n_train + n_val
print(f"train:{n_train}, val:{n_val}, test:{n_test}")

train_simulated_C_output_dataset_shuffle = simulated_C_output_dataset_shuffle[:n_train]
val_simulated_C_output_dataset_shuffle = simulated_C_output_dataset_shuffle[n_train:n_train_val]
test_simulated_C_output_dataset_shuffle = simulated_C_output_dataset_shuffle[n_train_val:]
train_simulated_drug_output_index_dataset_shuffle = simulated_drug_output_index_dataset_shuffle[:n_train]
val_simulated_drug_output_index_dataset_shuffle = simulated_drug_output_index_dataset_shuffle[n_train:n_train_val]
test_simulated_drug_output_index_dataset_shuffle = simulated_drug_output_index_dataset_shuffle[n_train_val:]
train_simulated_feature_input_shuffle = simulated_feature_input_shuffle[:n_train]
val_simulated_feature_input_shuffle = simulated_feature_input_shuffle[n_train:n_train_val]
test_simulated_feature_input_shuffle = simulated_feature_input_shuffle[n_train_val:]


train_id_input_feature = []
train_id_input_label = []
train_qu_input_feature = [[] for _ in range(len(predict_drugs_branches))]
train_qu_input_label = [[] for _ in range(len(predict_drugs_branches))]

for i in range(len(train_simulated_C_output_dataset_shuffle)):
    train_id_input_feature += train_simulated_feature_input_shuffle[i]
    train_id_input_label += train_simulated_drug_output_index_dataset_shuffle[i]
    if train_simulated_drug_output_index_dataset_shuffle[i] == [0, 1]:
        train_qu_input_feature[0].append(train_simulated_feature_input_shuffle[i])
        train_qu_input_label[0].append(train_simulated_C_output_dataset_shuffle[i])
    elif train_simulated_drug_output_index_dataset_shuffle[i] == [2]:
        train_qu_input_feature[1].append(train_simulated_feature_input_shuffle[i])
        train_qu_input_label[1].append(train_simulated_C_output_dataset_shuffle[i])
    elif train_simulated_drug_output_index_dataset_shuffle[i] == [3]:
        train_qu_input_feature[2].append(train_simulated_feature_input_shuffle[i])
        train_qu_input_label[2].append(train_simulated_C_output_dataset_shuffle[i])
    elif train_simulated_drug_output_index_dataset_shuffle[i] == [4]:
        train_qu_input_feature[3].append(train_simulated_feature_input_shuffle[i])
        train_qu_input_label[3].append(train_simulated_C_output_dataset_shuffle[i])

val_id_input_feature = []
val_id_input_label = []
val_qu_input_feature = [[] for _ in range(len(predict_drugs_branches))]
val_qu_input_label = [[] for _ in range(len(predict_drugs_branches))]

for i in range(len(val_simulated_C_output_dataset_shuffle)):
    val_id_input_feature += val_simulated_feature_input_shuffle[i]
    val_id_input_label += val_simulated_drug_output_index_dataset_shuffle[i]
    if val_simulated_drug_output_index_dataset_shuffle[i] == [0, 1]:
        val_qu_input_feature[0].append(val_simulated_feature_input_shuffle[i])
        val_qu_input_label[0].append(val_simulated_C_output_dataset_shuffle[i])
    elif val_simulated_drug_output_index_dataset_shuffle[i] == [2]:
        val_qu_input_feature[1].append(val_simulated_feature_input_shuffle[i])
        val_qu_input_label[1].append(val_simulated_C_output_dataset_shuffle[i])
    elif val_simulated_drug_output_index_dataset_shuffle[i] == [3]:
        val_qu_input_feature[2].append(val_simulated_feature_input_shuffle[i])
        val_qu_input_label[2].append(val_simulated_C_output_dataset_shuffle[i])
    elif val_simulated_drug_output_index_dataset_shuffle[i] == [4]:
        val_qu_input_feature[3].append(val_simulated_feature_input_shuffle[i])
        val_qu_input_label[3].append(val_simulated_C_output_dataset_shuffle[i])

id_train_input_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(train_id_input_feature), dtype = torch.float32), torch.tensor(np.array(train_id_input_label).reshape(-1, 1)))
id_val_input_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(val_id_input_feature), dtype = torch.float32), torch.tensor(np.array(val_id_input_label).reshape(-1, 1)))
id_train_loader = torch.utils.data.DataLoader(id_train_input_dataset, IdentQuantParameters.batch_size_id, shuffle = True)
id_val_loader = torch.utils.data.DataLoader(id_val_input_dataset, IdentQuantParameters.batch_size_id)
qu_train_loader_list = []
qu_val_loader_list = []
for i in range(len(predict_drugs_branches)):
    qu_train_input_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(train_qu_input_feature[i]), dtype = torch.float32), torch.tensor(np.array(train_qu_input_label[i])))
    qu_val_input_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(val_qu_input_feature[i]), dtype = torch.float32), torch.tensor(np.array(val_qu_input_label[i])))
    qu_train_loader = torch.utils.data.DataLoader(qu_train_input_dataset, IdentQuantParameters.batch_size_qu, shuffle = True)
    qu_val_loader = torch.utils.data.DataLoader(qu_val_input_dataset, IdentQuantParameters.batch_size_qu)
    qu_train_loader_list.append(qu_train_loader)
    qu_val_loader_list.append(qu_val_loader)

identification_path = os.path.join(results_dir, 'optimized_identquantnet_identification')
identification_label_index = [0]
identification_model_build = identification_model.ANN_identification(device, [4, 8], len(identification_feature_index), len(predict_drugs), 'relu')
identification_model_build.to(device)
identification_citerion = nn.CrossEntropyLoss()
identification_optimizer = torch.optim.Adam(identification_model_build.parameters(), lr = IdentQuantParameters.lr_id)
identification_lossfunction_index = [0]
if IdentQuantParameters.group_identification_training_flag:
    id_train_acc_list, id_val_acc_list, id_train_loss_list, id_val_loss_list = identification_model.run(identification_model_build, id_train_loader, id_val_loader, IdentQuantParameters.num_epochs_id, identification_optimizer, identification_citerion, device, identification_lossfunction_index, identification_label_index, identification_feature_index)

    fig = plt.figure(figsize = (20, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(id_train_loss_list, label = 'Train Loss')
    ax1.plot(id_val_loss_list, label = 'Val Loss')
    ax1.set_xlabel('Epochs', fontsize = HyperParameters.fontsize_fig)
    ax1.set_ylabel('Loss', fontsize = HyperParameters.fontsize_fig)
    ax1.set_title('Identification Loss', fontsize = HyperParameters.fontsize_fig)
    ax2.plot(id_train_acc_list, label = 'Train Acc')
    ax2.plot(id_val_acc_list, label = 'Val Acc')
    ax2.set_xlabel('Epochs', fontsize = HyperParameters.fontsize_fig)
    ax2.set_ylabel('Acc', fontsize = HyperParameters.fontsize_fig)
    ax2.set_title('Identification Accuracy', fontsize = HyperParameters.fontsize_fig)
    for label in ax1.xaxis.get_ticklabels():
            label.set_fontsize(HyperParameters.fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(HyperParameters.fontsize_fig)
    for label in ax2.xaxis.get_ticklabels():
        label.set_fontsize(HyperParameters.fontsize_fig)
    for label in ax2.yaxis.get_ticklabels():
        label.set_fontsize(HyperParameters.fontsize_fig)
    ax1.legend()
    ax2.legend()
    plt.show()

    torch.save(identification_model_build.state_dict(), identification_path + ".pth")
    print("Saved PyTorch Model State to model.pth")

    list_file = open(identification_path + '.pickle','wb')
    pickle.dump((id_train_acc_list, id_val_acc_list, id_train_loss_list, id_val_loss_list), list_file)
    list_file.close()

quantification_regression_path = [os.path.join(results_dir, 'optimized_identquantnet_quantification_regression_' + predict_drugs_names[i]) for i in range(len(predict_drugs_branches))]
weighted_quantification_regression_path = [os.path.join(results_dir, 'optimized_identquantnet_weighted_quantification_regression_' + predict_drugs_names[i]) for i in range(len(predict_drugs_branches))]
classification_DecisionTreeClassifier_path = [os.path.join(results_dir, 'optimized_identquantnet_decision_tree_classifier_' + predict_drugs[i]['name'] + '.joblib') for i in range(len(predict_drugs))]
classification_RandomForestClassifier_path = [os.path.join(results_dir, 'optimized_identquantnet_random_forest_classifier_' + predict_drugs[i]['name'] + '.joblib') for i in range(len(predict_drugs))]
mse_error_prediction_path = [os.path.join(results_dir, 'optimized_identquantnet_mse_error_prediction_' + predict_drugs[i]['name']) for i in range(len(predict_drugs))]
weightedmse_error_prediction_path = [os.path.join(results_dir, 'optimized_identquantnet_weightedmse_error_prediction_' + predict_drugs[i]['name']) for i in range(len(predict_drugs))]
error_prediction_path = [os.path.join(results_dir, 'optimized_identquantnet_error_prediction_' + predict_drugs[i]['name']) for i in range(len(predict_drugs))]
all_weightedmse_error_prediction_path = [os.path.join(results_dir, 'optimized_identquantnet_weightedmse_error_prediction_all_' + predict_drugs[i]['name']) for i in range(len(predict_drugs))]
classification_mlp_path = [os.path.join(results_dir, 'optimized_identquantnet_mlp_classifier_' + predict_drugs[i]['name']) for i in range(len(predict_drugs))]


quantification_regression_label_index = [0]

quantification_regression_layer_list = [[[64, 64], [], [256, 256]], [64, 32, 16], [64, 32, 16], [64, 32, 16]]
weighted_quantification_regression_layer_list = [[[64, 64], [], [256, 256]], [64, 32, 16], [64, 32, 16], [64, 32, 16]]

mse_error_prediction_layer_list = [[64, 32, 16], [64, 32, 16], [64, 32, 16], [64, 32, 16], [64, 32, 16]]
weightedmse_error_prediction_layer_list = [[64, 32, 16], [64, 32, 16], [64, 32, 16], [64, 32, 16], [64, 32, 16]]

for i in IdentQuantParameters.group_quantification_regression_training_index_mse: 
    print('-' * 50)
    loss_type = 'MSE'
    if i == 0:
        quantification_model_build = TwoWayANN(device, quantification_regression_layer_list[i], [len(x) for x in quantification_feature_index[i]], [1, 1], 'relu')

    else:
        quantification_model_build = identification_model.ANN_identification(device, quantification_regression_layer_list[i], len(quantification_feature_index[i]), 1, 'relu')

    quantification_model_build.to(device)
    quantification_criterion = nn.MSELoss(reduction='mean').to(torch.float64)

    quantification_optimizer = torch.optim.Adam(quantification_model_build.parameters(), lr = IdentQuantParameters.lr_qu_mse)
    qu_train_loss_list, qu_val_loss_list = identification_model.run_regression(quantification_model_build, qu_train_loader_list[i], qu_val_loader_list[i], IdentQuantParameters.num_epochs_qu_mse, quantification_optimizer, quantification_criterion, device, quantification_regression_label_index, loss_type, quantification_feature_index[i])


    fig = plt.figure(figsize = (10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(qu_train_loss_list, label = 'Train Loss')
    ax1.plot(qu_val_loss_list, label = 'Val Loss')
    ax1.set_xlabel('Epochs', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_ylabel('Loss', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_title('Quantification Loss (%s)'%(predict_drugs_branches[i]), fontsize = IdentQuantParameters.fontsize_fig)

    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)

    ax1.legend()
    plt.show()

    torch.save(quantification_model_build.state_dict(), quantification_regression_path[i] + ".pth")
    print("Saved PyTorch Model State to model.pth")

    list_file = open(quantification_regression_path[i] + '.pickle','wb')
    pickle.dump((qu_train_loss_list, qu_val_loss_list), list_file)
    list_file.close()


for i in IdentQuantParameters.group_quantification_regression_training_index_weightedmse: 
    print('-' * 50)
    loss_type = 'WeightedMSE'
    if i == 0:
        quantification_model_build = TwoWayANN(device, weighted_quantification_regression_layer_list[i], [len(x) for x in quantification_feature_index[i]], [1, 1], 'relu')

    else:
        quantification_model_build = identification_model.ANN_identification(device, weighted_quantification_regression_layer_list[i], len(quantification_feature_index[i]), 1, 'relu')

    quantification_model_build.to(device)
    quantification_criterion = WeightedMSE_Loss(threshold= IdentQuantParameters.error_tolerance[i], error_weight=100, avoid= "underestimation").to(torch.float64)


    quantification_optimizer = torch.optim.Adam(quantification_model_build.parameters(), lr = IdentQuantParameters.lr_qu_weightedmse)
    qu_train_loss_list, qu_val_loss_list = identification_model.run_regression(quantification_model_build, qu_train_loader_list[i], qu_val_loader_list[i], IdentQuantParameters.num_epochs_qu_weightedmse, quantification_optimizer, quantification_criterion, device, quantification_regression_label_index, loss_type, quantification_feature_index[i])


    fig = plt.figure(figsize = (10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(qu_train_loss_list, label = 'Train Loss')
    ax1.plot(qu_val_loss_list, label = 'Val Loss')
    ax1.set_xlabel('Epochs', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_ylabel('Loss', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_title('Quantification Loss (%s)'%(predict_drugs_branches[i]), fontsize = IdentQuantParameters.fontsize_fig)

    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)

    ax1.legend()
    plt.show()

    torch.save(quantification_model_build.state_dict(), weighted_quantification_regression_path[i] + ".pth")
    print("Saved PyTorch Model State to model.pth")

    list_file = open(weighted_quantification_regression_path[i] + '.pickle','wb')
    pickle.dump((qu_train_loss_list, qu_val_loss_list), list_file)
    list_file.close()


identification_model_build.load_state_dict(torch.load(identification_path + '.pth'))
identification_model_build.eval()

quantification_regression_model_build_list = []
weighted_quantification_regression_model_build_list = []
for i in range(len(predict_drugs_branches)):
    if i == 0:
        quantification_model_build = TwoWayANN(device, quantification_regression_layer_list[i], [len(x) for x in quantification_feature_index[i]], [1, 1], 'relu')
        weighted_quantification_model_build = TwoWayANN(device, weighted_quantification_regression_layer_list[i], [len(x) for x in quantification_feature_index[i]], [1, 1], 'relu')
    else:
        quantification_model_build = identification_model.ANN_identification(device, quantification_regression_layer_list[i], len(quantification_feature_index[i]), 1, 'relu')
        weighted_quantification_model_build = identification_model.ANN_identification(device, weighted_quantification_regression_layer_list[i], len(quantification_feature_index[i]), 1, 'relu')

    quantification_model_build.load_state_dict(torch.load(quantification_regression_path[i] + '.pth'))
    quantification_model_build.eval()
    quantification_regression_model_build_list.append(quantification_model_build)

    weighted_quantification_model_build.load_state_dict(torch.load(weighted_quantification_regression_path[i] + '.pth'))
    weighted_quantification_model_build.eval()
    weighted_quantification_regression_model_build_list.append(weighted_quantification_model_build)


def label_assign(error_mse, error_weightedmse, std):
    if error_mse < - std:
        label_mse = 1
    elif (error_mse >= - std) and (error_mse < 0):
        label_mse = 2
    elif (error_mse >= 0) and (error_mse < std):
        label_mse = 3
    elif error_mse >= std:
        label_mse = 4

    if error_weightedmse < std:
        label_weightedmse = 1
    elif error_weightedmse >= std:
        label_weightedmse = 2

    if((label_mse == 1) and (label_weightedmse == 2)) or ((label_mse == 4) and (label_weightedmse == 2)):
        return 0
    elif ((label_mse == 2) and (label_weightedmse == 2)) or ((label_mse == 3) and (label_weightedmse == 2)) or  ((label_mse == 3) and (label_weightedmse == 1)):
        return 1
    elif ((label_mse == 1) and (label_weightedmse == 1)) or ((label_mse == 2) and (label_weightedmse == 1)) or ((label_mse == 4) and (label_weightedmse == 1)):
        return 2


def mix_quantification_train(qu_mse, qu_weightedmse, error_mse, error_weightedmse):
    abs_error_mse = abs(error_mse)
    abs_error_weightedmse = abs(error_weightedmse)
    weight_mse = 1/abs_error_mse/(1/abs_error_mse + 1/abs_error_weightedmse)
    weight_weightedmse = 1/abs_error_weightedmse/(1/abs_error_mse + 1/abs_error_weightedmse)
    return weight_mse * qu_mse + weight_weightedmse * qu_weightedmse


train_quantification_regression_prediction = [[] for _ in range(len(predict_drugs))]
weighted_train_quantification_regression_prediction = [[] for _ in range(len(predict_drugs))]
train_quantification_concentration_label = [[] for _ in range(len(predict_drugs))]
train_error_mse = [[] for _ in range(len(predict_drugs))]
train_error_weightedmse = [[] for _ in range(len(predict_drugs))]
train_feature = [[] for _ in range(len(predict_drugs))]
train_over_under = [[] for _ in range(len(predict_drugs))]
train_std = []
label_index = 0
for i in range(len(train_simulated_feature_input_shuffle)):
    qu_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array([train_simulated_feature_input_shuffle[i]]), dtype = torch.float32), torch.tensor(np.array([train_simulated_C_output_dataset_shuffle[i]])))
    qu_loader = torch.utils.data.DataLoader(qu_dataset)
    if train_simulated_drug_output_index_dataset_shuffle[i] == [0, 1]:
        qu_model = quantification_regression_model_build_list[0]
        weighted_qu_model = weighted_quantification_regression_model_build_list[0]
        feature_index = quantification_feature_index[0]
        for _, (inputs_org, labels_org) in enumerate(qu_loader):
            inputs, labels = torch.cat((inputs_org[:, 0, feature_index[0]], inputs_org[:, 1, feature_index[1]]), dim = 1).unsqueeze(1), labels_org.to(device)
            feature_0 = [0, 0, 0, 0]
            feature_1 = [0, 0, 0, 0]

            for k in range(len(quantification_feature_index[0][0])):
                feature_0[quantification_feature_index[0][0][k]] = inputs_org[:, 0, :].tolist()[0][quantification_feature_index[0][0][k]]

            for k in range(len(quantification_feature_index[0][1])):
                feature_1[quantification_feature_index[0][1][k]] = inputs_org[:, 1, :].tolist()[0][quantification_feature_index[0][1][k]]

            train_feature[0].append(feature_0)
            train_feature[1].append(feature_1)

            outputs = qu_model(inputs)
            train_quantification_regression_prediction[0].append(outputs[0].tolist()[0][0][0])
            train_quantification_regression_prediction[1].append(outputs[1].tolist()[0][0][0])

            outputs = weighted_qu_model(inputs)
            weighted_train_quantification_regression_prediction[0].append(outputs[0].tolist()[0][0][0])
            weighted_train_quantification_regression_prediction[1].append(outputs[1].tolist()[0][0][0])

            train_quantification_concentration_label[0].append(labels.tolist()[0][0])
            train_quantification_concentration_label[1].append(labels.tolist()[0][1])

            train_error_mse[0].append(train_quantification_regression_prediction[0][-1] - train_quantification_concentration_label[0][-1])
            train_error_mse[1].append(train_quantification_regression_prediction[1][-1] - train_quantification_concentration_label[1][-1])

            train_error_weightedmse[0].append(weighted_train_quantification_regression_prediction[0][-1] - train_quantification_concentration_label[0][-1])
            train_error_weightedmse[1].append(weighted_train_quantification_regression_prediction[1][-1] - train_quantification_concentration_label[1][-1])


            if train_quantification_concentration_label[0][-1] > train_quantification_regression_prediction[0][-1]:
                train_over_under[0].append(0)
            else:
                train_over_under[0].append(1)

            if train_quantification_concentration_label[1][-1] > train_quantification_regression_prediction[1][-1]:
                train_over_under[1].append(0)
            else:
                train_over_under[1].append(1)

    else:
        Id = train_simulated_drug_output_index_dataset_shuffle[i][0] - 1
        qu_model = quantification_regression_model_build_list[Id]
        weighted_qu_model = weighted_quantification_regression_model_build_list[Id]
        feature_index = quantification_feature_index[Id]
        for _, (inputs_org, labels_org) in enumerate(qu_loader):
            feature = [0, 0, 0, 0]
            if inputs_org.dim() == 2:
                inputs, labels = inputs_org[:, feature_index].to(device), labels_org.to(device)
                for k in feature_index:
                    feature[k] = inputs_org.tolist()[0][k]
            elif inputs_org.dim() == 3:
                inputs, labels = inputs_org[:, 0, feature_index].to(device), labels_org.to(device)
                for k in feature_index:
                    feature[k] = inputs_org.tolist()[0][0][k]

            train_feature[Id + 1].append(feature)
            train_quantification_regression_prediction[Id + 1].append(qu_model(inputs).tolist()[0][0])
            weighted_train_quantification_regression_prediction[Id + 1].append(weighted_qu_model(inputs).tolist()[0][0])
            train_quantification_concentration_label[Id + 1].append(labels.tolist()[0][0])

            train_error_mse[Id + 1].append(train_quantification_regression_prediction[Id + 1][-1] - train_quantification_concentration_label[Id + 1][-1])
            train_error_weightedmse[Id + 1].append(weighted_train_quantification_regression_prediction[Id + 1][-1] - train_quantification_concentration_label[Id + 1][-1])

            if train_quantification_concentration_label[Id + 1][-1] > train_quantification_regression_prediction[Id + 1][-1]:
                train_over_under[Id + 1].append(0)
            else:
                train_over_under[Id + 1].append(1)


for i in range(len(predict_drugs)):
    print('=' * 50)
    print(predict_drugs[i]['name'])

    now_over_under = train_over_under[i]
    now_feature = train_feature[i]


    for j in range(len(quantification_feature_index_single_drug[i])):
        feature_index = quantification_feature_index_single_drug[i][j]
        feature_now = np.array(now_feature)[:, feature_index].tolist()

        df = pd.DataFrame({
            'binary': now_over_under,
            'continuous': feature_now
        })

        df['binned'] = pd.cut(df['continuous'], bins=10)

        grouped = df.groupby(['binned', 'binary']).size().unstack(fill_value=0)

        grouped = grouped.div(grouped.sum(axis=1), axis=0)

        grouped.plot(kind='bar', stacked=True, color=['red', 'blue'])

        plt.title(predict_drugs[i]['name'] + '(%s)'% (feature_name[feature_index]))
        plt.xlabel("Continuous Value Ranges")
        plt.ylabel("Proportion")
        plt.legend(title='Binary Value', labels=['label > pred', 'label <= pred'], loc='upper right')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    std_dev = np.std(train_error_mse[i])
    train_std.append(std_dev)
    x_vals = np.linspace(-3 * std_dev, 3 * std_dev, 100)  # x 轴范围设为 3 倍标准差以内
    gaussian_pdf = norm.pdf(x_vals, loc=0, scale=std_dev)  # 0 均值，数据的标准差

    plt.plot(x_vals, gaussian_pdf, color="red", label="Gaussian PDF (mean=0)")

    plt.hist(train_error_mse[i], bins=10, density=True, alpha=0.5, color='blue', label="MSE")
    plt.hist(train_error_weightedmse[i], bins=10, density=True, alpha=0.5, color='green', label="WeightedMSE")

    plt.legend()
    plt.title("%s"%(predict_drugs[i]['name']))
    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.show()


max_error_mse_positive_list = []
for i in range(len(train_error_mse)):
    max_error_mse_positive_list.append(max(train_error_mse[i]))

train_mix_label = [[] for _ in range(len(predict_drugs))]
new_train_quantification_regression_prediction = [[] for _ in range(len(predict_drugs))]
for i in range(len(predict_drugs)):
    print('%s'%(predict_drugs[i]['name']))
    std_now = train_std[i]
    for j in range(len(train_error_mse[i])):
        label_new = label_assign(train_error_mse[i][j], train_error_weightedmse[i][j], std_now)
        train_mix_label[i].append(label_new)
        if label_new == 0:
            new_train_quantification_regression_prediction[i].append(mix_quantification_train(train_quantification_regression_prediction[i][j], weighted_train_quantification_regression_prediction[i][j], train_error_mse[i][j], train_error_weightedmse[i][j]))
        elif label_new == 1:
            new_train_quantification_regression_prediction[i].append(train_quantification_regression_prediction[i][j])
        elif label_new == 2:
            new_train_quantification_regression_prediction[i].append(weighted_train_quantification_regression_prediction[i][j])
    print('Mix: %d' % (train_mix_label[i].count(0)))
    print('MSE: %d' % (train_mix_label[i].count(1)))
    print('WeightedMSE: %d' % (train_mix_label[i].count(2)))


    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(train_quantification_concentration_label[i], weighted_train_quantification_regression_prediction[i], s = 20, label = 'WeightedMSE')
    ax1.plot(train_quantification_concentration_label[i], train_quantification_concentration_label[i], linestyle = "dashed", c = "black")
    ax1.plot(train_quantification_concentration_label[i], [x + IdentQuantParameters.error_tolerance[i] for x in train_quantification_concentration_label[i]] , linestyle = "dashed", c = "black")


    ax1.set_title("Drug Quantification Performance (%s)"%(predict_drugs[i]['name']), fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_xlabel(r'True Concentration ($\mu$M)', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_ylabel(r"Predicted Concentration ($\mu$M)", fontsize = IdentQuantParameters.fontsize_fig)
    ax1.legend(fontsize = IdentQuantParameters.fontsize_fig, loc = 2)
    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    plt.show()

    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(train_quantification_concentration_label[i], train_quantification_regression_prediction[i], s = 20, label = 'MSE')
    ax1.plot(train_quantification_concentration_label[i], train_quantification_concentration_label[i], linestyle = "dashed", c = "black")
    ax1.plot(train_quantification_concentration_label[i], [x-IdentQuantParameters.error_tolerance[i] for x in train_quantification_concentration_label[i]] , linestyle = "dashed", c = "black")


    ax1.set_title("Drug Quantification Performance (%s)"%(predict_drugs[i]['name']), fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_xlabel(r'True Concentration ($\mu$M)', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_ylabel(r"Predicted Concentration ($\mu$M)", fontsize = IdentQuantParameters.fontsize_fig)
    ax1.legend(fontsize = IdentQuantParameters.fontsize_fig, loc = 2)
    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    plt.show()

    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(train_quantification_concentration_label[i], train_quantification_regression_prediction[i], s = 20, label = 'MSE')
    ax1.plot(train_quantification_concentration_label[i], train_quantification_concentration_label[i], linestyle = "dashed", c = "black")

    for k in range(3):
        indices = [y for y, x in enumerate(train_mix_label[i]) if x == k]
        selected_point = [new_train_quantification_regression_prediction[i][y] for y in indices]
        selected_x = [train_quantification_concentration_label[i][y] for y in indices]
        ax1.scatter(selected_x, selected_point, s = 20, label = 'Mix-%d'%(k))


    ax1.set_title("Drug Quantification Performance (%s)"%(predict_drugs[i]['name']), fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_xlabel(r'True Concentration ($\mu$M)', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_ylabel(r"Predicted Concentration ($\mu$M)", fontsize = IdentQuantParameters.fontsize_fig)
    ax1.legend(fontsize = IdentQuantParameters.fontsize_fig, loc = 2)
    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    plt.show()

    print("Metrics:")
    print("MSE:")
    print('MAPE')
    print(mape(train_quantification_concentration_label[i], train_quantification_regression_prediction[i]))
    print('MAE')
    print(mae(train_quantification_concentration_label[i], train_quantification_regression_prediction[i]))
    print('MSE')
    print(mse(train_quantification_concentration_label[i], train_quantification_regression_prediction[i]))
    print("R2")
    print(r2_score(train_quantification_concentration_label[i], train_quantification_regression_prediction[i]))
    print('Accuracy')
    print(accurcay(train_quantification_concentration_label[i], train_quantification_regression_prediction[i]))
    print('Over_percentage')
    print(overestimation_percentage(train_quantification_concentration_label[i], train_quantification_regression_prediction[i]))


    print("WeightedMSE:")
    print('MAPE')
    print(mape(train_quantification_concentration_label[i], weighted_train_quantification_regression_prediction[i]))
    print('MAE')
    print(mae(train_quantification_concentration_label[i], weighted_train_quantification_regression_prediction[i]))
    print('MSE')
    print(mse(train_quantification_concentration_label[i], weighted_train_quantification_regression_prediction[i]))
    print("R2")
    print(r2_score(train_quantification_concentration_label[i], weighted_train_quantification_regression_prediction[i]))
    print('Accuracy')
    print(accurcay(train_quantification_concentration_label[i], weighted_train_quantification_regression_prediction[i]))
    print('Over_percentage')
    print(overestimation_percentage(train_quantification_concentration_label[i], weighted_train_quantification_regression_prediction[i]))


    print("Mix:")
    print('MAPE')
    print(mape(train_quantification_concentration_label[i], new_train_quantification_regression_prediction[i]))
    print('MAE')
    print(mae(train_quantification_concentration_label[i], new_train_quantification_regression_prediction[i]))
    print('MSE')
    print(mse(train_quantification_concentration_label[i], new_train_quantification_regression_prediction[i]))
    print("R2")
    print(r2_score(train_quantification_concentration_label[i], new_train_quantification_regression_prediction[i]))
    print('Accuracy')
    print(accurcay(train_quantification_concentration_label[i], new_train_quantification_regression_prediction[i]))
    print('Over_percentage')
    print(overestimation_percentage(train_quantification_concentration_label[i], new_train_quantification_regression_prediction[i]))


for i in IdentQuantParameters.group_mix_classification_index:
    input_feature = np.array(train_feature[i])[:, quantification_feature_index_single_drug[i]]
    input_label = np.array(train_mix_label[i])
    clf = DecisionTreeClassifier()
    clf.fit(input_feature, input_label)
    dump(clf, classification_DecisionTreeClassifier_path[i])

    random_forest_one = RandomForestClassifier()
    random_forest_one.fit(input_feature, input_label)
    dump(random_forest_one, classification_RandomForestClassifier_path[i])


for i in IdentQuantParameters.group_mlp_mix_classification_index:
    input_feature = np.array(train_feature[i])[:, quantification_feature_index_single_drug[i]]
    input_label = np.array(train_mix_label[i])
    input_dataset = torch.utils.data.TensorDataset(torch.tensor(input_feature, dtype = torch.float32), torch.tensor(input_label.reshape(-1,1)))
    input_loader = torch.utils.data.DataLoader(input_dataset, IdentQuantParameters.batch_size_id, shuffle = True)
    cls_model_build = identification_model.ANN_identification(device, [4, 16, 8], len(quantification_feature_index_single_drug[i]), 3, 'relu')
    cls_model_build.to(device)
    cls_citerion = nn.CrossEntropyLoss()
    cls_optimizer = torch.optim.Adam(cls_model_build.parameters(), lr = IdentQuantParameters.lr_id)
    id_train_acc_list, id_val_acc_list, id_train_loss_list, id_val_loss_list = identification_model.run(cls_model_build, input_loader, input_loader, IdentQuantParameters.num_epochs_mix_id, cls_optimizer, cls_citerion, device, [0], [0])

    torch.save(cls_model_build.state_dict(), classification_mlp_path[i] + ".pth")
    print("Saved PyTorch Model State to model.pth")

    list_file = open(classification_mlp_path[i] + '.pickle','wb')
    pickle.dump((id_train_acc_list, id_val_acc_list, id_train_loss_list, id_val_loss_list), list_file)
    list_file.close()


mlp_classification_mix_model_build_list = []
for i in range(len(predict_drugs)):
    mlp_classification_mix_model_build = identification_model.ANN_identification(device, [4, 16, 8], len(quantification_feature_index_single_drug[i]), 3, 'relu')
    mlp_classification_mix_model_build.load_state_dict(torch.load(classification_mlp_path[i] + '.pth'))
    mlp_classification_mix_model_build_list.append(mlp_classification_mix_model_build)


train_quantification_regression_prediction_mlpcls = [[] for _ in range(len(predict_drugs))]
for i in range(len(predict_drugs)):
    for j in range(len(train_feature[i])):
        input_feature = np.array([train_feature[i][j]])[:, quantification_feature_index_single_drug[i]]
        input_label = np.array([train_mix_label[i][j]])
        input_dataset = torch.utils.data.TensorDataset(torch.tensor(input_feature, dtype = torch.float32), torch.tensor(input_label.reshape(-1,1)))
        input_loader = torch.utils.data.DataLoader(input_dataset)


        for _, (inputs_org, labels_org) in enumerate(input_loader):
            pred = mlp_classification_mix_model_build_list[i](inputs_org)
            softmax_output = torch.nn.functional.softmax(pred, dim=1)
            mix_value_0 = mix_quantification_train(train_quantification_regression_prediction[i][j], weighted_train_quantification_regression_prediction[i][j], train_error_mse[i][j], train_error_weightedmse[i][j])
            regression_1 = train_quantification_regression_prediction[i][j]
            weighted_regression_2 = weighted_train_quantification_regression_prediction[i][j]
            pred_mix = softmax_output[0, 0].item() * mix_value_0 + softmax_output[0, 1].item() * regression_1 + softmax_output[0, 2].item() * weighted_regression_2
            train_quantification_regression_prediction_mlpcls[i].append(pred_mix)


for i in range(len(predict_drugs)):

    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(train_quantification_concentration_label[i], train_quantification_regression_prediction[i], s = 20, label = 'MSE')
    ax1.plot(train_quantification_concentration_label[i], train_quantification_concentration_label[i], linestyle = "dashed", c = "black")


    ax1.scatter(train_quantification_concentration_label[i], train_quantification_regression_prediction_mlpcls[i], s = 20, label = 'Mix')


    ax1.set_title("Drug Quantification Performance (%s)"%(predict_drugs[i]['name']), fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_xlabel(r'True Concentration ($\mu$M)', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_ylabel(r"Predicted Concentration ($\mu$M)", fontsize = IdentQuantParameters.fontsize_fig)
    ax1.legend(fontsize = IdentQuantParameters.fontsize_fig, loc = 2)
    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    plt.show()


classification_mix_model_build_list = []
randomforest_classification_mix_model_build_list = []
for i in range(len(predict_drugs)):
    clf_loaded = load(classification_DecisionTreeClassifier_path[i])
    classification_mix_model_build_list.append(clf_loaded)
    forest_loaded = load(classification_RandomForestClassifier_path[i])
    randomforest_classification_mix_model_build_list.append(forest_loaded)

    print("=" * 50)
    print("Training set Classification:")

    input_feature = np.array(train_feature[i])[:, quantification_feature_index_single_drug[i]]
    input_label = np.array(train_mix_label[i])

    print("Classification-%s" % (predict_drugs[i]['name']))
    output_label = clf_loaded.predict(input_feature)

    precision = precision_score(np.array(input_label), np.array(output_label), average = 'macro')
    print("Precision:", precision)

    recall = recall_score(np.array(input_label), np.array(output_label), average = 'macro')
    print("Recall:", recall)

    f1 = f1_score(np.array(input_label), np.array(output_label), average = 'macro')
    print("F1 score:", f1)

    print("Random_forest")
    output_label = forest_loaded.predict(input_feature)

    precision = precision_score(np.array(input_label), np.array(output_label), average = 'macro')
    print("Precision:", precision)

    recall = recall_score(np.array(input_label), np.array(output_label), average = 'macro')
    print("Recall:", recall)

    f1 = f1_score(np.array(input_label), np.array(output_label), average = 'macro')
    print("F1 score:", f1)


train_error_input_feature = []
train_error_input_feature_two = []
train_error_mse_label = []
train_error_weightedmse_label = []
train_error_label = []
regression_label_index = 0
for i in range(len(predict_drugs)):
    indices = [y for y, x in enumerate(train_mix_label[i]) if x == 0]
    input_feature = [[train_feature[i][y]] for y in indices]
    input_feature_two = [[train_feature[i][y], train_feature[i][y]] for y in indices]
    train_error_input_feature_two.append(input_feature_two)
    train_error_input_feature.append(input_feature)
    input_mse_label = [[abs(train_error_mse[i][y])] for y in indices]
    train_error_mse_label.append(input_mse_label)
    input_weightedmse_label = [[abs(train_error_weightedmse[i][y])] for y in indices]
    train_error_weightedmse_label.append(input_weightedmse_label)
    input_label = [[abs(train_error_mse[i][y]), abs(train_error_weightedmse[i][y])] for y in indices]
    train_error_label.append(input_label)

all_train_error_input_feature = []
all_train_error_input_feature_two = []
all_train_error_mse_label = []
all_train_error_weightedmse_label = []
all_train_error_label = []
for i in range(len(predict_drugs)):
    indices = [y for y in range(len(train_mix_label[i]))]
    input_feature = [[train_feature[i][y]] for y in indices]
    input_feature_two = [[train_feature[i][y], train_feature[i][y]] for y in indices]
    all_train_error_input_feature_two.append(input_feature_two)
    all_train_error_input_feature.append(input_feature)
    input_mse_label = [[abs(train_error_mse[i][y])] for y in indices]
    all_train_error_mse_label.append(input_mse_label)
    input_weightedmse_label = [[abs(train_error_weightedmse[i][y])] for y in indices]
    all_train_error_weightedmse_label.append(input_weightedmse_label)
    input_label = [[abs(train_error_mse[i][y]), abs(train_error_weightedmse[i][y])] for y in indices]
    all_train_error_label.append(input_label)


for i in IdentQuantParameters.group_errorprediction_index:
    input_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(all_train_error_input_feature_two[i]), dtype = torch.float32), torch.tensor(np.array(all_train_error_label[i])))
    train_loader = torch.utils.data.DataLoader(input_dataset, IdentQuantParameters.batch_size_error, shuffle = True)
    model_build = TwoWayANN(device, [[64, 64], [], [256, 256]], [len(quantification_feature_index_single_drug[i]), len(quantification_feature_index_single_drug[i])], [1, 1], 'relu')

    model_build.to(device)
    criterion_error = nn.MSELoss(reduction='mean').to(torch.float64)
    optimizer = torch.optim.Adam(model_build.parameters(), lr = IdentQuantParameters.lr_qu_mse)
    train_loss_list, val_loss_list = identification_model.run_regression(model_build, train_loader, train_loader, IdentQuantParameters.num_epochs_qu_mse, optimizer, criterion_error, device, regression_label_index, loss_type, [quantification_feature_index_single_drug[i], quantification_feature_index_single_drug[i]])

    fig = plt.figure(figsize = (10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(train_loss_list, label = 'Train Loss')
    ax1.plot(val_loss_list, label = 'Val Loss')
    ax1.set_xlabel('Epochs', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_ylabel('Loss', fontsize = IdentQuantParameters.fontsize_fig)
    ax1.set_title('Quantification Loss (%s)'%(predict_drugs[i]['name']), fontsize = IdentQuantParameters.fontsize_fig)

    for label in ax1.xaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontsize(IdentQuantParameters.fontsize_fig)

    ax1.legend()
    plt.show()

    torch.save(model_build.state_dict(), error_prediction_path[i] + ".pth")
    print("Saved PyTorch Model State to model.pth")

    list_file = open(error_prediction_path[i] + '.pickle','wb')
    pickle.dump((train_loss_list, val_loss_list), list_file)
    list_file.close()
