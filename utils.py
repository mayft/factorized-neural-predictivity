import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import linear_model

import numpy as np
from tqdm import tqdm
from collections import OrderedDict

FEATURES = ('linear','xor_xor_xor')
TRAIN_DATASET_SIZE = 32768
TEST_DATASET_SIZE = 1024
BATCH_SIZE = 512
LAYERS = [64,256,128,64,64,2]
OP_FNS = {'and': np.logical_and, 'or': np.logical_or, 'xor': np.logical_xor}


def make_dataset(features=('features','xor_xor_xor'),input_unit_size=16,num_samples=128,seed=123):
    np.random.seed(seed)

    inputs = np.random.binomial(1, 0.5, size=(num_samples, len(features) * input_unit_size))
    outputs = []

    for i, feature_type in enumerate(features):
        input_unit = inputs[:, i * input_unit_size:(i + 1) * input_unit_size]

        if feature_type == 'linear':
            input_unit[:, :4] = input_unit[:, :1]
            output_unit = input_unit[:, :1]
        else:
            output_unit = np.zeros_like(input_unit[:, :1])
            def evaluate_fn(inputs):
                top, left, right = [OP_FNS[op] for op in feature_type.split('_')]
                return top(left(inputs[0], inputs[1]), right(inputs[2], inputs[3]))

            for number, index in enumerate(np.random.permutation(num_samples)):
                while evaluate_fn(input_unit[index]) != number % 2:
                    input_unit[index] = np.random.binomial(1, 0.5, input_unit_size)
                output_unit[index] = evaluate_fn(input_unit[index])

        outputs.append(output_unit)
    return {'inputs': inputs, 'labels': np.concatenate(outputs, axis=1)}

class Data(Dataset):
    def __init__(self, inputs, labels, device):
        self.data = torch.tensor(inputs,device=device,dtype=torch.float)
        self.target = torch.tensor(labels,device=device,dtype=torch.float)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return (x,y)

    def __len__(self):
        return len(self.data)


def to_tensors(dataset, device='cpu'):
    x = torch.tensor(dataset['inputs'], device=device, dtype=torch.float)
    y = torch.tensor(dataset['labels'], device=device, dtype=torch.float)
    return x,y


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        layers = []
        for i in range(len(layer_sizes) - 2):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            std = np.sqrt(2 / (layer_sizes[i] + layer_sizes[i + 1]))
            torch.nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
            layers.append((f'linear_{i}', layer))
            layers.append((f'leaky_relu_{i}', nn.LeakyReLU()))
        self.model_body = nn.Sequential(OrderedDict(layers))
        self.output = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        std = np.sqrt(2 / (layer_sizes[-2] + layer_sizes[-1]))
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, x):
        x = self.model_body(x)
        out = self.output(x)
        return out, x


def analyze_rep_var_explained(fit_reps, fit_labels, test_reps, test_labels):
    scores = []
    total_var = torch.sum(torch.var(test_reps,dim=0))

    for feat in range(fit_labels.shape[-1]):
        reg = linear_model.LinearRegression()
        reg.fit(fit_labels[:, feat:feat+1], fit_reps)
        scores.append(reg.score(test_labels[:, feat:feat+1], test_reps))

    return scores, total_var

def accuracy(y,labels):
    accuracy = []
    for i in range(len(y[0])):
        accuracy.append(((y[:, i] > 0) == labels[:, i]).float().mean())
    return [((y > 0) == labels).float().mean()]+accuracy



def run(data_features=FEATURES, train_features=None, pretrain=None, units_per_feature=16, max_epochs=50000, train_size=TRAIN_DATASET_SIZE,batch_size=BATCH_SIZE,seeds=1, device='cpu', filename='results', epsilon=0, verbose=0,record_variance=False,early_stopping=True):
    device_ = torch.device(device)
    out_features = len(train_features) if train_features else len(data_features)
    model_layers = [units_per_feature*len(data_features), 256, 128, 64, 64, out_features]

    #output file setup
    file = open(f'results/{filename}.csv',mode='w')
    features = range(len(data_features)) if train_features is None else train_features
    v_file = open(f'results/variances/{filename}_variances.csv',mode='w')
    variance_labels = ['seed','total-variance']+[f'feature{i}-{data_features[i]}' for i in features]
    v_format = ','.join(['%d']+['%.8f']*(len(variance_labels)-1))+'\n'
    v_file.write(','.join(variance_labels)+'\n')
    data_labels = ['seed', 'epoch', 'test-loss','total-acc'] + ['%s_feature%i' % (stat, i) for stat in ('test-acc', 'test-loss') for i in features]
    out_format = ','.join(['%d', '%d'] + ['%.8f'] * (len(data_labels) - 2))+'\n'
    file.write(','.join(data_labels)+'\n')

    models = []
    for seed in range(seeds):
        torch.manual_seed(123 + seed)
        rng = np.random.default_rng(123 + seed)

        train_data = make_dataset(features=data_features, num_samples=train_size, input_unit_size=units_per_feature, seed=123+seed)
        val_data = make_dataset(features=data_features, num_samples=TEST_DATASET_SIZE, input_unit_size=units_per_feature, seed=1234+seed)
        test_data = make_dataset(features=data_features, num_samples=TEST_DATASET_SIZE, input_unit_size=units_per_feature, seed=12345+seed)

        if train_features:
            train_data['labels'] = train_data['labels'][:, train_features]
            val_data['labels'] = val_data['labels'][:, train_features]
            test_data['labels'] = test_data['labels'][:, train_features]

        train_loader = DataLoader(Data(train_data['inputs'], train_data['labels'], device), batch_size=batch_size, shuffle=True)
        x_val = torch.tensor(val_data['inputs'],device=device, dtype=torch.float)
        #y_val = torch.tensor(val_data['labels'],device=device, dtype=torch.float)
        x_test = torch.tensor(test_data['inputs'],device=device, dtype=torch.float)
        y_test = torch.tensor(test_data['labels'],device=device, dtype=torch.float)

        model = MLP(layer_sizes=model_layers).to(device_)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = optim.SGD(model.parameters())

        if pretrain:
            current_feature = pretrain[0]
            feature_number = 1

        vars_ = torch.Tensor()
        for epoch in tqdm(range(max_epochs)):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs,_ = model(inputs)
                if pretrain:
                    outputs=outputs[:,current_feature]
                    labels=labels[:,current_feature]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if (epoch%100 == 0) or (epoch == max_epochs-1):
                with torch.no_grad():
                    model.eval()
                    y_pred, test_reps = model(x_test)
                    test_loss= criterion(y_pred, y_test)
                    test_accuracies = accuracy(y_pred,y_test)
                    test_feature_losses = [criterion(y_pred[:,i], y_test[:,i]) for i in range(len(y_pred[0]))]

                    if record_variance:
                        vp, val_reps = model(x_val)
                        reps = torch.stack([val_reps,test_reps],dim=0)
                        vars_ = torch.cat([vars_,reps])

                outputs= (seed, epoch, test_loss, *test_accuracies, *test_feature_losses)
                file.write(out_format % outputs)
                if pretrain:
                    test_loss = criterion(y_pred[:,current_feature],y_test[:,current_feature])
                if verbose and epoch%1000==0:
                    print(f'epoch {epoch} test loss {test_loss}')
                if test_loss < epsilon:
                    if pretrain and feature_number == 1:
                        if pretrain[1] <= epoch:
                            current_feature = [0, 1]
                            feature_number += 1
                            print(f'switching to feature {current_feature} at epoch {epoch}')
                    elif early_stopping:
                        print(f'stopping early at epoch {epoch}')
                        break
        file.flush()
        if record_variance:
            for i in range(0,len(vars_),2):
                variance_scores, total_variance = analyze_rep_var_explained(vars_[i].cpu(), val_data['labels'], vars_[i+1].cpu(), test_data['labels'])
                v_out = (seed,total_variance,*variance_scores)
                v_file.write(v_format % v_out)
        v_file.flush()
        models.append(model)
        path =f'models/{filename}-{seed}.pt'
        torch.save(model.state_dict(),path)
        print(f'seed {seed} acc: {test_accuracies[0]}, loss: {test_loss}')

    return models

if __name__ == "__main__":

    run_name = 'pretrain_easy-v3'
    print(run_name)
    run(data_features=['linear', 'xor_xor_xor'], seeds=5, max_epochs=30000, epsilon=1e-3,filename=run_name, device='cpu', record_variance=True, pretrain=[0, 5000], units_per_feature=32,early_stopping=False)

    run_name = 'pretrain_hard-v3'
    print(run_name)
    run(data_features=['linear', 'xor_xor_xor'], seeds=5, max_epochs=30000, epsilon=1e-3,filename=run_name, device='cpu', record_variance=True, pretrain=[1, 25000], units_per_feature=32,early_stopping=False)

    #run_name = 'multiple_easy-v1'
    #print(run_name)
    #run(data_features=['linear', 'linear'], seeds=5, max_epochs=200000, epsilon=1e-3,filename=run_name, device='cpu',record_variance=True)

    #run_name = 'multiple_hard-v1'
    #print(run_name)
    #run(data_features=['xor_xor_xor', 'xor_xor_xor'], seeds=5, max_epochs=200000, epsilon=1e-3,filename=run_name, device='cpu',record_variance=True)