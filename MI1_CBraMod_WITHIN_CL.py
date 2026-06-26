import time
start_time = time.time()
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import json
import os
import random
import torch
import torch.nn as nn
from models.cbramod import CBraMod
from einops.layers.torch import Rearrange

seed = 1  # set random seed
# CUDA_VISIBLE_DEVICES=3 python -u MI2_EEGNET_CROSS_EA.py
# nohup bash -c "CUDA_VISIBLE_DEVICES=3 python -u MI2_EEGNET_CROSS_EA.py" > MI2_EEGNET_CROSS_EA.log 2>&1 &

print(f"Seed:{seed} GPU:{os.environ['CUDA_VISIBLE_DEVICES']}")

def set_random_seed(seed: int) -> None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

set_random_seed(seed)

# EA (Euclidean Alignment)
from scipy.linalg import fractional_matrix_power
def EA(x):
        cov = np.zeros((x.shape[0], 59, 59))
        for i in range(x.shape[0]):
                cov[i] = np.cov(x[i])
        refEA = np.mean(cov, 0)
        sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(59)
        XEA = np.zeros(x.shape)
        for i in range(x.shape[0]):
                XEA[i] = np.dot(sqrtRefEA, x[i])
        return XEA

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import math


class EEGNet(nn.Module):
    def __init__(self, in_chan=0, fc_num=0, out_chann=0):
        super(EEGNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, in_chan), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        self.fc1 = nn.Linear(fc_num, out_chann)

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(x.size()[0], -1)
        # print('x', x.shape)
        x = self.fc1(x)
        return x


def prepare_data(X_train, y_train, X_test, y_test):
    """
    Data preprocessing function
    Convert numpy arrays to PyTorch tensors and add necessary dimensions
    """
    # Add channel dimension (trials, 1, channels, timepoints)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # Convert labels to LongTensor
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_test, y_test


def train_model(model_name, model, classifier, train_loader, criterion, optimizer, epochs=1):
    """
    Model training function
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # print('data', data.shape)
            if model_name != 'EEGNet':
                data = data.permute(0, 1, 3, 2)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            target = target.to(device)
            output = classifier(model(data))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Print loss every epoch
        if (epoch + 1) % 1 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')




def test_model(model_name, model, classifier, test_loader):
    """
    Model testing function
    Returns test accuracy
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in test_loader:
            if model_name != 'EEGNet':
                data = data.permute(0, 1, 3, 2)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data = data.to(device)
            output = classifier(model(data))
            _, predicted = torch.max(output.data, 1)
            y_true.extend(target.numpy())
            y_pred.extend(predicted.detach().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f'Test Accuracy: {acc * 100:.2f}%')
    return acc


def load_all_subjects_data():
    """
    Load data for all subjects
    Returns a list containing data for all subjects
    """
    all_subjects_data = []

    # Iterate through all subjects (A1.mat to A9.mat)
    for i in range(1, 8):
        # Load .mat file
        data = loadmat(f'/data2/hywu/LLaMA-Factory/MI1/A{i}.mat')
        x = data['x']  # EEG data (22 channels x 750 timepoints x 144 trials)
        y = data['y'].flatten()  # Labels (144,)

        # Transpose from (channels, time, trials) to (trials, channels, time)
        x = np.transpose(x, (2, 0, 1))  # (144, 22, 750)

        # Store subject data
        subject_data = {
            'x': x,  # EEG data (144, 22, 750)
            'y': y,  # Labels (144,)
            'subject_id': i  # Subject ID
        }
        all_subjects_data.append(subject_data)

    return all_subjects_data


import numpy as np


def sliding_window_augmentation(X, y, window_length=250, step_size=10):
    """
    Data augmentation via sliding window

    Args:
        X: Input data, shape (n_trials, n_channels, n_timesteps)
        window_length: Window length, default 250
        step_size: Step size, default 10

    Returns:
        X_augmented: Augmented data, shape (n_augmented, n_channels, window_length)
    """
    n_trials, n_channels, n_timesteps = X.shape
    augmented_samples = []
    augmented_labels = []

    for trial in range(n_trials):
        start_idx = 0
        # Calculate number of windows that can be extracted from current trial
        while start_idx + window_length <= n_timesteps:
            # Extract window
            window_data = X[trial, :, start_idx:start_idx + window_length]
            augmented_samples.append(window_data)
            augmented_labels.append(y[trial])
            start_idx += step_size

    # Stack all windows along dimension 0
    X_augmented = np.stack(augmented_samples, axis=0)
    y_augmented = np.stack(augmented_labels, axis=0)
    return X_augmented, y_augmented



def prepare_within_subject_data(all_subjects_data, test_subject_idx):
    """
    Prepare within-subject training and testing data
    test_subject_idx: Index of subject used as test set (0-8)
    """
    # Initialize train and test sets
    X_train_list, y_train_list = [], []
    X_test, y_test = None, None

    for idx, subject_data in enumerate(all_subjects_data):
        x = subject_data['x']  # (144, 22, 750)
        y = subject_data['y']  # (144,)

        if idx == test_subject_idx:
            # Current subject as test set
            # X_test = x
            # y_test = y

            # Calculate 80% split point
            n_trials = x.shape[0]
            split_idx = int(0.8 * n_trials)

            # Split data (first 80% train, last 20% test)
            X_train = x[:split_idx, :, :]  # (n_trials, 22, 750)
            y_train = y[:split_idx]
            X_test = x[split_idx:, :, :]  # (22, 750, 29)
            y_test = y[split_idx:]
        else:
            pass

    X_augmented, y_augmented = sliding_window_augmentation(X_train, y_train, window_length=200, step_size=3000)

    print(f"Original data shape: {X_train.shape}")
    X_train = X_augmented
    X_train = EA(X_train)
    y_train = y_augmented
    X_test = X_test[:, :, 0:200]
    X_test = EA(X_test)
    print(f"Augmented data shape: {X_augmented.shape}")
    print('y_train', y_train.shape)

    # Concatenate training data
    # X_train = np.concatenate(X_train_list, axis=0)  # (n_trials, 22, 750)
    # y_train = np.concatenate(y_train_list, axis=0)  # (n_trials,)

    # Adjust data dimensions to match EEGNet input requirements
    # Original: (trials, channels, timepoints)
    # Target: (trials, 1, timepoints, channels)
    X_train = X_train.transpose(0, 2, 1)  # (n_trials, 750, 22)
    X_train = np.expand_dims(X_train, axis=1)  # (n_trials, 1, 750, 22)

    X_test = X_test.transpose(0, 2, 1)  # (n_trials, 750, 22)
    X_test = np.expand_dims(X_test, axis=1)  # (n_trials, 1, 750, 22)

    print('X_train, y_train, X_test, y_test', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test


model_name = 'CBraMod'
print('Model:', model_name)


if model_name == 'EEGNet':
    model = EEGNet(in_chan=59, fc_num=120, out_chann=2)  # 750 corresponds to 376, 250 corresponds to 120
elif model_name == 'CBraMod':  # Modify settings according to the dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CBraMod().to(device)
    model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth', map_location=device))
    model.proj_out = nn.Identity()
    classifier = nn.Sequential(
        Rearrange('b c s p -> b (c s p)'),
        nn.Linear(59 * 1 * 200, 1 * 200),
        nn.ELU(),
        nn.Dropout(0.1),
        nn.Linear(1 * 200, 200),
        nn.ELU(),
        nn.Dropout(0.1),
        nn.Linear(200, 2),
    ).to(device)


frozen = False

backbone_params = []
other_params = []
for name, param in model.named_parameters():
    backbone_params.append(param)

    if frozen:  # Optional backbone freezing
        param.requires_grad = False
    else:
        param.requires_grad = True

for name, param in classifier.named_parameters():
    other_params.append(param)

optimizer_type = 'AdamW'
multi_lr = True

# Multi-learning-rate optimizer configuration
if optimizer_type == 'AdamW':
    if multi_lr:  # Multi-learning-rate setting
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 0.0001},
            {'params': other_params, 'lr': 0.001 * (64/256) ** 0.5}
        ])
    else:  # Single learning rate
        pass
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
        #                                    weight_decay=self.params.weight_decay)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Main program
if __name__ == "__main__":
    # 1. Load data for all subjects
    print("Loading data for all subjects...")
    all_subjects_data = load_all_subjects_data()
    num_subjects = len(all_subjects_data)
    print(f"Successfully loaded data for {num_subjects} subjects")

    # Store test accuracy for each subject
    subject_accuracies = []
    test_loader_save = []

    # 2. Perform leave-one-subject-out cross-validation
    for test_idx in range(num_subjects):
        print(f"\n{'=' * 50}")
        print(f"Running cross-validation fold {test_idx + 1}/{num_subjects}")
        print(f"Test subject: A{test_idx + 1}")
        print(f"{'=' * 50}")

        # Prepare data for current cross-validation split
        X_train, y_train, X_test, y_test = prepare_within_subject_data(
            all_subjects_data, test_idx
        )

        # Print data shape information
        print(f"Training set shape: {X_train.shape}, Label shape: {y_train.shape}")
        print(f"Test set shape: {X_test.shape}, Label shape: {y_test.shape}")

        # Data preprocessing
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data(
            X_train, y_train, X_test, y_test
        )

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_loader_save.append(test_loader)

        # Train model (reduce epochs to speed up cross-validation)
        print("Starting model training...")
        train_model(model_name, model, classifier, train_loader, criterion, optimizer, epochs=50)

        # Test model
        print("Starting model testing...")
        accuracy = test_model(model_name, model, classifier, test_loader)
        subject_accuracies.append(accuracy)

        print(f"Test accuracy for subject A{test_idx + 1}: {accuracy * 100:.2f}%")

        end_time = time.time()
        print(f"Total running time: {end_time - start_time:.2f} seconds")

    # 3. Calculate and display average accuracy across all subjects
    print(f"\n{'=' * 60}")
    print("Within-subject test results summary")
    print(f"{'=' * 60}")

    # Print best accuracy for each subject
    for i, acc in enumerate(subject_accuracies):
        print(f"Subject A{i + 1}: {acc * 100:.2f}%")

    # Calculate mean accuracy and standard deviation
    mean_accuracy = np.mean(subject_accuracies) * 100
    std_accuracy = np.std(subject_accuracies) * 100

    print(f"\nWithin-subject test average accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

    subject_accuracies_cl = []
    for i in range(len(test_loader_save)):
        accuracy = test_model(model_name, model, classifier, test_loader_save[i])
        subject_accuracies_cl.append(accuracy)

    # Print accuracy for each subject during continual learning
    for j, acc in enumerate(subject_accuracies_cl):
        print(f"Subject A{j + 1}: {acc * 100:.2f}%")

    mean_accuracy = np.mean(subject_accuracies_cl) * 100
    std_accuracy = np.std(subject_accuracies_cl) * 100

    print(f"\nWithin-subject test average accuracy (continual learning): {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

    # BWT calculation

    # Calculate pairwise differences
    differences = [a - b for a, b in zip(subject_accuracies_cl, subject_accuracies)]

    # Calculate average of differences
    BWT = (sum(differences) / (len(differences) - 1)) * 100

    # print("Differences list:", differences)
    # print("Average difference BWT:", BWT)
    print('Current dataset: MI1')
    print("Current model:", model_name)
    print(f"\nAverage difference BWT: {BWT:.2f}%")


