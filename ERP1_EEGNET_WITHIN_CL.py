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

seed = 1  # Set random seed
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

# EA
from scipy.linalg import fractional_matrix_power


def EA(x):
    cov = np.zeros((x.shape[0], 8, 8))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(8)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA

from DBConformer import DBConformer

# Create argument object
class Args:
    def __init__(self):
        self.data_name = 'MI1-7'
        self.chn = 22
        self.time_sample_num = 1000
        self.patch_size = 125
        self.class_num = 2
        self.emb_size = 40
        self.spa_dim = 16
        self.tem_depth = 5
        self.chn_depth = 5
        self.branch = 'all'
        self.gate_flag = False
        self.posemb_flag = True
        self.chn_atten_flag = True
        self.seed = 42
        self.batch_size = 32
        self.epochs = 50
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.log_interval = 10
        self.num_workers = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import math

# 1. Convolution module (for local feature extraction)
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),  # Temporal convolution
            nn.Conv2d(40, 40, (22, 1), (1, 1)),  # Spatial convolution
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # Pooling, slice along temporal dimension to get "patches"
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # Adjust channel dimension
            Rearrange('b e (h) (w) -> b (h w) e'),  # Rearrange to sequence form
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x

# 2. Multi-head attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # Attention energy
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)  # Attention weights
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)  # Weighted sum
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# 3. Residual connection module
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# 4. Feed-forward network module
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# 5. Transformer encoder block
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

# 6. Transformer encoder (multi-layer stacking)
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

# 7. Classification head
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # Global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        # Fully connected classifier (backup, actually uses clshead)
        self.fc = nn.Sequential(
            nn.Linear(1680, 256),  # needs to be modified
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)  # Flatten
        #print('x',x.shape)
        out = self.fc(x)  # Classify through fully connected layer
        return out  # Return features and classification results

# 8. Complete Conformer model
class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),      # 1. Convolution for local feature extraction
            TransformerEncoder(depth, emb_size),  # 2. Transformer encoder
            ClassificationHead(emb_size, n_classes)  # 3. Classification head
        )


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
        # print('x',x.shape)
        x = self.fc1(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),  # 10 -> 5
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1, bias=False),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes),
                                 nn.Dropout(p=0.2))  ####################### classifier
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)
        output = self.clf(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, fc_ch, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = input_ch

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 13), stride=1, padding=(6, 7)),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))

        self.fc = nn.Linear(fc_ch, n_classes)

    def forward(self, x):
        # x = x.permute(0, 1, 3, 2)  # disable when running pretrain
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 35), (1, 7))
        x = torch.log(x)
        x = x.flatten(1)
        x = torch.nn.functional.dropout(x)
        # print('x',x.shape)
        x = self.fc(x)
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


def train_model(model_name, model, train_loader, criterion, optimizer, epochs=1):
    """
    Model training function
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if model_name != 'EEGNet':
                data = data.permute(0, 1, 3, 2)
            if model_name == 'DBConformer':
                _, output = model(data)
            else:
                output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Print loss every 10 epochs
        if (epoch + 1) % 1 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')


from sklearn.metrics import accuracy_score, confusion_matrix


def test_model(model_name, model, test_loader):
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
            if model_name == 'DBConformer':
                _, output = model(data)
            else:
                output = model(data)
            _, predicted = torch.max(output.data, 1)
            y_true.extend(target.numpy())
            y_pred.extend(predicted.numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f'Test Accuracy: {acc * 100:.2f}%')

    # Calculate Balanced Classification Accuracy (BCA)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    bca = per_class_acc.mean()

    print(f'Balanced Classification Accuracy (BCA): {bca * 100:.2f}%')
    return bca


def load_all_subjects_data():
    """
    Load data for all subjects
    Returns a list containing all subjects' data
    """
    all_subjects_data = []

    data = loadmat(f'RSVP/RSVP.mat')
    # print(data)
    x = data['xAll']  # EEG data (22 channels x 750 timepoints x 144 trials)
    y = data['yAll'].flatten()  # Labels (144,)
    print('x', x.shape)
    print('y', y.shape)

    trial_lengths = [398, 389, 433, 398, 440, 399, 368, 487, 463, 565, 443]

    # Split x along the last dimension
    start = 0
    i = 0
    for length in trial_lengths:
        end = start + length
        # Get x slice corresponding to each trial (8, 45, length)
        x_trial = x[:, :, start:end]
        # Slice y
        y_trial = y[start:end]
        x_trial = np.transpose(x_trial, (2, 0, 1))  # (144, 22, 750)
        print('x', x.shape)
        # Store subject data
        subject_data = {
            'x': x_trial,  # EEG data (144, 22, 750)
            'y': y_trial,  # Labels (144,)
            'subject_id': i  # Subject ID
        }
        all_subjects_data.append(subject_data)
        i += 1
        start = end

    return all_subjects_data


import numpy as np


def sliding_window_augmentation(X, y, window_length=250, step_size=10):
    """
    Data augmentation via sliding window

    Parameters:
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
    Prepare within-subject train and test data
    test_subject_idx: Index of subject used as test set (0-8)
    """
    # Initialize train and test sets
    X_train_list, y_train_list = [], []
    X_test, y_test = None, None

    for idx, subject_data in enumerate(all_subjects_data):
        x = subject_data['x']  # (144, 22, 750)
        y = subject_data['y']  # (144,)

        if idx == test_subject_idx:
            # Current subject used as test set
            # X_test = x
            # y_test = y

            # Calculate 80% split point
            n_trials = x.shape[0]
            split_idx = int(0.8 * n_trials)

            # Split data (first 80% training, last 20% testing)
            X_train = x[:int(0.8 * n_trials), :, :]  # (n_trials, 22, 750)
            y_train = y[:int(0.8 * n_trials)]
            X_test = x[split_idx:, :, :]  # (22, 750, 29)
            y_test = y[split_idx:]
        else:
            pass

    # X_augmented,y_augmented = sliding_window_augmentation(X_train,y_train, window_length=260, step_size=30)

    print(f"Original data shape: {X_train.shape}")
    # X_train = X_augmented
    # y_train = y_augmented
    X_test = X_test[:, :, 0:260]
    # X_train = EA(X_train)
    # X_test = EA(X_test)
    # print(f"Augmented data shape: {X_augmented.shape}")
    print('y_train', y_train.shape)

    # Concatenate training data
    # X_train = np.concatenate(X_train_list, axis=0)  # (n_trials, 22, 750)
    # y_train = np.concatenate(y_train_list, axis=0)  # (n_trials,)

    # Adjust data dimensions to match EEGNet input requirements
    # Original dimensions: (trials, channels, timepoints)
    # Target dimensions: (trials, 1, timepoints, channels)
    X_train = X_train.transpose(0, 2, 1)  # (n_trials, 750, 22)
    X_train = np.expand_dims(X_train, axis=1)  # (n_trials, 1, 750, 22)

    X_test = X_test.transpose(0, 2, 1)  # (n_trials, 750, 22)
    X_test = np.expand_dims(X_test, axis=1)  # (n_trials, 1, 750, 22)

    print('X_train, y_train, X_test, y_test', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test


model_name = 'DBConformer'
print('model', model_name)


if model_name == 'DeepConvNet':
    model = DeepConvNet(2, 8, 260)  # 750 is the timepoint length
elif model_name == 'ShallowConvNet':
    model = ShallowConvNet(2, 8, 208)  # 9152 needs to be calculated empirically
elif model_name == 'EEGNet':
    model = EEGNet(in_chan=8, fc_num=24, out_chann=2)
elif model_name == 'EEGConformer':
    model = Conformer(emb_size=40, depth=6, n_classes=2)  # Need to adjust fully connected layer definition in model definition
elif model_name == 'DBConformer':
    args = Args()
    args.chn = 8  # 22 channels
    args.time_sample_num = 45  # 1000 timepoints
    args.patch_size = 45  # 125 points as one patch
    args.class_num = 2  # 4-class task
    # Model instantiation
    model = DBConformer(
        args,
        emb_size=40,  # Medium dimension, balancing performance and parameter count
        tem_depth=5,  # 5 temporal Transformer layers
        chn_depth=5,  # 5 spatial Transformer layers
        chn=args.chn,  # 22 channels
        n_classes=args.class_num  # 4 outputs
    )

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
        print(f"Cross-validation round {test_idx + 1}/{num_subjects}")
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

        # Initialize model
        # Note: fc_num needs to be calculated based on input data shape, using previous value 376 here

        # Train model (reduce epochs to speed up cross-validation)
        print("Starting model training...")
        train_model(model_name, model, train_loader, criterion, optimizer, epochs=50)

        # Test model
        print("Starting model testing...")
        accuracy = test_model(model_name, model, test_loader)
        subject_accuracies.append(accuracy)

        print(f"Test accuracy for subject A{test_idx + 1}: {accuracy * 100:.2f}%")

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
        accuracy = test_model(model_name, model, test_loader_save[i])
        subject_accuracies_cl.append(accuracy)

    # Print accuracy for each subject during continual learning process
    for j, acc in enumerate(subject_accuracies_cl):
        print(f"Subject A{j + 1}: {acc * 100:.2f}%")

    mean_accuracy = np.mean(subject_accuracies_cl) * 100
    std_accuracy = np.std(subject_accuracies_cl) * 100

    print(f"\nWithin-subject test average accuracy - Continual Learning: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")

    # BWT calculation

    # Calculate difference at corresponding positions
    differences = [a - b for a, b in zip(subject_accuracies_cl, subject_accuracies)]

    # Calculate average of differences
    BWT = (sum(differences) / (len(differences) - 1)) * 100

    # print("Difference list:", differences)
    # print("Average difference BWT:", BWT)
    print('Current dataset RSVP')
    print("Current model", model_name)
    print(f"\nAverage difference BWT: {BWT:.2f}%")

end_time = time.time()
print(f"Total running time: {end_time - start_time:.2f} seconds")