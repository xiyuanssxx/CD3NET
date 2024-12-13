# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Modified from: https://github.com/aliasvishnu/EEGNet
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
""" Feature Extractor """
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch import permute
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
# from mundus.models.builder import BACKBONE

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F





class XAnet(nn.Module):
    def __init__(self, input_channels_A, input_channels_B, dropout_rate=0.5):
        super(XAnet, self).__init__()
        # conv2d
        self.convA = nn.Conv2d(input_channels_A, 64, kernel_size=3, stride=1, padding=1)
        self.convB = nn.Conv2d(input_channels_B, 64, kernel_size=3, stride=1, padding=1)

        # BN
        self.bn_left = nn.BatchNorm2d(input_channels_A)
        self.bn_right = nn.BatchNorm2d(input_channels_B)

        # MP
        self.pool = nn.MaxPool2d(2)

        # conv block
        self.conv1dA = nn.Conv1d(64, input_channels_A, 3, padding=1)
        self.conv1dB = nn.Conv1d(64, input_channels_B, 3, padding=1)

        self.pool1d = nn.MaxPool1d(1)  # No pooling along width

        self.cross_attnA = nn.MultiheadAttention(input_channels_A, num_heads=1)
        self.cross_attnB = nn.MultiheadAttention(input_channels_B, num_heads=1)

        self.dropout = nn.Dropout(dropout_rate)

        # Conv1d layers for reducing sequence length by half
        self.reduce_seq_len_A = nn.Conv1d(input_channels_A, input_channels_A, kernel_size=3, stride=2, padding=1)
        self.reduce_seq_len_B = nn.Conv1d(input_channels_B, input_channels_B, kernel_size=3, stride=2, padding=1)

    def forward(self, A, B, channel_type='standard'):
        bs, channels_A, window_size, window_num = A.size()
        _, channels_B, _, _ = B.size()

        attn_outputs_A = []
        attn_outputs_B = []

        for i in range(window_num):
            # 处理每个 window
            A_window = A[:, :, :, i]  # 取出当前 window
            B_window = B[:, :, :, i]

            A_window = A_window.unsqueeze(-1)  # 变成四维 [batch_size, channels, length, 1]
            B_window = B_window.unsqueeze(-1)
            # Conv2d and BatchNorm
            A_window = F.relu(self.convA(self.bn_left(A_window)))
            B_window = F.relu(self.convB(self.bn_right(B_window)))

            # Reshape
            A_window = A_window.view(A_window.size(0), A_window.size(1), -1)
            B_window = B_window.view(B_window.size(0), B_window.size(1), -1)

            # 1D Conv
            A_ori = F.relu(self.conv1dA(A_window))
            B_ori = F.relu(self.conv1dB(B_window))

            # 1D Pooling
            A_reshape = F.relu(self.conv1dB(A_window))
            B_reshape = F.relu(self.conv1dA(B_window))

            A_ori = self.pool1d(A_ori)
            B_ori = self.pool1d(B_ori)

            # Flatten
            A_flattened = A_ori.flatten(start_dim=2)
            B_flattened = B_ori.flatten(start_dim=2)

            A_flattened = A_flattened.permute(2, 0, 1)  # [seq_len, batch_size, num_channels]
            B_flattened = B_flattened.permute(2, 0, 1)

            A_reshape_flattened = A_reshape.flatten(start_dim=2)
            B_reshape_flattened = B_reshape.flatten(start_dim=2)

            A_reshape_flattened = A_reshape_flattened.permute(2, 0, 1)
            B_reshape_flattened = B_reshape_flattened.permute(2, 0, 1)

            # Cross-attention
            A_attn, _ = self.cross_attnA(A_flattened, B_reshape_flattened, B_reshape_flattened)
            B_attn, _ = self.cross_attnB(B_flattened, A_reshape_flattened, A_reshape_flattened)

            # 恢复形状
            A_attn = A_attn.permute(1, 2, 0).view(bs, channels_A, window_size, 1)
            B_attn = B_attn.permute(1, 2, 0).view(bs, channels_B, window_size, 1)

            # 将结果保存以供之后拼接
            attn_outputs_A.append(A_attn)
            attn_outputs_B.append(B_attn)

        # 在 window_num 维度上进行拼接
        A_concat = torch.cat(attn_outputs_A, dim=3)
        B_concat = torch.cat(attn_outputs_B, dim=3)

        # Concatenate along channel dimension
        x_concat = torch.cat((A_concat, B_concat), dim=1)

        x_concat = x_concat.view(x_concat.size(0), -1)
        x_concat = self.dropout(x_concat)
        x_concat = x_concat.view(bs, channels_A + channels_B, window_size, window_num)

        return x_concat





class EEG_Net_8_Stack(nn.Module):

    def __init__(self, mtl=False):
        super(EEG_Net_8_Stack, self).__init__()
        #cross-attention
        # -1-
        self.XAnet = XAnet(input_channels_A=1, input_channels_B=1)
        self.XAnet2 = XAnet(input_channels_A=2, input_channels_B=1)

        #layer1
        self.Conv2d = nn.Conv2d
        self.conv1 = nn.Conv2d(8, 16, (1, 3), padding=0)
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



    def forward(self, x):

        one_channels = [0]
        two_channels = [1]  #
        three_channels = [2]
        x_one = x[:, :, :, one_channels]
        x_two = x[:, :, :, two_channels]
        x_three = x[:,:,:,three_channels]
        #
        x_one = x_one.permute(0, 3, 2, 1)
        x_two = x_two.permute(0, 3, 2, 1)
        x_three = x_three.permute(0, 3, 2, 1)

        # -1-
        x1 = self.XAnet(x_one, x_two, channel_type='standard')
        x2 = self.XAnet2(x1, x_three, channel_type='standard')
        # -2-
        # x1 = self.XAnet(x_one, x_three, channel_type='standard')
        # x2 = self.XAnet2(x1, x_two, channel_type='standard')
        # -3-
        # x1 = self.XAnet(x_two, x_three, channel_type='standard')
        # x2 = self.XAnet2(x1, x_one, channel_type='standard')

    # EEGNET
        x = x2#16,22,224,8
        x = x.permute(0, 3, 2, 1)   #16, 8, 224, 22

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = x.permute(0, 3, 1, 2)  #16,22,8,224
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pooling2(x)
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pooling3(x)
        # print(x.shape)

        x = x.contiguous().view(-1, 4 * 2 * 14)

        return x


# @BACKBONE.register_obj
def EEG_original(cfg):
    return EEG_Net_8_Stack(**cfg.model.backbone.param)


if __name__ == "__main__":
    # model = EEG_Net_1x()
    model = EEG_Net_8_Stack()
    sample_data = torch.randn(16, 8, 224, 3)
    sample_out = model(sample_data)
    # a(sample_out.size())
