import torch
import torch.nn as nn

class ALCNN(nn.Module):

    def __init__(self, NumClassesOut, N_time, N_Features, device, N_LSTM_Out=128, N_LSTM_layers=1, Conv1_NF=128, Conv2_NF=256,
                 Conv3_NF=128, lstmDropP=0.8, FC_DropP=0.3):

        super(ALCNN, self).__init__()

        self.device = device
        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.channel_dim = 14

        self.attentionLayer = nn.TransformerEncoderLayer(d_model=self.channel_dim, nhead=1, batch_first=True)
        self.attentionEncoder = nn.TransformerEncoder(encoder_layer=self.attentionLayer, num_layers=3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.channel_dim))

        self.C1 = nn.Conv1d(self.channel_dim, self.Conv1_NF, 8)
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 5)
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.channel_dim, self.NumClassesOut)


    def forward(self, x):
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time samples
        #                                         F = features
        # [128, 28, 14]
        # Transformer [128, 28, 14]
        # 14  ->  mlp(1)   mlp(1)   mlp(12) -> cat[]   ->>  relu tanh  LayerNorm((96))

        x1 = torch.cat([x, self.cls_token.repeat(x.size(0), 1, 1)], dim=1)
        x1 = self.attentionEncoder(x1)
        x1 = x1[:, -1, :]

        x2 = x.transpose(2, 1)
        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.FC(x_all)
        # [128, 2]
        return x_out
