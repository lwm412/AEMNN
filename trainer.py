import torch
import torch.optim as optim
from model.model import ALCNN
from model.LSTM import LSTMClassifier
from model.FCNLSTM import FCN_model
from model.Resnet import ResNet1D
from model.MACNN import MACNN
from model.VGG import VGG
from model.PatchTsT import PatchTSTModel
import torch.nn.functional as F
import torch.nn as nn
from utils import *



class Trainer():

    def __init__(self, args, scaler, device, lr_decay, log, max_grad_norm,
                 linear_dim1, linear_dim2, linear_dim3, linear_dim, encoder_dim, classifier_dim):
        super(Trainer, self).__init__()

        self.model = ALCNN(
            NumClassesOut=2,
            N_time=28,
            N_Features=14,
            device=device,
            N_LSTM_layers=3
        )

        # self.model = LSTMClassifier(
        #     in_dim=14,
        #     hidden_dim=100,
        #     num_layers=3,
        #     dropout=0.8,
        #     bidirectional=True,
        #     num_classes=2,
        #     batch_size=args.batch_size
        # )

        # self.model = FCN_model(
        #     NumClassesOut=2,
        #     N_time=28,
        #     N_Features=14,
        #     device=device,
        #     N_LSTM_layers=3
        # )

        # self.model = ResNet1D(
        #     in_channels=14,
        #     base_filters=128,
        #     kernel_size=7,
        #     stride=2,
        #     groups=32,
        #     n_block=64,
        #     n_classes=2,
        #     downsample_gap=6,
        #     increasefilter_gap=12,
        #     use_do=True
        # )

        # self.model = MACNN(
        #     in_channels=14,
        #     channels=100,
        #     num_classes=2
        # )

        # self.model = VGG(
        #     num_classes=2
        # )

        # self.model = PatchTSTModel()


        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(device)

        self.model_parameters_init()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        if lr_decay:
            log_string(log, 'Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                     milestones=lr_decay_steps,
                                                                     gamma=args.lr_decay_rate)

        self.loss = torch.nn.CrossEntropyLoss()
        self.scaler = scaler
        self.clip = None


        log_string(log, "模型可训练参数: {:,}".format(count_parameters(self.model)))
        log_string(log, 'GPU使用情况:{:,}'.format(torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0))

    def model_parameters_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.0003)
            else:
                nn.init.uniform_(p)

    def train(self, input, real_val):
        """
        :param input: shape [B, len, C]
        :param real_val: shape [B, 1]
        """

        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(input)

        # predict = self.scaler.inverse_transform(output)
        predict = output

        loss = self.loss(predict, real_val)
        # loss = self.loss(torch.sigmoid(predict), real_val)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        return loss.item()

    def evel(self, input, real_val):
        """
        :param input: shape [B, len, C]
        :param real_val: shape [B, 1]
        """
        self.model.eval()

        output = self.model(input)

        # predict = self.scaler.inverse_transform(output)
        predict = output

        loss = self.loss(predict, real_val).item()
        # loss = self.loss(torch.sigmoid(predict), real_val).item()
        pred, correct, num_instance, conf_mat = get_acc(predict, real_val)


        return loss, pred, correct, num_instance, conf_mat
