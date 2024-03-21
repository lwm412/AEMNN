import argparse
import math
import time

import numpy as np

from utils import *
from trainer import Trainer
import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

parser = argparse.ArgumentParser(description='PyTorch Time series classification')
parser.add_argument('--data', type=str, default='./dataset/raw_dataset_div4.csv',
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='./save/',
                    help='path to save the final model')

parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--normalizer', type=int, default=0) # 0
parser.add_argument('--column_wise', type=eval, default=False)
parser.add_argument('--device', type=str, default='cuda:0', help='')

parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--linear_dim', type=float, default=128)
parser.add_argument('--linear_dim1', type=float, default=32)
parser.add_argument('--linear_dim2', type=float, default=32)
parser.add_argument('--linear_dim3', type=float, default=32)
parser.add_argument('--encoder_dim', type=float, default=128)
parser.add_argument('--classifier_dim', type=float, default=64)

parser.add_argument('--batch_size', type=int,default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=600, help='')
parser.add_argument('--patience', type=int, default=30, help=" ")


parser.add_argument('--lr_decay', type=eval, default=False)
parser.add_argument("--lr_decay_step", type=str, default="15,40,70,105,145", help="在几个epoch进行初始学习率衰减")
parser.add_argument("--lr_decay_rate", type=float, default=0.3, help="学习率衰减率")
parser.add_argument('--max_grad_norm', type=float, default=5, help="梯度阈值")


parser.add_argument('--log_file', default="./log/log", help='log file')
parser.add_argument('--expid', type=int, default=1, help='实验ID')
parser.add_argument('--print_every', type=int, default=50, help='几个batch输出训练损失')


args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(3)

log = open(args.log_file, 'w')
log_string(log, str(args))


def out_to_csv(truth, pred):
    with open("pred/pred.csv", "w") as f:
        for i in range(int(truth.shape[0])):
            t = truth[i].astype(int)
            p = pred[i].astype(int)
            line = str(t) + "," + str(p) + "\n"
            f.write(line)
        f.close()


def main():

    dataloader = load_dataset(dataset_dir=args.data,
                              normalizer=args.normalizer,
                              batch_size=args.batch_size,
                              valid_batch_size=args.batch_size,
                              test_batch_size=args.batch_size,
                              column_wise=args.column_wise,
                              train_ratio=0.8,
                              test_ratio=0.2)

    scaler = dataloader['scaler']

    log_string(log, 'loading data...')
    log_string(log, f'trainX: {torch.tensor(dataloader["train_loader"].xs).shape}\t\t '
                    f'trainY: {torch.tensor(dataloader["train_loader"].ys).shape}')
    log_string(log, f'valX:   {torch.tensor(dataloader["val_loader"].xs).shape}\t\t'
                    f'valY:   {torch.tensor(dataloader["val_loader"].ys).shape}')
    log_string(log, f'testX:   {torch.tensor(dataloader["test_loader"].xs).shape}\t\t'
                    f'testY:   {torch.tensor(dataloader["test_loader"].ys).shape}')
    log_string(log, 'data loaded!')

    engine = Trainer(args=args,
                     scaler=scaler,
                     device=device,
                     lr_decay=args.lr_decay,
                     log=log,
                     max_grad_norm=args.max_grad_norm,
                     linear_dim1=args.linear_dim1,
                     linear_dim2=args.linear_dim2,
                     linear_dim3=args.linear_dim3,
                     linear_dim=args.linear_dim,
                     encoder_dim=args.encoder_dim,
                     classifier_dim=args.classifier_dim)
    # 开始训练
    log_string(log, 'compiling model...')
    his_loss = []
    val_time = []
    train_time = []

    wait = 0
    val_acc_min = float('-inf')
    best_model_wts = None

    for i in tqdm.tqdm(range(1, args.epochs + 1)):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {i:04d}')
            break

        train_loss = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()

        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            # [B, T, C]

            trainy = torch.Tensor(y[:, :]).to(device)
            # [B, 1]

            loss = engine.train(trainx, trainy)
            train_loss.append(loss)

            if iter % args.print_every == 0:
                logs = 'Iter: {:03d}, Train Loss: {:.4f}, lr: {}'
                print(logs.format(iter, train_loss[-1],
                                  engine.optimizer.param_groups[0]['lr']), flush=True)

        if args.lr_decay:
            engine.lr_scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)


        valid_loss = []
        valid_pred = []
        valid_correct = []
        valid_num_instance = []
        valid_conf_mat = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device)
            # [B, T, C]

            valy = torch.Tensor(y[:, :]).to(device)
            # [B, 1]

            loss, pred, correct, num_instance, conf_mat = engine.evel(valx, valy)
            valid_loss.append(loss)
            valid_correct.append(correct)
            valid_num_instance.append(num_instance)
            valid_conf_mat.append(conf_mat)

        valid_acc = 100. * np.sum(valid_correct) / np.sum(valid_num_instance)
        valid_loss = np.mean(valid_loss)

        s2 = time.time()
        logs = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        log_string(log, logs.format(i, (s2-s1)))

        val_time.append(s2 - s1)


        logs = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid ACC: {:.4f}, Training Time: {:.4f}/epoch'
        log_string(log, logs.format(i, np.mean(train_loss), valid_loss, valid_acc, (t2 - t1)))

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        if valid_acc >= val_acc_min:
            log_string(
                log,
                f'val acc increase from {val_acc_min:.4f} to {valid_acc:.4f}, '
                f'save model to {args.save + "exp_" + str(args.expid) + "_" + str(round(valid_acc, 2)) + "_best_model.pth"}'
            )
            wait = 0
            val_acc_min = valid_acc
            best_model_wts = engine.model.state_dict()
            torch.save(best_model_wts,
                       args.save + "exp_" + str(args.expid) + "_" + str(round(val_acc_min, 2)) + "_best_model.pth")
        else:
            wait += 1

    log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string(log, "Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    log_string(log, "Training finished")
    log_string(log, "The valid acc on best model is " + str(round(val_acc_min, 4)))
    log_string(log, "Start evaluating, result:")

    # 测试
    engine.model.load_state_dict(
        torch.load(args.save + "exp_" + str(args.expid) + "_" + str(round(val_acc_min, 2)) + "_best_model.pth"))
    torch.save(engine.model, args.save + "exp_" + str(args.expid) + "_best_model.pth")


    outputs = []
    realy = torch.Tensor(dataloader['y_test'][:, :]).to(device)
    # B, 1

    for iter, (x, y) in tqdm.tqdm(enumerate(dataloader['test_loader'].get_iterator())):
        testx = torch.Tensor(x).to(device)
        with torch.no_grad():
            preds = engine.model(testx)
            # [B, 1]
            outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  # 在做batch的时候，可能会padding出新的sample以满足batch_size的要求

    realy = realy.cpu().numpy().squeeze()
    pred = yhat.cpu().numpy().squeeze()

    realy_class = np.argmax(realy, axis=1)
    pred_class = np.argmax(pred, axis=1)

    # out_to_csv(realy_class, pred_class)

    result = classification_report(realy_class, pred_class, digits=4)

    # precision = precision_score(realy_class, pred_class, average='binary')
    # recall = recall_score(realy_class, pred_class, average='binary')
    # f1 = f1_score(realy_class, pred_class, average='binary')

    # log_string(log, "precision: {}, recall: {}, f1: {}".format(precision, recall, f1))

    log_string(log, result)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
    log.close()

