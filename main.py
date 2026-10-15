from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm

from models.CC_Time import CC_Time


import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='CC-Time')

parser.add_argument('--model_id', type=str, default='CC-Time')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./datasets/ETT-small/')
parser.add_argument('--data_path', type=str, default='ETTm1.csv')
parser.add_argument('--data', type=str, default='ett_m_multivariates')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=0)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=100)

parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--train_epochs', type=int, default=20)
parser.add_argument('--lradj', type=str, default='TST')
parser.add_argument('--patience', type=int, default=5)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--llm_d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--llm_d_ff', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mae')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--ze', type=int, default=1)
parser.add_argument('--model', type=str, default='CC-Time')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--cos', type=int, default=0)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=3, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='2', help='device ids of multile gpus')

parser.add_argument('--ts_layer_num', type=int, default=3)
parser.add_argument('--ts_d_model',type=int, default=128)
parser.add_argument('--ts_d_ff',type=int, default=256)
parser.add_argument('--dataset_name',type=str,default='ETTm1')
parser.add_argument('--cross_norm', type=str, default='BatchNorm')
parser.add_argument('--variable_num', type=int, default=2)
parser.add_argument('--bank_dim',type=int, default=64)
parser.add_argument('--alpha',type=float, default=0.4)



args = parser.parse_args()
print(args)

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []



for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, args.seq_len, args.label_len, args.pred_len,
                                                                    args.llm_d_model, args.n_heads, args.e_layers, args.gpt_layers,
                                                                    args.llm_d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)


    if not os.path.exists(path):
        os.makedirs(path)


    if args.freq == 0:
        args.freq = 'h'

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))

    device = torch.device('cuda:0')

    time_now = time.time()
    train_steps = len(train_loader)


    model = CC_Time(args)
    model = nn.DataParallel(model)
    model.cuda()



    print("-------------------------------")
    print(model)
    print("-------------------------------")

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'mae':
        criterion = nn.L1Loss()
    elif args.loss_func == 'smoothl1':
        criterion = nn.SmoothL1Loss()


    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=len(train_loader),
                                        pct_start=0.3,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)
    early_stopping.best_score = None


    train_len = len(train_loader)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        trend_loss_list = []
        sea_loss_list = []
        res_loss_list = []
        epoch_time = time.time()

        alpha = args.alpha

        for i, (batch_x, batch_y) in enumerate(train_loader):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float().cuda()

            output_dic = model(batch_x, batch_y)
            llm_outputs = output_dic["llm_outputs"]
            ts_outputs = output_dic["ts_outputs"]

            loss = 0.6*criterion(ts_outputs, batch_y) + 0.4*criterion(llm_outputs, batch_y)

            train_loss.append(loss.item())


            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
                scheduler.step()

        train_loss = np.average(train_loss)
        _, vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        llm_test_loss, ts_test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))



        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, args)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    mse, mae = test(model, test_data, test_loader, args, device, ii)
    mses.append(mse)
    maes.append(mae)

mses = np.array(mses)
maes = np.array(maes)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))