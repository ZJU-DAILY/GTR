import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data.dataset
from torch.utils.data import Dataset
import numpy as np
import pandas
import math
import json
import random
from tqdm import tqdm

random.seed(5000)
np.random.seed(5000)

class update_model_Dataset(Dataset):  # similarity
    def __init__(self, args):
        with open(args.pre_path + args.settings) as f:
            setting = json.load(f)

        with open(args.pre_path + args.vocab_file) as f:
            vocab = json.load(f)

        super().__init__()
        self.input_token = []
        self.input_week = []
        self.input_day = []
        self.trj_id = []
        self.attention_mask = []

        self.data = []
        self.max_len = args.max_len
        self.vocab_size = setting['vocab_size']
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4

    def load(self, inputfile, dt_type, device):
        file1 = pd.read_csv(inputfile)
        trjs = file1['rel_token']
        times = file1['timestamp']
        length = len(trjs)

        for i in tqdm((range((int)(length / 10))), desc='data loading', ncols=80):
            trj, tm = eval(trjs[i]), eval(times[i])
            weekday_list, minute_list = [0], [0]

            for j in range(len(tm)):
                dt = datetime.datetime.fromtimestamp(tm[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)

            la = len(trj)
            token = []
            week = [0]
            day = [0]
            if dt_type == 'query':
                for j in range(la):
                    if j % 2 == 0:
                        token.append(trj[j])
                        week.append(weekday_list[j])
                        day.append(minute_list[j])
            else:
                for j in range(la):
                    if j % 2 == 1:
                        token.append(trj[j])
                        week.append(weekday_list[j])
                        day.append(minute_list[j])

            last = self.max_len - len(token)
            for j in range(last - 1):
                token.append(self.PAD)
                week.append(0)
                day.append(0)
            for j in range(self.max_len - la - 1):
                trj.append(self.PAD)
                weekday_list.append(0)
                minute_list.append(0)

            mask_a = [0] + [1 if x != self.PAD else 0 for x in token]
            self.input_token.append([self.SOS] + token)
            self.input_day.append(minute_list)
            self.input_week.append(weekday_list)
            self.attention_mask.append(mask_a)
            self.trj_id.append([i])

        self.attention_mask = torch.tensor(self.attention_mask).to(device)
        self.input_token = torch.tensor(self.input_token).to(device)
        self.input_day = torch.tensor(self.input_day).to(device)
        self.input_week = torch.tensor(self.input_week).to(device)

    def __len__(self):
        return len(self.input_token)

    def __getitem__(self, index):
        return self.input_token[index], self.input_day[index], self.input_week[index], self.attention_mask[index], self.trj_id[index]

