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


class ClassificationDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        with open(args.pre_path + args.vocab_file) as f:
            vocab = json.load(f)

        if args.city == 'beijing':
            with open(args.pre_path + '/data/beijing/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/beijing/bj_grid_poi.json') as f:
                poi_grid = json.load(f)
        else:
            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/porto_grid_poi.json') as f:
                poi_grid = json.load(f)

        self.inputToken = []
        self.inputTime = []
        self.cls_labels = []
        self.attention_mask = []
        self.data = []
        self.max_len = args.max_len  # not enough to 10 then add padding
        self.vocab_size = args.vocab_size
        self.vocab = vocab
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4
        self.daytime = []
        self.weekday = []
        self.grid = []
        self.day = []
        self.poi = []

        self.neighbor = grid
        self.poi_grid = poi_grid

        self.task_info = args.task_emb
        self.task_list = []

    def load(self, inputfile, device, args):
        cnt = 0
        file1 = pd.read_csv(inputfile)
        if args.city == 'beijing':
            trjs = file1['path']
            times = file1['tlist']
            cls = file1['vflag']
        else:
            trjs = file1['rel_token']
            times = file1['timestamp']
            cls = file1['CALL_TYPE']  # A for center predetermine, B for call of station, C from the roadside
        # 0 for A, 1 for B, 2 for C

        # for trj, time, cl in tqdm(zip(trjs, times, cls), desc='data input loading', ncols=80):
        for i in tqdm(range(int(len(trjs))), desc='data input loading', ncols=80):
            trj, time = eval(trjs[i]), eval(times[i])
            cl = cls[i]
            weekday_list = [0]
            minute_list = [0]
            day_list = [0]
            grid_list = [0]
            poi_list = [0]

            for j in range(len(time)):
                dt = datetime.datetime.fromtimestamp(time[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)
                day_list.append(dt.timetuple().tm_yday)

            for j in range(len(trj)):
                grid_list.append(self.neighbor[str(trj[j])])
                poi_list.append(self.poi_grid[ str(self.neighbor[str(trj[j])]) ])
                if args.city == 'beijing':
                    trj[j] = trj[j] + 5

            for j in range(self.max_len - len(trj) - 1):
                trj.append(self.PAD)
                minute_list.append(0)
                weekday_list.append(0)
                day_list.append(0)
                grid_list.append(0)
                poi_list.append(0)

            mask = [0] + [1 if x != self.PAD else 0 for x in trj]
            self.attention_mask.append(mask)
            self.inputToken.append([self.SOS] + trj)
            self.weekday.append(weekday_list)
            self.daytime.append(minute_list)
            self.grid.append(grid_list)
            self.day.append(day_list)
            if args.city == 'beijing':
                self.cls_labels.append([cl])
                if cl == 1:
                    cnt += 1
            else:
                self.cls_labels.append([ord(cl) - ord('A')])
            self.task_list.append(self.task_info)
            self.poi.append(poi_list)

        self.cls_labels, self.inputToken = torch.tensor(self.cls_labels).to(device), torch.tensor(self.inputToken).to(device)
        self.attention_mask = torch.tensor(self.attention_mask).to(device)
        self.daytime, self.weekday = torch.tensor(self.daytime).to(device), torch.tensor(self.weekday).to(device)
        self.grid, self.day = torch.tensor(self.grid).to(device), torch.tensor(self.day).to(device)
        self.task_list = torch.tensor(self.task_list).to(device)
        self.poi = torch.tensor(self.poi).to(device)

    def __len__(self):
        return len(self.inputToken)
    def __getitem__(self, index):
        return self.inputToken[index], self.daytime[index], self.weekday[index], self.cls_labels[index], self.attention_mask[index], self.day[index], self.grid[index], self.poi[index], self.task_list[index]


class time_estimate_Dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        with open(args.pre_path + args.vocab_file) as f:
            vocab = json.load(f)

        if args.city == 'beijing':
            with open(args.pre_path + '/data/beijing/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/beijing/bj_grid_poi.json') as f:
                poi_grid = json.load(f)
        else:
            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/porto_grid_poi.json') as f:
                poi_grid = json.load(f)

        self.inputToken = []
        self.inputTime = []
        self.padding_index = []
        self.time_labels = []

        self.daytime = []
        self.weekday = []
        self.data = []
        self.neighbor = grid
        self.day = []
        self.grid = []
        self.poi = []

        self.poi_grid = poi_grid

        self.max_len = args.max_len  # not enough to 10 then add padding
        self.vocab_size = args.vocab_size
        self.minTime = 10000000000
        self.maxTime = 0
        self.vocab = vocab
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4
        self.mask_attention = []
        self.task_info = args.task_emb
        self.task_list = []

    def load(self, inputfile, device, args):
        file1 = pd.read_csv(inputfile)
        if args.city == 'beijing':
            trjs = file1['path']
            times = file1['tlist']
        else :
            trjs = file1['rel_token']
            times = file1['timestamp']

        if args.city == 'porto':
            self.minTime = 1.25
            self.maxTime = 14.75
        else:
            self.minTime = 1.0
            self.maxTime = 60.0

        for j in tqdm(range(int(len(trjs))), desc='data input loading', ncols=80):
            trj, time = eval(trjs[j]), eval(times[j])
            dt1 = datetime.datetime.fromtimestamp(time[0])
            dt2 = datetime.datetime.fromtimestamp(time[len(time) - 1])
            dis = dt2 - dt1
            dis_time = (float(dis.total_seconds() / 60.0) - self.minTime) / (self.maxTime - self.minTime)
            weekday_list = [0]
            minute_list = [0]
            day_list = [0]
            grid = [0]
            poi_list = [0]

            dt = datetime.datetime.fromtimestamp(time[0])
            weekday = dt.weekday() + 1
            minute = dt.hour * 60 + dt.minute + 1
            weekday_list.append(weekday)
            minute_list.append(minute)
            day_list.append(dt.timetuple().tm_yday)

            for j in range(len(trj)):
                grid.append(self.neighbor[str(trj[j])])
                poi_list.append(self.poi_grid[str(self.neighbor[str(trj[j])])])
                if args.city == 'beijing':
                    trj[j] = trj[j] + 5

            for _ in range(self.max_len - len(trj) - 1):
                trj.append(self.PAD)
                grid.append(0)
                poi_list.append(0)

            for _ in range(self.max_len - len(minute_list)):
                minute_list.append(0)
                weekday_list.append(0)
                day_list.append(0)

            mask = [0] + [1 if x != self.PAD else 0 for x in trj]
            self.inputToken.append([self.SOS] + trj)
            self.time_labels.append(dis_time)
            self.daytime.append(minute_list)
            self.weekday.append(weekday_list)
            self.mask_attention.append(mask)
            self.grid.append(grid)
            self.day.append(day_list)
            self.task_list.append(self.task_info)
            self.poi.append(poi_list)

        self.inputToken = torch.tensor(self.inputToken).to(device)
        self.inputTime = torch.tensor(self.inputTime).to(device)
        self.time_labels = torch.tensor(self.time_labels).to(device)
        self.mask_attention = torch.tensor(self.mask_attention).to(device)
        self.daytime, self.weekday = torch.tensor(self.daytime).to(device), torch.tensor(self.weekday).to(device)
        self.grid, self.day = torch.tensor(self.grid).to(device), torch.tensor(self.day).to(device)
        self.poi = torch.tensor(self.poi).to(device)
        self.task_list = torch.tensor(self.task_list).to(device)

    def __len__(self):
        return len(self.inputToken)
    def __getitem__(self, index):
        return self.inputToken[index], self.daytime[index], self.weekday[index], self.time_labels[index], self.mask_attention[index], self.day[index], self.grid[index], self.poi[index], self.task_list[index]
    def getone(self, index):
        return self.inputToken[index], self.daytime[index], self.weekday[index], self.time_labels[index], self.mask_attention[index], self.day[index], self.grid[index], self.poi[index], self.task_list[index]


class simplifyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        with open(args.pre_path + args.vocab_file) as f:
            vocab = json.load(f)

        if args.city == 'beijing':
            with open(args.pre_path + '/data/beijing/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/beijing/bj_grid_poi.json') as f:
                poi_grid = json.load(f)
        else:
            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/porto_grid_poi.json') as f:
                poi_grid = json.load(f)

        self.inputToken = []
        self.weekday = []
        self.daytime = []
        self.simple_labels = []
        self.data = []
        self.max_len = args.max_len  # not enough to 10 then add padding
        self.vocab_size = args.vocab_size
        self.minTime = 10000000000
        self.maxTime = 0
        self.vocab = vocab
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4
        self.attention_mask = []

        self.grid = []
        self.day = []
        self.poi = []

        self.neighbor = grid
        self.poi_grid = poi_grid

        self.task_info = args.task_emb
        self.task_list = []

    def load(self, inputfile, device, args):
        file1 = pd.read_csv(inputfile)
        if args.city == 'beijing':
            trjs = file1['path']
            times = file1['tlist']
            simple_index = file1['simple_index']
        else:
            trjs = file1['rel_token']
            times = file1['timestamp']
            simple_index = file1['simple_index']

        for i in tqdm(range(int(len(trjs))), desc='data input loading', ncols=80):
            trj, time, index = eval(trjs[i]), eval(times[i]), eval(simple_index[i])
            weekday_list = [0]
            minute_list = [0]
            day_list = [0]
            grid = [0]
            poi_list = [0]

            for j in range(len(time)):
                dt = datetime.datetime.fromtimestamp(time[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)
                day_list.append(dt.timetuple().tm_yday)

            for j in range(len(trj)):
                grid.append(self.neighbor[str(trj[j])])
                poi_list.append(self.poi_grid[str(self.neighbor[str(trj[j])])])
                if args.city == 'beijing':
                    trj[j] = trj[j] + 5

            for j in range(self.max_len - len(trj) - 1):
                minute_list.append(0)
                weekday_list.append(0)
                trj.append(self.PAD)
                index.append(-100)
                grid.append(0)
                day_list.append(0)
                poi_list.append(0)

            mask = [0] + [1 if x != self.PAD else 0 for x in trj]
            self.inputToken.append([self.SOS] + trj)
            self.simple_labels.append([-100] + index)
            self.attention_mask.append(mask)
            self.weekday.append(weekday_list)
            self.daytime.append(minute_list)
            self.grid.append(grid)
            self.day.append(day_list)
            self.poi.append(poi_list)
            self.task_list.append(self.task_info)

        self.inputToken = torch.tensor(self.inputToken).to(device)
        self.daytime, self.weekday = torch.tensor(self.daytime).to(device), torch.tensor(self.weekday).to(device)
        self.simple_labels = torch.tensor(self.simple_labels).to(device)
        self.attention_mask = torch.tensor(self.attention_mask).to(device)
        self.grid, self.day = torch.tensor(self.grid).to(device), torch.tensor(self.day).to(device)
        self.task_list = torch.tensor(self.task_list).to(device)
        self.poi = torch.tensor(self.poi).to(device)

    def __len__(self):
        return len(self.inputToken)

    def __getitem__(self, index):
        return self.inputToken[index], self.daytime[index], self.weekday[index], self.simple_labels[index], self.attention_mask[index], self.day[index], self.grid[index],  self.poi[index], self.task_list[index]

class imputationDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        if args.city == 'beijing':
            with open(args.pre_path + '/data/beijing/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/beijing/bj_grid_poi.json') as f:
                poi_grid = json.load(f)
        else:
            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/porto_grid_poi.json') as f:
                poi_grid = json.load(f)

        self.inputToken = []
        self.weekday = []
        self.daytime = []
        self.mask_index = []
        self.token_labels = []
        self.data = []
        self.grid = []
        self.day = []
        self.max_len = args.max_len  # not enough to 10 then add padding
        self.vocab_size = args.vocab_size
        self.minTime = 10000000000
        self.maxTime = 0
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4

        self.poi = []

        self.neighbor = grid
        self.poi_grid = poi_grid

        self.task_info = args.task_emb
        self.task_list = []


    def load(self, inputfile, device, args):
        file1 = pd.read_csv(inputfile)
        if args.city == 'beijing':
            trjs = file1['path']
            times = file1['tlist']
        else:
            trjs = file1['rel_token']
            times = file1['timestamp']

        for i in tqdm(range(int(len(trjs))), desc='data input loading', ncols=80):
            trj, time = eval(trjs[i]), eval(times[i])
            weekday_list = [0]
            minute_list = [0]
            grid = [0]
            day_list = [0]
            poi_list = [0]

            for j in range(len(time)):
                dt = datetime.datetime.fromtimestamp(time[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)
                day_list.append(dt.timetuple().tm_yday)

            for j in range(len(trj)):
                grid.append(self.neighbor[str(trj[j])])
                poi_list.append(self.poi_grid[ str(self.neighbor[str(trj[j])]) ])
                if args.city == 'beijing':
                    trj[j] = trj[j] + 5
            l = len(trj)
            valid_mask_index = list(range(1, l - 1))
            mask_position = np.random.choice(valid_mask_index, max(1, int(len(valid_mask_index) * 0.2)), replace=False).astype(int)
            trj_input = trj.copy()
            for index in mask_position:
                trj_input[index] = self.MASK
                grid[index + 1] = self.MASK

            for j in range(self.max_len - len(trj) - 1):
                minute_list.append(0)
                weekday_list.append(0)
                trj_input.append(self.PAD)
                trj.append(self.PAD)
                grid.append(0)
                day_list.append(0)
                poi_list.append(0)

            mask_index = [0] + [1 if x == self.MASK else 0 for x in trj_input]
            self.token_labels.append([self.SOS] + trj)
            self.inputToken.append([self.SOS] + trj_input)
            self.weekday.append(weekday_list)
            self.daytime.append(minute_list)
            self.mask_index.append(mask_index)
            self.day.append(day_list)
            self.grid.append(grid)
            self.task_list.append(self.task_info)
            self.poi.append(poi_list)

        self.inputToken = torch.tensor(self.inputToken).to(device)
        self.mask_index = torch.tensor(self.mask_index).to(device)
        self.daytime, self.weekday = torch.tensor(self.daytime).to(device), torch.tensor(self.weekday).to(device)
        self.token_labels = torch.tensor(self.token_labels).to(device)
        self.grid, self.day = torch.tensor(self.grid).to(device), torch.tensor(self.day).to(device)
        self.task_list = torch.tensor(self.task_list).to(device)
        self.poi = torch.tensor(self.poi).to(device)

    def __len__(self):
        return len(self.inputToken)

    def __getitem__(self, index):
        return self.inputToken[index], self.daytime[index], self.weekday[index], self.token_labels[index], self.mask_index[index], self.day[index], self.grid[index], self.poi[index], self.task_list[index]

    def getone(self, index):
        return self.inputToken[index], self.daytime[index], self.weekday[index], self.token_labels[index], self.mask_index[index], self.day[index], self.grid[index], self.poi[index], self.task_list[index]

class generator_for_predict_Dataset( Dataset):
    def __init__(self, args):
        super().__init__()
        if args.city == 'beijing':
            with open(args.pre_path + '/data/beijing/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/beijing/bj_grid_poi.json') as f:
                poi_grid = json.load(f)
        else:
            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/porto_grid_poi.json') as f:
                poi_grid = json.load(f)

        with open(args.pre_path + args.vocab_file) as f:
            vocab = json.load(f)
        self.weekday = []
        self.daytime = []
        self.token = []
        self.data = []
        self.mask_index = []
        self.label = []

        self.grid = []
        self.day = []
        self.max_len = args.max_len
        self.vocab_size = args.vocab_size
        self.minTime = 10000000000
        self.maxTime = 0
        self.vocab = vocab
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4
        self.poi = []

        self.neighbor = grid
        self.poi_grid = poi_grid

        self.task_info = args.task_emb
        self.task_list = []

    def load(self, inputfile, device, args):
        file1 = pd.read_csv(inputfile)
        if args.city == 'beijing':
            trjs = file1['path']
            times = file1['tlist']
        else:
            trjs = file1['rel_token']
            times = file1['timestamp']

        for i in tqdm(range((int)(len(trjs))), desc='data input loading', ncols=80):
            trj, time = eval(trjs[i]), eval(times[i])
            weekday_list = [0]
            minute_list = [0]
            day_list = [0]
            grid = [0]
            poi_list = [0]

            for j in range(len(time)):
                dt = datetime.datetime.fromtimestamp(time[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)
                day_list.append(dt.timetuple().tm_yday)

            for j in range(len(trj)):
                grid.append(self.neighbor[str(trj[j])])
                poi_list.append(self.poi_grid[ str(self.neighbor[str(trj[j])]) ])
                if args.city == 'beijing':
                    trj[j] = trj[j] + 5

            len_mask = len(trj) - int(len(trj) * 0.2)
            token = trj.copy()
            token = token[:int(len_mask) - 1]
            for j in range(len_mask + 1, len(trj) + 1):
                grid[j] = self.MASK

            for j in range(len(trj) - len(token)):
                token.append(self.MASK)

            for j in range(self.max_len - len(trj) - 1):
                minute_list.append(0)
                weekday_list.append(0)
                trj.append(self.PAD)
                token.append(self.PAD)
                grid.append(0)
                day_list.append(0)
                poi_list.append(0)

            mask_index = [0] + [1 if x == self.MASK else 0 for x in token]
            self.token.append([self.SOS] + token)
            self.weekday.append(weekday_list)
            self.daytime.append(minute_list)
            self.mask_index.append(mask_index)
            self.label.append([self.SOS] + trj)
            self.grid.append(grid)
            self.day.append(day_list)
            self.task_list.append(self.task_info)
            self.poi.append(poi_list)

        self.token = torch.tensor(self.token).to(device)
        self.daytime, self.weekday = torch.tensor(self.daytime).to(device), torch.tensor(self.weekday).to(device)
        self.mask_index = torch.tensor(self.mask_index).to(device)
        self.label = torch.tensor(self.label).to(device)
        self.grid, self.day = torch.tensor(self.grid).to(device), torch.tensor(self.day).to(device)
        self.task_list = torch.tensor(self.task_list).to(device)
        self.poi = torch.tensor(self.poi).to(device)
    def __len__(self):
        return len(self.token)

    def __getitem__(self, index):
        return self.token[index], self.daytime[index], self.weekday[index], self.label[index], self.mask_index[index], self.day[index], self.grid[index], self.poi[index], self.task_list[index]


class MLM_and_triplet_Dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        with open(args.pre_path + args.vocab_file) as f:
            vocab = json.load(f)
        if args.city == 'beijing':
            with open(args.pre_path + '/data/beijing/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/beijing/bj_grid_poi.json') as f:
                poi_grid = json.load(f)
        else:
            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/grid.json') as f:
                grid = json.load(f)

            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/porto_grid_poi.json') as f:
                poi_grid = json.load(f)

        self.inputToken = []
        self.token_labels = []
        self.daytime = []
        self.weekday = []
        self.day = []
        self.data = []
        self.grid = []
        self.poi = []

        self.inputToken_p = []
        self.daytime_p = []
        self.weekday_p = []
        self.day_p = []
        self.grid_p = []
        self.poi_p = []

        self.inputToken_n = []
        self.daytime_n = []
        self.weekday_n = []
        self.day_n = []
        self.grid_n = []
        self.poi_n = []

        self.max_len = args.max_len
        self.vocab_size = args.vocab_size
        self.vocab = vocab
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4
        self.neighbor = grid
        self.poi_grid = poi_grid

        self.task_info = args.task_emb
        self.task_list = []

    def load(self, inputfile, device, args):
        file1 = pd.read_csv(inputfile)
        if args.city == 'beijing':
            trjs = file1['path']
            times = file1['tlist']
            neg_file = pd.read_csv(args.pre_path + '/data/beijing/vae_train.csv')
        else:
            trjs = file1['rel_token']
            times = file1['timestamp']
            neg_file = pd.read_csv(args.pre_path + '/data/porto_edge1/process_data/vae_train.csv')
        neg_df = neg_file['gen_token']
        # mlm task
        for f_index in tqdm(range(len(trjs)), desc='data input_mlm loading', ncols=80):
            trj, time = eval(trjs[f_index]), eval(times[f_index])
            if f_index > len(trjs) // 2:
                n_index = random.randint(0, f_index)
            else:
                n_index = random.randint(f_index + 1, len(trjs) - 1)

            trj_n, time_n = eval(trjs[n_index]), eval(times[n_index])
            if random.random() < 0:
                trj_n = eval(neg_df[f_index])
                time_n = time
                trj_n = trj_n[:len(trj)]
                time_n = time_n[:len(trj_n)]

            seq = [self.SOS]
            labels = [self.PAD]
            Grid = [self.PAD]
            poi = [self.PAD]

            seq_n = [self.SOS]
            Grid_n = [0]
            poi_n = [self.PAD]

            weekday_list = [0]
            minute_list = [0]
            day_list = [0]

            weekday_list_n = [0]
            minute_list_n = [0]
            day_list_n = [0]
            for j in range(len(time)):
                dt = datetime.datetime.fromtimestamp(time[j])
                weekday = dt.weekday()
                minute = dt.hour * 60 + dt.minute
                day = dt.timetuple().tm_yday
                weekday_list.append(weekday)
                minute_list.append(minute)
                day_list.append(day)


            for j in range(len(time_n)):
                dt_n = datetime.datetime.fromtimestamp(time_n[j])
                weekday_n = dt_n.weekday()
                minute_n = dt_n.hour * 60 + dt_n.minute
                day_n = dt.timetuple().tm_yday
                weekday_list_n.append(weekday_n)
                minute_list_n.append(minute_n)
                day_list_n.append(day_n)

            for j in range(len(trj_n)):
                if args.city == 'beijing':
                    trj_n[j] = trj_n[j] + 5

                seq_n.append(trj_n[j])
                if args.city == 'beijing':
                    Grid_n.append(self.neighbor[str(trj_n[j] - 5)])
                    poi_n.append(self.poi_grid[ str(self.neighbor[str(trj_n[j] - 5)]) ])
                else:
                    Grid_n.append(self.neighbor[str(trj_n[j])])
                    poi_n.append(self.poi_grid[ str(self.neighbor[str(trj_n[j])]) ])

            mask_ratio = 0.3
            num_pairs = int(len(trj) * mask_ratio / 2)

            candidate_indices = list(range(len(trj) - 1))
            random.shuffle(candidate_indices)
            selected_indices = sorted(candidate_indices[:num_pairs])

            mask_set = set()
            for idx in selected_indices:
                mask_set.add(idx)
                mask_set.add(idx + 1)

            for j in range(len(trj)):
                if args.city == 'beijing':
                    trj[j] = trj[j] + 5
                if j in mask_set:
                    if random.random() < 0.75:
                        new_token = self.MASK
                    else:
                        new_token = random.randint(5, self.vocab_size - 2)
                    seq.append(new_token)
                    labels.append(trj[j])
                    Grid.append(self.MASK)
                    poi.append(self.MASK)
                else:
                    seq.append(trj[j])
                    labels.append(self.PAD)
                    if args.city == 'beijing':
                        Grid.append(self.neighbor[str(trj[j] - 5)])
                        poi.append(self.poi_grid[ str(self.neighbor[str(trj[j] - 5)]) ])
                    else:
                        Grid.append(self.neighbor[str(trj[j])])
                        poi.append(self.poi_grid[ str(self.neighbor[str(trj[j])]) ])
            rg = [0.7, 0.8, 0.9]
            rd_index = int(rg[random.randint(0,2)] * len(self.inputToken))
            seq_p = seq[:rd_index]
            Grid_p = Grid[: rd_index]
            weekday_list_p = weekday_list[: rd_index]
            minute_list_p = minute_list[: rd_index]
            day_list_p = day_list[: rd_index]
            poi_p = poi[: rd_index]

            for j in range(self.max_len - len(seq)):
                labels.append(self.PAD)
                seq.append(self.PAD)
                Grid.append(self.PAD)
                minute_list.append(0)
                weekday_list.append(0)
                day_list.append(0)
                poi.append(self.PAD)
            seq[-1] = self.EOS

            for j in range(self.max_len - len(seq_p)):
                seq_p.append(self.PAD)
                Grid_p.append(self.PAD)
                minute_list_p.append(0)
                weekday_list_p.append(0)
                day_list_p.append(0)
                poi_p.append(self.PAD)

            for j in range(self.max_len - len(seq_n)):
                seq_n.append(self.PAD)
                Grid_n.append(self.PAD)
                minute_list_n.append(0)
                weekday_list_n.append(0)
                day_list_n.append(0)
                poi_n.append(self.PAD)

            self.inputToken.append(seq)
            self.token_labels.append(labels)
            self.daytime.append(minute_list)
            self.weekday.append(weekday_list)
            self.day.append(day_list)
            self.grid.append(Grid)
            self.poi.append(poi)

            self.inputToken_p.append(seq_p)
            self.daytime_p.append(minute_list_p)
            self.weekday_p.append(weekday_list_p)
            self.day_p.append(day_list_p)
            self.grid_p.append(Grid_p)
            self.poi_p.append(poi_p)

            self.inputToken_n.append(seq_n)
            self.daytime_n.append(minute_list_n)
            self.weekday_n.append(weekday_list_n)
            self.day_n.append(day_list_n)
            self.grid_n.append(Grid_n)
            self.poi_n.append(poi_n)

            self.task_list.append(self.task_info)

        self.token_labels = torch.tensor(self.token_labels).to(device)
        self.inputToken = torch.tensor(self.inputToken).to(device)
        self.daytime, self.weekday = torch.tensor(self.daytime).to(device), torch.tensor(self.weekday).to(device)
        self.day = torch.tensor(self.day).to(device)
        self.grid = torch.tensor(self.grid).to(device)
        self.poi = torch.tensor(self.poi).to(device)

        self.inputToken_p = torch.tensor(self.inputToken_p).to(device)
        self.daytime_p, self.weekday_p = torch.tensor(self.daytime_p).to(device), torch.tensor(self.weekday_p).to(device)
        self.day_p = torch.tensor(self.day_p).to(device)
        self.grid_p = torch.tensor(self.grid_p).to(device)
        self.poi_p = torch.tensor(self.poi_p).to(device)

        self.inputToken_n = torch.tensor(self.inputToken_n).to(device)
        self.daytime_n, self.weekday_n = torch.tensor(self.daytime_n).to(device), torch.tensor(self.weekday_n).to(device)
        self.day_n = torch.tensor(self.day_n).to(device)
        self.grid_n = torch.tensor(self.grid_n).to(device)
        self.poi_n = torch.tensor(self.poi_n).to(device)

        self.task_list = torch.tensor(self.task_list).to(device)

    def __len__(self):
        return len(self.inputToken)

    def __getitem__(self, index):
        return self.inputToken[index], self.token_labels[index], self.daytime[index], self.weekday[index], self.day[index], self.grid[index], \
            self.inputToken_p[index], self.daytime_p[index], self.weekday_p[index], self.day_p[index], self.grid_p[index], \
            self.inputToken_n[index], self.daytime_n[index], self.weekday_n[index], self.day_n[index], self.grid_n[index], \
            self.poi[index], self.poi_p[index], self.poi_n[index], self.task_list[index]

