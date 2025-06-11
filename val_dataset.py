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


seed = 5000
random.seed(seed)
np.random.seed(seed)


class similarity_Dataset_A(Dataset):  # similarity
    def __init__(self, args):
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

        super().__init__()
        self.input_token = []
        self.input_week = []
        self.input_day = []
        self.trj_id = []
        self.attention_mask = []
        self.data = []
        self.max_len = args.max_len
        self.vocab_size = args.vocab_size
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4
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
        else:
            trjs = file1['rel_token']
            times = file1['timestamp']

        for i in tqdm((range((int)(50000))), desc='data loading', ncols=80):
            trj, tm = eval(trjs[i]), eval(times[i])
            weekday_list, minute_list = [], []
            day_list = []
            poi_list = []

            for j in range(len(tm)):
                dt = datetime.datetime.fromtimestamp(tm[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)
                day_list.append(dt.timetuple().tm_yday)
            la = len(trj)
            token = []
            week = [0]
            day = [0]
            year = [0]
            grid_list = []
            grid = [0]
            poi = [0]

            for j in range(la):
                grid_list.append(self.neighbor[str(trj[j])])
                poi_list.append(self.poi_grid[str(self.neighbor[str(trj[j])])])
                if args.city == 'beijing':
                    trj[j] += 5

            for j in range(la):
                token.append(trj[j])
                week.append(weekday_list[j])
                day.append(minute_list[j])
                year.append(day_list[j])
                grid.append(grid_list[j])
                poi.append(poi_list[j])

            last = self.max_len - len(token)
            for j in range(last - 1):
                token.append(self.PAD)
                week.append(0)
                day.append(0)
                year.append(0)
                grid.append(0)
                poi.append(0)

            mask_a = [0] + [1 if x != self.PAD else 0 for x in token]
            self.input_token.append([self.SOS] + token)
            self.input_day.append(day)
            self.input_week.append(week)
            self.grid.append(grid)
            self.attention_mask.append(mask_a)
            self.trj_id.append([i])
            self.day.append(year)
            self.poi.append(poi)
            self.task_list.append(self.task_info)

        self.attention_mask = torch.tensor(self.attention_mask).to(device)
        self.input_token = torch.tensor(self.input_token).to(device)
        self.input_day = torch.tensor(self.input_day).to(device)
        self.input_week = torch.tensor(self.input_week).to(device)
        self.trj_id = torch.tensor(self.trj_id).to(device)
        self.grid = torch.tensor(self.grid).to(device)
        self.day = torch.tensor(self.day).to(device)
        self.poi = torch.tensor(self.poi).to(device)
        self.task_list = torch.tensor(self.task_list).to(device)

    def __len__(self):
        return len(self.input_token)

    def __getitem__(self, index):
        return self.input_token[index], self.input_day[index], self.input_week[index], self.attention_mask[index], self.trj_id[index], self.grid[index], self.day[index], self.poi[index], self.task_list[index]

class similarity_Dataset_Q(Dataset):  # similarity
    def __init__(self, args):
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

        super().__init__()
        self.input_token = []
        self.input_week = []
        self.input_day = []
        self.trj_id = []
        self.attention_mask = []
        self.data = []
        self.max_len = args.max_len
        self.vocab_size = args.vocab_size
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4
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
        else:
            trjs = file1['rel_token']
            times = file1['timestamp']

        detour_rate = 0.2

        # 定义一个生成detour轨迹的函数
        def detour(rate=0.9):
            p = np.random.random_sample()  # 生成一个[0,1)的随机数
            # return np.random.randint(args.vocab_size) if p > rate else 0  # 根据概率决定返回一个随机值或padding_id
            return 0 if p > rate else 0  # 根据概率决定返回一个随机值或padding_id

        # 随机打乱样本索引
        random_index = np.random.permutation(50000)
        random_index = random_index[:5000]
        if args.top_k > 0:
            random_index = [78]

        for i in tqdm((range(len(random_index))), desc='data loading', ncols=80):
            trj, tm = eval(trjs[random_index[i]]), eval(times[random_index[i]])
            weekday_list, minute_list = [], []
            poi_list = []
            day_list = []
            for j in range(len(tm)):
                dt = datetime.datetime.fromtimestamp(tm[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)
                day_list.append(dt.timetuple().tm_yday)
            la = len(trj)
            token = []
            week = [0]
            day = [0]
            year = [0]
            grid_list = []
            grid = [0]
            poi = [0]

            for j in range(la):
                grid_list.append(self.neighbor[str(trj[j])])
                poi_list.append(self.poi_grid[str(self.neighbor[str(trj[j])])])
                if args.city == 'beijing':
                    trj[j] += 5
            for j in range(la):
                token.append(trj[j])
                week.append(weekday_list[j])
                day.append(minute_list[j])
                year.append(day_list[j])
                grid.append(grid_list[j])
                poi.append(poi_list[j])

            detour_pos = np.random.choice(len(token), int(len(token) * detour_rate), replace=False)
            path = [detour() if i in detour_pos else e for i, e in enumerate(token)]
            token = path

            last = self.max_len - len(token)
            for j in range(last - 1):
                token.append(self.PAD)
                week.append(0)
                day.append(0)
                year.append(0)
                grid.append(0)
                poi.append(0)

            mask_a = [0] + [1 if x != self.PAD else 0 for x in token]
            self.input_token.append([self.SOS] + token)
            self.input_day.append(day)
            self.input_week.append(week)
            self.grid.append(grid)
            self.attention_mask.append(mask_a)
            self.trj_id.append([random_index[i]])
            self.day.append(year)
            self.poi.append(poi)
            self.task_list.append(self.task_info)

        self.attention_mask = torch.tensor(self.attention_mask).to(device)
        self.input_token = torch.tensor(self.input_token).to(device)
        self.input_day = torch.tensor(self.input_day).to(device)
        self.input_week = torch.tensor(self.input_week).to(device)
        self.trj_id = torch.tensor(self.trj_id).to(device)
        self.grid = torch.tensor(self.grid).to(device)
        self.day = torch.tensor(self.day).to(device)
        self.poi = torch.tensor(self.poi).to(device)
        self.task_list = torch.tensor(self.task_list).to(device)

    def __len__(self):
        return len(self.input_token)

    def __getitem__(self, index):
        return self.input_token[index], self.input_day[index], self.input_week[index], self.attention_mask[index], self.trj_id[index], self.grid[index], self.day[index], self.poi[index], self.task_list[index]



class val_simplify_Dataset(Dataset):
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
        self.simple_labels = []
        self.rel_trj = []
        self.data = []
        self.max_len = args.max_len  # not enough to 10 then add padding
        self.vocab_size = args.vocab_size
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
            rel_trj = file1['coordinate']
            simple_index = file1['simple_index']
        else:
            trjs = file1['rel_token']
            times = file1['timestamp']
            rel_trj = file1['coordinate']
            simple_index = file1['simple_index']

        for i in tqdm(range((int)(len(trjs))), desc='data input loading', ncols=80):
            trj, time, index = eval(trjs[i]), eval(times[i]), eval(simple_index[i])
            rel_cor = eval(rel_trj[i])
            weekday_list = [0]
            minute_list = [0]
            year_list = [0]
            grid_list = [0]
            poi_list = [0]

            for j in range(len(time)):
                dt = datetime.datetime.fromtimestamp(time[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)
                year_list.append(dt.timetuple().tm_yday)

            for j in range(len(trj)):
                grid_list.append(self.neighbor[str(trj[j])])
                poi_list.append(self.poi_grid[str(self.neighbor[str(trj[j])])])
                if args.city == 'beijing':
                    trj[j] += 5

            for j in range(self.max_len - len(trj) - 1):
                minute_list.append(0)
                weekday_list.append(0)
                trj.append(self.PAD)
                index.append(0)
                rel_cor.append([0])
                grid_list.append(0)
                year_list.append(0)
                poi_list.append(0)

            mask = [0] + [1 if x != self.PAD else 0 for x in trj]
            self.rel_trj.append([0] + rel_cor)
            self.inputToken.append([self.SOS] + trj)
            self.simple_labels.append([0] + index)
            self.attention_mask.append(mask)
            self.weekday.append(weekday_list)
            self.daytime.append(minute_list)
            self.grid.append(grid_list)
            self.day.append(year_list)
            self.poi.append(poi_list)
            self.task_list.append(self.task_info)

        self.inputToken = torch.tensor(self.inputToken).to(device)
        self.daytime, self.weekday = torch.tensor(self.daytime).to(device), torch.tensor(self.weekday).to(device)
        self.simple_labels = torch.tensor(self.simple_labels).to(device)
        self.attention_mask = torch.tensor(self.attention_mask).to(device)
        self.grid = torch.tensor(self.grid).to(device)
        self.day = torch.tensor(self.day).to(device)
        self.poi = torch.tensor(self.poi).to(device)
        self.task_list = torch.tensor(self.task_list).to(device)

    def __len__(self):
        return len(self.inputToken)

    def __getitem__(self, index):
        return self.inputToken[index], self.daytime[index], self.weekday[index], self.simple_labels[index], \
        self.attention_mask[index], self.rel_trj[index], self.grid[index], self.day[index], self.poi[index], self.task_list[index]


class val_generation_Dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        if args.city == 'beijing':
            with open(args.pre_path + '/data/beijing/grid.json') as f:
                grid = json.load(f)
        else:
            with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/grid.json') as f:
                grid = json.load(f)
        self.weekday = []
        self.daytime = []
        self.token = []
        self.data = []
        self.mask_index = []
        self.max_len = args.max_len  # not enough to 10 then add padding
        self.vocab_size = args.vocab_size
        self.minTime = 10000000000
        self.maxTime = 0
        self.PAD = 0
        self.UNK = 1
        self.SOS = 2
        self.EOS = 3
        self.MASK = 4
        self.grid = []
        self.neighbor = grid
        self.day = []


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
            grid_list = [0]
            year_list = [0]
            for j in range(len(time)):
                dt = datetime.datetime.fromtimestamp(time[j])
                weekday = dt.weekday() + 1
                minute = dt.hour * 60 + dt.minute + 1
                weekday_list.append(weekday)
                minute_list.append(minute)
                year_list.append(dt.timetuple().tm_yday)

            for j in range(len(trj)):
                grid_list.append(self.neighbor[str(trj[j])])
                if args.city == 'beijing':
                    trj[j] += 5

            for j in range(self.max_len - len(trj) - 1):
                minute_list.append(0)
                weekday_list.append(0)
                trj.append(self.PAD)
                grid_list.append(0)
                year_list.append(0)

            self.token.append([self.SOS] + trj)
            self.weekday.append(weekday_list)
            self.daytime.append(minute_list)
            self.day.append(year_list)
            self.grid.append(grid_list)
        self.token = torch.tensor(self.token).to(device)
        self.daytime, self.weekday = torch.tensor(self.daytime).to(device), torch.tensor(self.weekday).to(device)
        self.grid, self.day = torch.tensor(self.grid).to(device), torch.tensor(self.day).to(device)

    def __len__(self):
        return len(self.token)

    def __getitem__(self, index):
        return self.token[index], self.daytime[index], self.weekday[index], self.grid[index], self.day[index]


