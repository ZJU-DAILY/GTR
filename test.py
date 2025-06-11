import datetime
import json
import random

import pandas as pd
import osm2gmns as og
from tqdm import tqdm
import geopandas as gpd
from shapely import wkt


def gen_network_data():
    net = og.getNetFromFile('./data/porto_edge1/porto_1.osm', default_lanes=True, default_speed=True, POI=True)
    og.outputNetToCSV(net, './data/porto_edge1/road')



def generateID(lng, lat, col_num, row_num):
    if lng != '':
        lng = float(lng)

    if lat != '':
        lat = float(lat)

    # beijing
    lngMax = 116.5018476
    lngMin = 116.2500662
    latMax = 40.0006224
    latMin = 39.7955236

    # porto
    # lngMax = -8.5472841
    # lngMin = -8.69681086
    # latMax = 41.19714406
    # latMin = 41.126014

    if lng < lngMin or lng > lngMax or lat < latMin or lat > latMax:
        return -1

    col = (lngMax - lngMin) / col_num
    row = (latMax - latMin) / row_num

    return int((lng - lngMin) / col) + 1 + int((lat - latMin) / row) * col_num

def generate_enhanced_dataset_beijing():
    # time shift  ( +1 and -1 )    (30%)
    # trajectory shift  (neighbor)   (30%)
    # trajectory delete (0.05 - 0.1)  20%

    neb = {}  # neighbor
    df_rel = pd.read_csv('data/beijing/bj_roadmap_edge/bj_roadmap_edge.rel')

    for j in tqdm(range(len(df_rel))):
        origin_id = int(df_rel['origin_id'][j])
        destination_id = int(df_rel['destination_id'][j])
        neb.setdefault(origin_id, []).append(destination_id)

    df = pd.read_csv('data/beijing/train_bj.csv')
    ad = [1, -1]

    for i in tqdm(range(len(df))):
        trj = eval(df['path'][i])
        tm = eval(df['tlist'][i])

        if random.random() < 0.2:
            # delete
            len_del = int(len(trj) * random.uniform(0.05, 0.1))
            if random.random() < 0.5:
                trj = trj[len_del:]
                tm = tm[len_del:]
            else:
                trj = trj[:-len_del]
                tm = tm[:-len_del]

        for j in range(len(trj)):
            if random.random() < 0.3:
                tm[j] += ad[random.randint(0, 1)]

            if random.random() < 0.3:
                if trj[j] in neb:
                    if neb[trj[j]]:
                        trj[j] = random.choice(neb[trj[j]])

        df.at[i, 'path'] = str(trj)
        df.at[i, 'tlist'] = str(tm)

    df.to_csv('./data/beijing/train_bj_enhanced.csv', index=False)



def generate_enhanced_dataset_porto():
    # time shift  ( +1 and -1 )    (30%)
    # trajectory shift  (neighbor)   (30%)
    # trajectory delete (0.05 - 0.1)  20%

    neb = {}  # neighbor
    # df_rel = pd.read_csv('data/porto_edge1/process_data/rn_porto/edges_with_neighbors.csv')
    with open('data/porto_edge1/process_data/rn_porto/neighbor.json') as f:
        neighbor = json.load(f)

    with open('data/porto_edge1/process_data/rn_porto/edge2id.json') as f:
        edge2id = json.load(f)

    # for j in tqdm(range(len(df_rel))):
    #     origin_id = int(df_rel['origin_id'][j])
    #     destination_id = int(df_rel['destination_id'][j])
    #     neb.setdefault(origin_id, []).append(destination_id)

    df = pd.read_csv('data/porto_edge1/process_data/train.csv')
    ad = [1, -1]

    for i in tqdm(range(len(df))):
        trj = eval(df['rel_token'][i])
        tm = eval(df['timestamp'][i])


        if random.random() < 0.2:
            # delete
            len_del = int(len(trj) * random.uniform(0.05, 0.1))
            if random.random() < 0.5:
                trj = trj[len_del:]
                tm = tm[len_del:]
            else:
                trj = trj[:-len_del]
                tm = tm[:-len_del]

        for j in range(len(trj)):
            if random.random() < 0.3:
                index = random.randint(0, 1)
                tm[j] += ad[index]

            if random.random() < 0.3:
                if str(trj[j]) in neighbor and neighbor[str(trj[j])] != '-1':
                    tar = str(random.choice(eval(neighbor[str(trj[j])])))
                    if tar not in edge2id:
                        continue
                    trj[j] = edge2id[tar]
        df.at[i, 'rel_token'] = str(trj)
        df.at[i, 'timestamp'] = str(tm)

    df.to_csv('./data/porto_edge1/process_data/train_porto_enhanced.csv', index=False)
