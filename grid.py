import pandas
import random
import math
import json
import numpy as np
import tqdm
import pandas as pd
from geopy.distance import geodesic
import folium


def computeLng_Lat(filepath):
    lngMax = float('-inf')  # max Lng
    lngMin = float('inf')  # min Lng
    latMax = float('-inf')  # max Lat
    latMin = float('inf')  # min Lng

    data = pd.read_csv(filepath)

    pt = data['POLYLINE']

    print('ok')
    dp = []
    cnt = 0
    for i in range(len(pt)):
        if i % 10000 == 0:
            print(i)
        line = eval(pt[i])
        flag = 0
        for j in range(len(line)):
            d1 = line[j]
            x, y = d1[0], d1[1]
            if not (x > -8.71 and x < -8.56 and y > 41.1 and y < 41.21):
                flag = 1
                break
            # print(x, y, sep=';')
            lngMax = max(lngMax, x)
            lngMin = min(lngMin, x)

            latMax = max(latMax, y)
            latMin = min(latMin, y)
        if flag == 0:
            cnt += 1
        if flag == 1:
            dp.append(i)
    data.drop(dp, inplace=True)
    data.to_csv('./data/porto/real/test.csv', index=False)

    js = {
        'lngMax': lngMax,  # max Lng
        'lngMin': lngMin,  # min Lng
        'latMax': latMax,  # max Lat
        'latMin': latMin  # min Lng
    }

    path = 'data/porto/real/latLng.json'
    with open(path, 'w') as file:
        json.dump(js, file, indent=4)

    print(cnt)


def generateID(lng, lat, col_num, row_num):
    rangeFilePath = 'data/porto/real/latLng.json'
    with open(rangeFilePath, 'r') as f:
        data = json.load(f)

    if lng == '' or lat == '':
        return -1

    if lng != '':
        lng = float(lng)

    if lat != '':
        lat = float(lat)
    lngMax = data['lngMax']
    lngMin = data['lngMin']
    latMax = data['latMax']
    latMin = data['latMin']

    if lng < lngMin or lng > lngMax or lat < latMin or lat > latMax:
        return -1

    col = (lngMax - lngMin) / col_num
    row = (latMax - latMin) / row_num

    return int((lng - lngMin) / col) + 1 + int((lat - latMin) / row) * col_num


def convert_to_mercator(lat, lon):
    r_major = 6378137.0  # 地球的半长轴
    x = r_major * math.radians(lon)
    scale = x / lon
    y = 180.0 / math.pi * math.log(math.tan(math.pi / 4.0 + lat * (math.pi / 180.0) / 2.0)) * scale
    return x, y


def test_dis():
    with open('data/porto/real/latLng.json', 'r') as f:
        data = json.load(f)

    lngMax = data['lngMax']
    lngMin = data['lngMin']
    latMax = data['latMax']
    latMin = data['latMin']

    lat = geodesic((lngMax, latMax), (lngMax, latMin)).km
    lng = geodesic((lngMax, latMax), (lngMin, latMax)).km

    print(lat, lng)


def makeTrjMap(l):
    m = folium.Map(location=[l[0][1], l[0][0]], zoom_start=12)
    for i in range(len(l)):
        x = l[i][0]
        y = l[i][1]
        folium.Marker(
            location=(y, x),
            popup='Location',
            icon=folium.Icon(icon='cloud')
        ).add_to(m)
    m.save('output/m4.html')


def make_vocab_file(datafile):
    dt = pd.read_csv(datafile)
    cnt = 3
    js = {
        'PAD': 0,
        'MASK': 1
    }

    path = dt['POLYLINE']
    point = []
    line = []
    for i in range(len(path)):
        if i % 10000 == 0:
            print(i)
        p1 = eval(path[i])
        l = []
        for j in range(len(p1)):
            x, y = p1[j][0], p1[j][1]
            id_ = generateID(x, y, 160, 120)
            if id_ not in point:
                point.append(id_)
                js[id_] = cnt
                l.append(cnt)
                cnt += 1
            else:
                l.append(js[id_])
        line.append(l)
    dt['token'] = line
    dt.to_csv('./data/porto/real/train4.csv')

    with open('data/porto/real/vocab.json', 'w') as f:
        json.dump(js, f, indent=4)
