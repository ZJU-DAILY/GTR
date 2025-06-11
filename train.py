import torch.nn
# from Model.model import TrajectoryBERT
from Model.model import VITwithGAT
from datasets import *
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


def init_adj_feature(args):
    rows = []
    cols = []
    data = []
    with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/neighbor.json', 'r') as f:
        nei = json.load(f)
    with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/edge2id.json') as f:
        edge2id = json.load(f)

    for key, value in nei.items():
        key = int(key)
        value = eval(value)
        for j in value:
            if str(j) not in edge2id:
                continue
            j = edge2id[str(j)]
            rows.append(key)
            cols.append(j)
            data.append(1)
    for i in range(0, 44646):
        rows.append(i)
        cols.append(i)
        data.append(1)

    indices = torch.tensor([rows, cols], dtype=torch.int32)
    values = torch.tensor(data, dtype=torch.int32)
    adj = torch.sparse_coo_tensor(indices, values, (args.vocab_size, args.vocab_size))
    adj = adj.to_dense()

    link = pd.read_csv(args.pre_path + '/data/porto_edge1/process_data/rn_porto/edges_with_neighbors1.csv')

    feature = torch.zeros(args.vocab_size, args.feature_dim)
    link_type = ['motorway_link', 'motorway', 'primary', 'trunk_link', 'trunk', 'primary_link', 'residential', 'secondary', 'tertiary', 'service', 'living_street', 'unclassified', 'secondary_link', 'tertiary_link']

    link_type_dict = {link_type: (i + 1) for i, link_type in enumerate(link_type)}

    for i in range(len(link)):
        eid = link['eid'][i]
        value = edge2id[str(eid)]
        feature[value, 0] = int(link['length'][i])
        feature[value, 1] = link_type_dict[link['highway'][i]]
        feature[value, 2] = int(link['lanes'][i])
        feature[value, 3] = int(link['maxspeed'][i])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = feature.to(device)
    return adj, feature


def init_adj_feature_beijing(args):
    rows = []
    cols = []
    data = []
    index_path = args.pre_path + '/data/beijing/bj_roadmap_edge/bj_roadmap_edge.rel'
    df = pd.read_csv(index_path, sep=',')

    for i in range(len(df)):
        rows.append(int(df['origin_id'][i]) + 5)
        cols.append(int(df['destination_id'][i]) + 5)
        data.append(1)

    for i in range(0, 40311):
        rows.append(i)
        cols.append(i)
        data.append(1)

    indices = torch.tensor([rows, cols], dtype=torch.int32)
    values = torch.tensor(data, dtype=torch.int32)
    adj = torch.sparse_coo_tensor(indices, values, (args.vocab_size, args.vocab_size))
    adj = adj.to_dense()

    road = pd.read_csv(args.pre_path + '/data/beijing/bj_roadmap_edge/bj_roadmap_edge.geo')

    feature = torch.zeros(args.vocab_size, args.feature_dim)
    for i in range(len(road)):
        gid = road['geo_id'][i] + 5
        highway = road['highway'][i]
        lines = road['lanes'][i]
        length = road['length'][i]
        maxspeed = road['maxspeed'][i]
        feature[gid, 0] = length
        feature[gid, 1] = lines
        feature[gid, 2] = maxspeed
        feature[gid, 3] = highway

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature = feature.to(device)

    return adj, feature


def train_classification_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('--------- adj and feature load success --------------------')

    trainData = ClassificationDataset(args)
    testData = ClassificationDataset(args)

    if args.city == 'beijing':
        train_input_filepath = args.pre_path + '/data/beijing/train_bj.csv'
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        train_input_filepath = args.pre_path + '/data/porto_edge1/process_data/train.csv'
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'

    trainData.load(train_input_filepath, device, args)
    testData.load(test_input_filepath, device, args)

    trainLoader = DataLoader(trainData, batch_size=args.batch_size)
    testLoader = DataLoader(testData, batch_size=args.batch_size)
    vocab_size = args.vocab_size
    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    if args.train != 4:
        bert = VITwithGAT(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)
    else:
        bert = VITwithGAT_ablation(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)

    model = TrajBERT(args=args, bert=bert, vocab_size=args.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    bestLoss = 100000
    bestAcc = 0

    if args.train != 4:
        modelpath = args.pre_path + '/output/pretrain_vit_triplet' + args.city + '.pth'

    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    for epoch in range(args.epochs):
        print(f'{epoch + 1} epoch starts')
        train_epochLoss = 0
        model.train()
        for inputToken, daytime, weekday, cls_labels, attention_mask, year, grid, poi, task_list in tqdm(trainLoader,desc=f'train: Epoch {epoch + 1}/{args.epochs}',ncols=80):
            optimizer.zero_grad()
            out = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='classification')
            cls_labels = cls_labels.squeeze(1)
            loss = criterion(out.squeeze(), cls_labels)
            loss.backward()
            optimizer.step()
            train_epochLoss += loss.item()

        train_epochLoss = train_epochLoss / len(trainLoader)
        test_epochLoss = 0
        test_acc = 0
        model.eval()
        for inputToken, daytime, weekday, cls_labels, attention_mask, year, grid, poi, task_list in tqdm(testLoader,desc=f'test: Epoch {epoch + 1}/{args.epochs}', ncols=80):
            out = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='classification')
            cls_labels = cls_labels.squeeze(1)
            out = out.squeeze()
            predict_cls = torch.argmax(out, dim=1).flatten()
            loss = criterion(out, cls_labels)
            test_epochLoss += loss.item()
            cls_labels = cls_labels.flatten()
            cnt = len(cls_labels)
            right = cls_labels.eq(predict_cls).sum().item()
            test_acc += right * 1.0 / cnt
        test_acc /= len(testLoader)
        test_epochLoss /= len(testLoader)

        print(f'{epoch + 1} epoch Acc is {test_acc}' + f' {epoch + 1} epoch test Loss is{test_epochLoss}')
        if bestAcc < test_acc:
            bestLoss = min(bestLoss, test_epochLoss)
            bestAcc = max(bestAcc, test_acc)
            savePath = args.pre_path + '/output'
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            if args.train != 4:
                torch.save(model.state_dict(), savePath + '/classification_VIT_'+args.city + '.pth')


        with open(args.pre_path + '/train_result/train_classification.log', 'a') as file:
            file.write(
                f'the {epoch + 1} epoch train loss:{train_epochLoss}'
                + f'the {epoch + 1} epoch test loss:{test_epochLoss}, Acc is {test_acc} \n')
        scheduler.step()


def train_time_estimate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainData = time_estimate_Dataset(args)
    testData = time_estimate_Dataset(args)

    if args.city == 'beijing':
        train_input_filepath = args.pre_path + '/data/beijing/train_bj.csv'
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        train_input_filepath = args.pre_path + '/data/porto_edge1/process_data/train.csv'
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'

    trainLoader = DataLoader(trainData, batch_size=args.batch_size)
    testLoader = DataLoader(testData, batch_size=args.batch_size)

    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    vocab_size = args.vocab_size
    if args.train != 4:
        bert = VITwithGAT(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)
    else:
        bert = VITwithGAT_ablation(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)

    trainData.load(train_input_filepath, device, args)
    testData.load(test_input_filepath, device, args)

    model = TrajBERT(args=args, bert=bert, vocab_size=args.vocab_size).to(device)
    # criterion = nn.MSELoss().to(device)
    criterion = nn.SmoothL1Loss(reduction='mean').to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    bestMae = 10000000000

    if args.train != 4:
        modelpath = args.pre_path + '/output/pretrain_vit_triplet' + args.city + '.pth'
    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    for epoch in range(args.epochs):
        print(f'{epoch + 1} epoch starts')
        train_epochLoss = 0
        model.train()
        for inputToken, daytime, weekday, time_labels, mask_attention, year, grid, poi, task_list in tqdm(trainLoader,desc=f'train: Epoch {epoch + 1}/{args.epochs}',ncols=80):
            optimizer.zero_grad()
            out = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='time_estimate')
            out = out.squeeze()
            loss = criterion(out, time_labels)
            loss.backward()
            optimizer.step()
            train_epochLoss += loss.item()
        train_epochLoss = train_epochLoss / len(trainLoader)
        test_epochLoss = 0
        epsilon = 1e-8  # 防止除零，特别是当实际值非常接近于0时
        test_mae = 0
        test_mse = 0
        test_mape = 0
        model.eval()
        for inputToken, daytime, weekday, time_labels, mask_attention, year, grid, poi, task_list in tqdm(testLoader,desc=f'test: Epoch {epoch + 1}/{args.epochs}',ncols=80):
            out = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='time_estimate')
            out = out.squeeze()
            # print(out)
            loss = criterion(out, time_labels)
            test_epochLoss += loss.item()
            out, time_labels = out.flatten(), time_labels.flatten()
            test_mae += torch.abs(out - time_labels).mean().item()
            test_mse += torch.mean((out - time_labels) ** 2).item()
            test_mape += (torch.abs((out - time_labels) / (time_labels + epsilon)) * 100).mean().item()

        test_mae /= len(testLoader)
        test_mse /= len(testLoader)
        test_mape /= len(testLoader)
        test_epochLoss /= len(testLoader)

        # 反归一化 MAE 和 MSE
        test_mae *= (testData.maxTime - testData.minTime)
        test_mse *= (testData.maxTime - testData.minTime) ** 2
        print(f'{epoch + 1} epoch mae is {test_mae}\n')
        if bestMae > test_mae:
            bestMae = min(bestMae, test_mae)
            savePath = args.pre_path + '/output'
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            if args.train != 4:
                if args.bert_type == 2:
                    torch.save(model.state_dict(), savePath + '/time_estimate_VIT_'+ args.city + '.pth')

        with open(args.pre_path + '/train_result/train_time_estimate.log', 'a') as file:
            file.write(
                f'the {epoch + 1} epoch train loss:{train_epochLoss}'
                + f'the {epoch + 1} epoch test loss:{test_epochLoss}, mae is {test_mae}\n')
        scheduler.step()


def train_simplify_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainData = simplifyDataset(args)
    testData = simplifyDataset(args)

    if args.city == 'beijing':
        train_input_filepath = args.pre_path + '/data/beijing/train_bj.csv'
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        train_input_filepath = args.pre_path + '/data/porto_edge1/process_data/train.csv'
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'


    trainData.load(train_input_filepath, device, args)
    testData.load(test_input_filepath, device, args)

    trainLoader = DataLoader(trainData, batch_size=args.batch_size)
    testLoader = DataLoader(testData, batch_size=args.batch_size)

    vocab_size = args.vocab_size
    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    if args.train != 4:
        bert = VITwithGAT(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)
    else:
        bert = VITwithGAT_ablation(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)

    model = TrajBERT(args=args, bert=bert, vocab_size=trainData.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    bestloss = 10000000000

    if args.train != 4:
        modelpath = args.pre_path + '/output/pretrain_vit_triplet' + args.city + '.pth'

    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    for epoch in range(args.epochs):
        print(f'{epoch + 1} epoch starts')
        train_epochLoss = 0
        model.train()
        for inputToken, daytime, weekday, simple_labels, a_mask, year, grid, poi, task_list in tqdm(trainLoader,
                                                                        desc=f'train: Epoch {epoch + 1}/{args.epochs}',
                                                                        ncols=80):
            optimizer.zero_grad()
            out = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='simplify')
            out = out.view(-1, 2)
            simple_labels = simple_labels.view(-1)
            loss = criterion(out, simple_labels)
            loss.backward()
            optimizer.step()
            train_epochLoss += loss.item()

        train_epochLoss = train_epochLoss / len(trainLoader)
        test_epochLoss = 0
        test_acc = 0
        model.eval()
        with torch.no_grad():
            for inputToken, daytime, weekday, simple_labels, a_mask, year, grid, poi, task_list in tqdm(testLoader,
                                                                            desc=f'test: Epoch {epoch + 1}/{args.epochs}',
                                                                            ncols=80):
                out = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='simplify')
                out = out.view(-1, 2)
                simple_labels = simple_labels.view(-1)
                loss = criterion(out, simple_labels)
                output = torch.argmax(out, dim=1).flatten()
                test_epochLoss += loss.item()
                simple_labels = simple_labels.flatten()
                simple_labels, a_mask = simple_labels.flatten(), a_mask.flatten()
                cnt, right = 0, 0
                for j in range(len(simple_labels)):
                    if a_mask[j] == 1:
                        cnt += 1
                        if output[j] == simple_labels[j]:
                            right += 1
                test_acc += right * 1.0 / cnt
        test_acc /= len(testLoader)
        test_epochLoss /= len(testLoader)

        print(f'{epoch + 1} epoch acc is {test_acc}\n')
        if bestloss > test_epochLoss:
            bestloss = min(bestloss, test_epochLoss)
            savePath = args.pre_path + '/output'
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            if args.train != 4:
                torch.save(model.state_dict(), savePath + '/simplify_VIT_'+args.city+'.pth')
        with open(args.pre_path + '/train_result/train_simplify.log', 'a') as file:
            file.write(
                f'the {epoch + 1} epoch train loss:{train_epochLoss}'
                + f'the {epoch + 1} epoch test loss:{test_epochLoss}, mae is {test_acc}\n')
        scheduler.step()


def train_imputation_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainData = imputationDataset(args)
    testData = imputationDataset(args)

    if args.city == 'beijing':
        train_input_filepath = args.pre_path + '/data/beijing/train_bj.csv'
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        train_input_filepath = args.pre_path + '/data/porto_edge1/process_data/train.csv'
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'


    trainData.load(train_input_filepath, device,args)
    testData.load(test_input_filepath, device,args)

    trainLoader = DataLoader(trainData, batch_size=args.batch_size)
    testLoader = DataLoader(testData, batch_size=args.batch_size)

    vocab_size = args.vocab_size

    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    if args.train != 4:
        bert = VITwithGAT(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)
    else:
        bert = VITwithGAT_ablation(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)

    model = TrajBERT(args=args, bert=bert, vocab_size=trainData.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    vocab_size = trainData.vocab_size
    bestloss = 100000000

    if args.train != 4:
        modelpath = args.pre_path + '/output/pretrain_vit_triplet' + args.city + '.pth'



    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    for epoch in range(args.epochs):
        print(f'{epoch + 1} epoch starts')
        train_epochLoss = 0
        model.train()
        for inputToken, daytime, weekday, token_labels, mask_index, year, grid, poi, task_list in tqdm(trainLoader,
                                                                           desc=f'train: Epoch {epoch + 1}/{args.epochs}',
                                                                           ncols=80):
            optimizer.zero_grad()
            out = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='imputation')
            out = out * mask_index.unsqueeze(-1)
            out = out.view(-1, vocab_size)
            token_labels = token_labels.view(-1)
            loss = criterion(out, token_labels)
            loss.backward()
            optimizer.step()
            train_epochLoss += loss.item()
        train_epochLoss = train_epochLoss / len(trainLoader)
        test_epochLoss = 0
        test_acc = 0
        model.eval()
        for inputToken, daytime, weekday, token_labels, mask_index, year, grid, poi, task_list in tqdm(testLoader,desc=f'test: Epoch {epoch + 1}/{args.epochs}', ncols=80):
            out = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='imputation')
            out = out * mask_index.unsqueeze(-1)
            output = torch.argmax(out, dim=2).flatten()
            out = out.view(-1, vocab_size)
            token_labels = token_labels.view(-1)
            loss = criterion(out, token_labels)
            test_epochLoss += loss.item()
            token_labels, mask_index = token_labels.flatten(), mask_index.flatten()
            cnt, right = 0, 0
            for j in range(len(token_labels)):
                if mask_index[j] == 1:
                    cnt += 1
                    if output[j] == token_labels[j]:
                        right += 1
            test_acc += right * 1.0 / cnt

        test_acc /= len(testLoader)
        test_epochLoss /= len(testLoader)

        print(f'{epoch + 1} epoch acc is {test_acc}\n')
        if bestloss > test_epochLoss:
            bestloss = min(bestloss, test_epochLoss)
            savePath = args.pre_path + '/output'
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            if args.train != 4:
                torch.save(model.state_dict(), savePath + '/imputation_VIT_'+args.city+'.pth')

        with open(args.pre_path + '/train_result/train_imputation.log', 'a') as file:
            file.write(
                f'the {epoch + 1} epoch train loss:{train_epochLoss}'
                + f'the {epoch + 1} epoch test loss:{test_epochLoss}, test_acc is {test_acc}\n')
        scheduler.step()


def train_generation_predict_model(args):
    print('------------trajectory generation task start-------------------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.city == 'beijing':
        train_input_filepath = args.pre_path + '/data/beijing/train_bj.csv'
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        train_input_filepath = args.pre_path + '/data/porto_edge1/process_data/train.csv'
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'


    trainData = generator_for_predict_Dataset(args)
    testData = generator_for_predict_Dataset(args)

    trainData.load(train_input_filepath, device, args)
    testData.load(test_input_filepath, device, args)

    trainLoader = DataLoader(trainData, batch_size=args.batch_size)
    testLoader = DataLoader(testData, batch_size=args.batch_size)

    vocab_size = args.vocab_size

    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    if args.train == 4:
        bert = VITwithGAT_ablation(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)
    else:
        bert = VITwithGAT(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)

    model = TrajBERT(args=args, bert=bert, vocab_size=trainData.vocab_size).to(device)

    criterion_CE = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    bestLoss = 10000
    bestAcc = 0

    if args.train != 4:
        modelpath = args.pre_path + '/output/pretrain_vit_triplet' + args.city + '.pth'

    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1
    print('--------------------start train generation for predict model--------------------')
    for epoch in range(args.epochs):
        print(f'{epoch + 1} epoch starts')
        train_epochLoss = 0
        model.train()
        for inputToken, daytime, weekday, token_labels, mask_index, year, grid, poi, task_list, in tqdm(trainLoader,desc=f'train: Epoch {epoch + 1}/{args.epochs}', ncols=80):
            optimizer.zero_grad()
            out1 = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='trj_predict')
            out1 = out1 * mask_index.unsqueeze(-1)
            out1 = out1.view(-1, vocab_size)
            token_labels = token_labels.view(-1)
            lossCEL = criterion_CE(out1, token_labels)
            lossCEL.backward()
            optimizer.step()
            train_epochLoss += lossCEL.item()
        train_epochLoss = train_epochLoss / len(trainLoader)
        test_acc = 0
        test_epochLoss = 0
        test_acc_count = 0
        model.eval()
        for inputToken, daytime, weekday, token_labels, mask_index, year, grid, poi, task_list in tqdm(testLoader,desc=f'test: Epoch {epoch + 1}/{args.epochs}', ncols=80):
            out1 = model(inputToken, daytime, weekday, year, grid, poi, task_list, task='trj_predict')
            out1 = out1 * mask_index.unsqueeze(-1)
            predictToken = torch.argmax(out1, dim=2).flatten()
            # compute the Loss of the test data
            out1 = out1.view(-1, vocab_size)
            token_labels = token_labels.view(-1)
            lossCEL = criterion_CE(out1, token_labels)
            test_epochLoss += lossCEL.item()
            # compute the Acc of prediction Tokens
            mask_index, labels = mask_index.flatten(), token_labels.flatten()
            count = 0
            all_count = 0
            for i in range(len(mask_index)):
                if mask_index[i] == 1:
                    if labels[i] == predictToken[i]:
                        count += 1
                    all_count += 1
            test_acc += count / all_count
            test_acc_count += count

        test_acc /= len(testLoader)
        test_epochLoss /= len(testLoader)
        print(f'{epoch + 1} epoch Acc is {test_acc}, test Loss is{test_epochLoss}')
        if bestAcc < test_acc:
            bestLoss = min(bestLoss, test_epochLoss)
            bestAcc = max(bestAcc, test_acc)
            savePath = args.pre_path + '/output'
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            if args.train != 4:
                torch.save(model.state_dict(), savePath + '/generation_predict_VIT_'+args.city+'.pth')
        with open(args.pre_path + '/train_result/train_predict.log', 'a') as file:
            file.write(f'the {epoch + 1} epoch train loss:{train_epochLoss}, '
                       + f'the {epoch + 1} epoch test loss:{test_epochLoss}, Acc is {test_acc}, right tokens count is{test_acc_count}' + '\n')
        scheduler.step()


def pretrain_mlm_triplet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainData = MLM_and_triplet_Dataset(args)
    testData = MLM_Only_Dataset(args)
    enhancedData = MLM_Only_Dataset(args)

    if args.city == 'beijing':
        train_input_filepath = args.pre_path + '/data/beijing/train_bj.csv'
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
        enhancedData_filepath = args.pre_path + '/data/beijing/train_bj_enhanced.csv'
    else:
        train_input_filepath = args.pre_path + '/data/porto_edge1/process_data/train.csv'
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'
        enhancedData_filepath = args.pre_path + '/data/porto_edge1/process_data/train_porto_enhanced.csv'

    trainData.load(train_input_filepath, device, args)
    testData.load(test_input_filepath, device, args)
    enhancedData.load(enhancedData_filepath, device, args)

    trainLoader = DataLoader(trainData, batch_size=args.batch_size)
    testLoader = DataLoader(testData, batch_size=args.batch_size)
    enhancedLoader = DataLoader(enhancedData, batch_size=args.batch_size)

    vocab_size = args.vocab_size
    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    bert = VITwithGAT(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)
    model = TrajBERT(args=args, bert=bert, vocab_size=trainData.vocab_size).to(device)
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=0).to(device)
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    bestLoss = 1000
    bestAcc = 0
    for epoch in range(args.epochs):
        print(f'{epoch + 1} epoch starts')
        train_epochLoss = 0
        train_acc = 0
        train_acc_count = 0
        all_word = 0
        model = model.train()
        for data in tqdm(trainLoader, desc=f'train: Epoch {epoch + 1}/{args.epochs}', ncols=80):
            inputToken, token_labels, daytime, weekday, day, grid, inputToken_p, daytime_p, weekday_p, day_p, grid_p, inputToken_n, daytime_n, weekday_n, day_n, grid_n, poi, poi_p, poi_n, task_list = data
            optimizer.zero_grad()
            out_mlm = model(inputToken, daytime, weekday, day, grid, poi, task_list, task='pretrain_mlm')
            # compute the Acc of prediction Tokens
            predictToken = torch.argmax(out_mlm, dim=2)

            count_mlm = predictToken.eq(token_labels).sum().item()
            allcount_mlm = token_labels.ne(0).sum().item()

            all_word += allcount_mlm
            train_acc += count_mlm / allcount_mlm

            train_acc_count += count_mlm
            out_mlm = out_mlm.view(-1, vocab_size)
            token_labels = token_labels.view(-1)

            out_a = model(inputToken, daytime, weekday, day, grid, poi, task_list, task='similarity')
            out_p = model(inputToken_p, daytime_p, weekday_p, day_p, grid_p, poi_p, task_list, task='similarity')
            out_n = model(inputToken_n, daytime_n, weekday_n, day_n, grid_n, poi_n, task_list, task='similarity')

            tripletloss = criterion_triplet(out_a, out_p, out_n)

            loss_mlm = criterion_mlm(out_mlm, token_labels)
            lossTotal = loss_mlm * 0.7 + tripletloss * 0.3
            lossTotal.backward()
            optimizer.step()
            train_epochLoss += lossTotal.item()

        train_acc = train_acc / len(trainLoader)
        train_epochLoss = train_epochLoss / len(trainLoader)

        if bestAcc < train_acc:
            bestLoss = min(bestLoss, train_epochLoss)
            bestAcc = max(bestAcc, train_acc)
            savePath = args.pre_path + '/output'
            if not os.path.exists(savePath):
                os.mkdir(savePath)
            torch.save(model.state_dict(), savePath + '/pretrain_vit_triplet' + args.city + '.pth')

        with open(args.pre_path + '/train_result/train_mlm_only.log', 'a') as file:
            file.write(
                f'the {epoch + 1} epoch train loss:{train_epochLoss}, train_acc is{train_acc_count}, '
                + f'Acc is {train_acc}, right tokens count is{train_acc_count}'
                + f', all word amount is {all_word}' + '\n')
        scheduler.step()




