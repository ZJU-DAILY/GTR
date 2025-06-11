from torchmetrics import Accuracy, F1Score, AUROC
import torch.nn
from Model.model import TrajBERT, VITwithGAT
from datasets import *
from torch.utils.data import DataLoader
import os
from lossfunc import *
from val_dataset import *
from dtw import dtw
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import euclidean_distances




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
    link_type = ['motorway_link', 'motorway', 'primary', 'trunk_link', 'trunk', 'primary_link', 'residential',
                 'secondary', 'tertiary', 'service', 'living_street', 'unclassified', 'secondary_link', 'tertiary_link']

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


def val_similarity_compute(args):
    print('validate the similarity task')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.city == 'beijing':
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'

    query_Data = similarity_Dataset_Q(args)
    answer_Data = similarity_Dataset_A(args)

    query_Data.load(inputfile=test_input_filepath, device=device, args=args)
    answer_Data.load(inputfile=test_input_filepath, device=device, args=args)

    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    bert = VITwithGAT(args=args, vocab_size=args.vocab_size, adj=adj, feature=feature).to(device)
    model = TrajBERT(args=args, bert=bert, vocab_size=query_Data.vocab_size).to(device)

    query_Loader = DataLoader(query_Data, batch_size=1, shuffle=False)
    answer_Loader = DataLoader(answer_Data, batch_size=1, shuffle=False)

    if args.train == 2:
        modelpath = args.pre_path + '/output/pretrain_vit_triplet' + args.city + '.pth'
    else:
        modelpath = args.pre_path + '/output/pretrain_vit_triplet' + args.city + '.pth'
    print(modelpath)
    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1
    model.eval()
    sum_rank = 0
    hits_at_1 = 0
    hits_at_5 = 0
    emb_a_list = []

    with torch.no_grad():
        for trj_a, day_a, week_a, att_a, id_a, grid_a, day_a, poi_a, task_list_a in tqdm(answer_Loader, ncols=80):
            emb_a = model(trj_a, day_a, week_a, day_a, grid_a, poi_a, task_list_a, task='similarity')
            input_mask_expanded = att_a.unsqueeze(-1).expand_as(emb_a).float()  # (batch_size, seq_length, feat_dim)
            sum_embeddings = torch.sum(emb_a * input_mask_expanded, dim=1)  # (batch_size, feat_dim)
            sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch_size, feat_dim)
            avg_embeddings = sum_embeddings / sum_mask
            emb_a_numpy = avg_embeddings.squeeze().cpu().detach().numpy()
            emb_a_list.append((emb_a_numpy, id_a.squeeze().cpu().numpy()))

        emb_q_list = []
        for trj, day, week, att, id_or, grid, day, poi, task_list in tqdm(query_Loader, ncols=80):
            emb_q = model(trj, day, week, day, grid, poi, task_list, task='similarity')
            input_mask_expanded = att.unsqueeze(-1).expand_as(emb_q).float()  # (batch_size, seq_length, feat_dim)
            sum_embeddings = torch.sum(emb_q * input_mask_expanded, dim=1)  # (batch_size, feat_dim)
            sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch_size, feat_dim)
            avg_embeddings = sum_embeddings / sum_mask
            emb_q_numpy = avg_embeddings.squeeze().cpu().detach().numpy()
            emb_q_list.append((emb_q_numpy, id_or.squeeze().cpu().numpy()))

        emb_q_vectors, emb_q_indices = zip(*emb_q_list)
        emb_a_vectors, emb_a_indices = zip(*emb_a_list)

        # 计算距离矩阵
        dist_matrix = euclidean_distances(emb_q_vectors, emb_a_vectors)

        # 对每个查询计算排名
        for i, q_index in enumerate(emb_q_indices):
            distances = dist_matrix[i]
            sorted_indices = np.argsort(distances)
            ranks = np.where(sorted_indices == q_index)[0] + 1
            sum_rank += ranks[0]
            if ranks[0] <= 1:
                hits_at_1 += 1
            if ranks[0] <= 5:
                hits_at_5 += 1
        num_queries = len(emb_q_list)
        mean_rank = sum_rank / num_queries
        hr_at_1 = hits_at_1 / num_queries
        hr_at_5 = hits_at_5 / num_queries
        print(f'Mean Rank: {mean_rank}, HR@1: {hr_at_1}, HR@5: {hr_at_5}')

    print(f'In {args.city}, Mean Rank: {mean_rank}, HR@1: {hr_at_1}, HR@5: {hr_at_5}\n')
    with open(args.pre_path + '/val_result/val_similarity.log', 'a') as file:
        file.write(f'In {args.city}, Mean Rank: {mean_rank}, HR@1: {hr_at_1}, HR@5: {hr_at_5}\n')


def val_simplify_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testData = val_simplify_Dataset(args)

    if args.city == 'beijing':
        id2location = {}
        df = pd.read_csv(args.pre_path + '/data/beijing/bj_roadmap_edge/bj_roadmap_edge.geo')
        for i in range(len(df)):
            cor = eval(df['coordinates'][i])
            id2location[int(df['geo_id'][i])] = ((cor[0][0] + cor[-1][0]) / 2, (cor[0][1] + cor[-1][1]) / 2)

    else:
        with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/edge2id.json') as f:
            dict_vocab = json.load(f)
            reversed_vocab = {v: k for k, v in dict_vocab.items()}
        df_edge = pd.read_csv(args.pre_path + '/data/porto_edge1/process_data/rn_porto/edge.csv')
        id2location = {}
        for j in range(len(df_edge)):
            cor = eval(df_edge['coordinate'][j])
            id2location[int(df_edge['eid'][j])] = (float(cor[0][0] + cor[-1][0]) / 2, float(cor[0][1] + cor[-1][1]) / 2)

    if args.city == 'beijing':
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'

    testData.load(test_input_filepath, device, args)
    testLoader = DataLoader(testData, batch_size=1)

    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    bert = VITwithGAT(args=args, vocab_size=args.vocab_size, adj=adj, feature=feature).to(device)

    model = TrajBERT(args=args, bert=bert, vocab_size=testData.vocab_size).to(device)

    if args.train == 2:
        modelpath = args.pre_path + '/output/simplify_VIT_' + args.city + '_update.pth'
    else:
        modelpath = args.pre_path + '/output/simplify_VIT_' + args.city + '.pth'

    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    model.eval()
    if args.city == 'porto':
        # -------------------porto-------------------
        with torch.no_grad():
            ped_all = 0
            sed_all = 0
            for inputToken, daytime, weekday, simple_labels, a_mask, rel_trj1, grid, day, poi, task_info in tqdm(testLoader, ncols=80):
                out = model(inputToken, daytime, weekday, day, grid, poi, task_info, task='simplify')
                out = out.view(-1, 2)
                simple_labels = simple_labels.view(-1)
                output = torch.argmax(out, dim=1).flatten()
                simple_labels = simple_labels.flatten()
                simple_labels, a_mask = simple_labels.flatten(), a_mask.flatten()
                inputToken = inputToken.flatten()

                pre_trj = []
                rel_trj = []
                cnt, right = 0, 0
                for j in range(len(simple_labels)):
                    if a_mask[j] == 1:
                        rel_token = inputToken[j].item()
                        ori_rel_token = reversed_vocab[rel_token]
                        rel_trj.append(id2location[int(ori_rel_token)])
                        cnt += 1
                        if output[j] == 1:
                            tar = inputToken[j].item()
                            ori_tar = reversed_vocab[tar]
                            location_tar = id2location[int(ori_tar)]
                            pre_trj.append(location_tar)
                ped_score = trajectory_distance(rel_trj, pre_trj)
                sed_score = dynamic_time_warping(rel_trj, pre_trj)
                sed_all += sed_score
                ped_all += ped_score
            ped_all /= len(testLoader)
            sed_all /= len(testLoader)
            print(f'In {args.city}, PED score = {ped_all}, SED score = {sed_all}\n')
            with open(args.pre_path + '/val_result/val_simplify.log', 'a') as file:
                if args.bert_type == 0:
                    file.write(f'In {args.city}, PED score = {ped_all}, SED score = {sed_all}\n')
                else:
                    file.write(f'In {args.city}, PED score = {ped_all}, SED score = {sed_all}\n')
    else:
        # -------------------beijing-------------------
        with torch.no_grad():
            ped_all = 0
            sed_all = 0
            for inputToken, daytime, weekday, simple_labels, a_mask, rel_trj1, grid, day, poi, task_info in tqdm(testLoader, ncols=80):
                out = model(inputToken, daytime, weekday, day, grid, poi, task_info, task='simplify')
                out = out.view(-1, 2)
                simple_labels = simple_labels.view(-1)
                output = torch.argmax(out, dim=1).flatten()
                simple_labels = simple_labels.flatten()
                simple_labels, a_mask = simple_labels.flatten(), a_mask.flatten()
                inputToken = inputToken.flatten()
                pre_trj = []
                rel_trj = []
                cnt, right = 0, 0
                for j in range(len(simple_labels)):
                    if a_mask[j] == 1:
                        rel_token = inputToken[j].item()
                        rel_trj.append(id2location[int(rel_token) - 5])
                        cnt += 1
                        if output[j] == 1:
                            tar = inputToken[j].item()
                            location_tar = id2location[int(tar) - 5]
                            pre_trj.append(location_tar)
                if len(rel_trj) < 5:
                    continue
                ped_score = trajectory_distance(rel_trj, pre_trj)
                sed_score = dynamic_time_warping(rel_trj, pre_trj)
                sed_all += sed_score
                ped_all += ped_score
            ped_all /= len(testLoader)
            sed_all /= len(testLoader)
            print(f'In {args.city}, PED score = {ped_all}, SED score = {sed_all}\n')
            with open(args.pre_path + '/val_result/val_simplify.log', 'a') as file:
                if args.bert_type == 0:
                    file.write(f'In {args.city}, PED score = {ped_all}, SED score = {sed_all}\n')
                else:
                    file.write(f'In {args.city}, PED score = {ped_all}, SED score = {sed_all}\n')


# mean acc, recall1, recall3
def val_imputation_compute(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testData = imputationDataset(args)
    if args.city == 'beijing':
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'
    testData.load(test_input_filepath, device, args)
    testLoader = DataLoader(testData, batch_size=1)

    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    bert = VITwithGAT(args=args, vocab_size=args.vocab_size, adj=adj, feature=feature).to(device)
    model = TrajBERT(args=args, bert=bert, vocab_size=testData.vocab_size).to(device)

    if args.train == 2:
        modelpath = args.pre_path + '/output/imputation_VIT_' + args.city + '_update.pth'
    else:
        modelpath = args.pre_path + '/output/imputation_VIT_' + args.city + '.pth'

    print(modelpath)
    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    test_acc = 0
    recall1, recall3 = 0, 0

    model.eval()
    for inputToken, daytime, weekday, token_labels, mask_index, day, grid, poi, task_info in tqdm(testLoader, ncols=80):
        out = model(inputToken, daytime, weekday, day, grid, poi, task_info, task='imputation')
        out = out * mask_index.unsqueeze(-1)
        output = torch.argmax(out, dim=2).flatten()

        bc, seq_len = mask_index.size()
        r1, r3 = [], []
        for i in range(bc):
            for j in range(seq_len):
                if mask_index[i][j] == 1:
                    top_1_preds = torch.topk(out[i][j], 3).indices
                    top_3_preds = torch.topk(out[i][j], 5).indices
                    if token_labels[i][j] in top_1_preds:
                        r1.append(1)
                    else:
                        r1.append(0)
                    if token_labels[i][j] in top_3_preds:
                        r3.append(1)
                    else:
                        r3.append(0)
        recall1 += np.mean(r1)
        recall3 += np.mean(r3)

        token_labels = token_labels.view(-1)
        token_labels, mask_index = token_labels.flatten(), mask_index.flatten()
        cnt, right = 0, 0
        for j in range(len(token_labels)):
            if mask_index[j] == 1:
                cnt += 1
                if output[j] == token_labels[j]:
                    right += 1
        test_acc += right * 1.0 / cnt
    test_acc /= len(testLoader)
    recall1 /= len(testLoader)
    recall3 /= len(testLoader)
    print(f'imputation {args.city} acc score = {test_acc}, recall1 = {recall1}, recall3 = {recall3}\n')
    with open(args.pre_path + '/val_result/val_imputation.log', 'a') as file:
        file.write(f'imputation {args.city} acc score = {test_acc}, recall1 = {recall1}, recall3 = {recall3}\n')


def val_generation_compute(args):
    print('------------------------validate the generation task------------------------')
    # use BERT transform the trajectory into embedding, and use cos_similarity to compute
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.city == 'beijing':
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'

    input_data = generator_for_predict_Dataset(args)
    input_data.load(inputfile=test_input_filepath, device=device, args=args)

    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)

    bert = VITwithGAT(args=args, vocab_size=args.vocab_size, adj=adj, feature=feature).to(device)

    model = TrajBERT(args=args, bert=bert, vocab_size=input_data.vocab_size).to(device)
    testLoader = DataLoader(input_data, batch_size=1)
    modelpath = args.pre_path + '/output/generation_predict_VIT_' + args.city + '.pth'


    if args.train == 2:
        modelpath = args.pre_path + '/output/generation_predict_VIT_' + args.city + '_update.pth'
    else:
        modelpath = args.pre_path + '/output/generation_predict_VIT_' + args.city + '.pth'

    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    if args.city == 'beijing':
        id2location = {}
        df = pd.read_csv(args.pre_path + '/data/beijing/bj_roadmap_edge/bj_roadmap_edge.geo')
        for i in range(len(df)):
            cor = eval(df['coordinates'][i])
            id2location[int(df['geo_id'][i])] = ((cor[0][0] + cor[-1][0]) / 2, (cor[0][1] + cor[-1][1]) / 2)

    else:
        with open(args.pre_path + '/data/porto_edge1/process_data/rn_porto/edge2id.json') as f:
            dict_vocab = json.load(f)
            reversed_vocab = {v: k for k, v in dict_vocab.items()}
        df_edge = pd.read_csv(args.pre_path + '/data/porto_edge1/process_data/rn_porto/edge.csv')
        id2location = {}
        for j in range(len(df_edge)):
            cor = eval(df_edge['coordinate'][j])
            id2location[int(df_edge['eid'][j])] = (float(cor[0][0] + cor[-1][0]) / 2, float(cor[0][1] + cor[-1][1]) / 2)

    all_value = 0
    all_hausdorff = 0
    if args.city == 'porto':
        for inputToken, daytime, weekday, token_labels, mask_index, grid, day, poi, task_info in tqdm(testLoader, ncols=80):
            out1 = model(inputToken, daytime, weekday, grid, day, poi, task_info, 'trj_predict')
            out1 = out1 * mask_index.unsqueeze(-1)
            predictToken = torch.argmax(out1, dim=2).flatten()
            # compute the Loss of the test data
            token_labels = token_labels.view(-1)
            # compute the Acc of prediction Tokens
            mask_index, labels = mask_index.flatten(), token_labels.flatten()
            pre_trj = []
            trj = []
            for j in range(len(token_labels)):
                tar = token_labels[j].item()
                pre_token = predictToken[j].item()
                if int(tar) <= 4 or int(pre_token) <= 4:
                    continue
                ori_tar = reversed_vocab[tar]
                ori_token = reversed_vocab[pre_token]
                if int(tar) <= 4 or int(ori_tar) <= 4:
                    continue
                location_tar = id2location[int(ori_tar)]
                trj.append(location_tar)

                if pre_token <= 4:
                    continue
                location_token = id2location[int(ori_token)]
                if mask_index[j] == 1:
                    pre_trj.append(location_token)
                else:
                    pre_trj.append(location_tar)

            # value = dtw(pre_trj, trj, dist=lambda x, y: np.abs(x - y))
            value = dtw(pre_trj, trj, dist=lambda x, y: np.linalg.norm(np.array(x) - np.array(y)))
            hausdorff_value = max(directed_hausdorff(pre_trj, trj)[0], directed_hausdorff(trj, pre_trj)[0])
            all_hausdorff += hausdorff_value
            all_value += value[0]

        all_value /= len(testLoader)
        all_hausdorff /= len(testLoader)
        print(f'generation dtw score = {all_value}, hausdorff value = {all_hausdorff}, in {args.city}\n')
        with open(args.pre_path + '/val_result/val_generation.log', 'a') as file:
            file.write(f'generation dtw score = {all_value}, hausdorff value = {all_hausdorff}, in {args.city}\n')
    else:
        for inputToken, daytime, weekday, token_labels, mask_index, grid, day, poi, task_info in tqdm(testLoader, ncols=80):
            out1 = model(inputToken, daytime, weekday, grid, day, poi, task_info, 'trj_predict')
            out1 = out1 * mask_index.unsqueeze(-1)
            predictToken = torch.argmax(out1, dim=2).flatten()
            token_labels = token_labels.view(-1)
            mask_index, labels = mask_index.flatten(), token_labels.flatten()
            pre_trj = []
            trj = []
            for j in range(len(token_labels)):
                tar = token_labels[j].item()
                pre_token = predictToken[j].item()

                if tar <= 4:
                    continue
                location_tar = id2location[int(tar) - 5]
                trj.append(location_tar)
                if pre_token <= 4:
                    continue
                location_token = id2location[int(pre_token) - 5]
                if mask_index[j] == 1:
                    pre_trj.append(location_token)
                else:
                    pre_trj.append(location_tar)
            # value = dtw(pre_trj, trj, dist=lambda x, y: np.abs(x - y))
            value = dtw(pre_trj, trj, dist=lambda x, y: np.linalg.norm(np.array(x) - np.array(y)))
            hausdorff_value = max(directed_hausdorff(pre_trj, trj)[0], directed_hausdorff(trj, pre_trj)[0])
            all_value += value[0]
            all_hausdorff += hausdorff_value
        all_value /= len(testLoader)
        all_hausdorff /= len(testLoader)
        print(f'generation dtw score = {all_value}, hausdorff value = {all_hausdorff}, in {args.city}\n')
        with open(args.pre_path + '/val_result/val_generation.log', 'a') as file:
            file.write(f'generation dtw score = {all_value}, hausdorff value = {all_hausdorff}, in {args.city}\n')


def val_time_estimate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testData = time_estimate_Dataset(args)

    if args.city == 'beijing':
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'

    testData.load(test_input_filepath, device, args)

    testLoader = DataLoader(testData, batch_size=args.batch_size)

    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)
    vocab_size = args.vocab_size

    bert = VITwithGAT(args=args, vocab_size=args.vocab_size, adj=adj, feature=feature).to(device)

    model = TrajBERT(args=args, bert=bert, vocab_size=args.vocab_size).to(device)
    criterion = nn.MSELoss().to(device)

    if args.train == 2:
        modelpath = args.pre_path + '/output/time_estimate_VIT_' + args.city + '_update.pth'
    else:
        modelpath = args.pre_path + '/output/time_estimate_VIT_' + args.city + '.pth'
    print(modelpath)
    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1
    test_epochLoss = 0
    test_mae = 0
    test_mse = 0
    test_mape = 0

    max_val = testData.maxTime
    min_val = testData.minTime
    model.eval()
    for inputToken, daytime, weekday, time_labels, mask_attention, year, grid, poi, task_info in tqdm(testLoader, desc=f'test for tte', ncols=80):
        out = model(inputToken, daytime, weekday, year, grid, poi, task_info, task='time_estimate')
        out = out.squeeze()
        loss = criterion(out, time_labels)
        test_epochLoss += loss.item()
        out, time_labels = out.flatten(), time_labels.flatten()

        out_denorm = out * (max_val - min_val) + min_val
        time_labels_denorm = time_labels * (max_val - min_val) + min_val

        test_mae += torch.abs(out_denorm - time_labels_denorm).mean().item()
        test_mse += torch.mean((out_denorm - time_labels_denorm) ** 2).item()
        mape_values = torch.abs((out_denorm - time_labels_denorm) / (time_labels_denorm))
        test_mape += mape_values.mean().item()

    test_mae /= len(testLoader)
    test_mse /= len(testLoader)
    test_mape /= len(testLoader)
    test_epochLoss /= len(testLoader)

    # 反归一化 MAE 和 MSE
    print(f'time estimate task {args.city}, test mae = {test_mae}, mse = {test_mse}, mape = {test_mape}\n')
    with open(args.pre_path + '/val_result/val_time_estimate.log', 'a') as file:
        file.write(f'time estimate task {args.city}, test mae = {test_mae}, mse = {test_mse}, mape = {test_mape}\n')


def val_classification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('--------- adj and feature load success --------------------')

    testData = ClassificationDataset(args)

    if args.city == 'beijing':
        test_input_filepath = args.pre_path + '/data/beijing/test_bj.csv'
    else:
        test_input_filepath = args.pre_path + '/data/porto_edge1/process_data/test.csv'

    testData.load(test_input_filepath, device, args)

    testLoader = DataLoader(testData, batch_size=args.batch_size)
    vocab_size = args.vocab_size
    if args.city == 'beijing':
        adj, feature = init_adj_feature_beijing(args)
    else:
        adj, feature = init_adj_feature(args)
    bert = VITwithGAT(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)
    model = TrajBERT(args=args, bert=bert, vocab_size=args.vocab_size).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if args.train == 2:
        modelpath = args.pre_path + '/output/classification_VIT_' + args.city + '_update.pth'
    else:
        modelpath = args.pre_path + '/output/classification_VIT_' + args.city + '.pth'

    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    if args.city == 'beijing':
        task = 'binary'
    else:
        task = 'multiclass'
    accuracy = Accuracy(task=task, num_classes=args.num_class)
    f1_score = F1Score(num_classes=args.num_class, average='weighted', task=task)
    f1_micro = F1Score(num_classes=args.num_class, average='micro', task=task)
    f1_macro = F1Score(num_classes=args.num_class, average='macro', task=task)
    auroc = AUROC(num_classes=args.num_class, average='macro', task=task)

    test_epochLoss = 0
    test_acc = 0
    all_predicts = []
    all_targets = []
    all_scores = []
    model.eval()
    num_zero = 0
    num_one = 0

    for inputToken, daytime, weekday, cls_labels, attention_mask, year, grid, poi, task_info in tqdm(testLoader, ncols=80):
        out = model(inputToken, daytime, weekday, year, grid, poi, task_info, task='classification')
        cls_labels = cls_labels.squeeze(1)
        predict_cls = torch.argmax(torch.softmax(out, dim=2), dim=2).squeeze().cpu().detach()
        predict_scores = torch.softmax(out, dim=2).cpu().detach()  # Softmax scores for AUROC calculation
        loss = criterion(out.squeeze(), cls_labels)
        test_epochLoss += loss.item()
        cls_labels = cls_labels.cpu().detach()

        # 收集所有预测、得分和真实标签
        all_predicts.append(predict_cls)
        all_targets.append(cls_labels)
        all_scores.append(predict_scores.squeeze())
        if task == 'binary':
            test_acc += accuracy(predict_cls, cls_labels).item()
        else:
            test_acc += accuracy(predict_scores.squeeze(), cls_labels).item()

    # 转换为单个张量
    all_predicts = torch.cat(all_predicts)
    all_targets = torch.cat(all_targets)
    all_scores = torch.cat(all_scores, dim=0)
    test_acc /= len(testLoader)
    test_epochLoss /= len(testLoader)
    f1_micro_score = f1_micro(all_predicts, all_targets)
    f1_macro_score = f1_macro(all_predicts, all_targets)
    f1_weighted_score = f1_score(all_predicts, all_targets)
    # print(all_scores.shape, all_targets.shape)
    if task == 'binary':
        auroc_score = auroc(all_scores[:, -1], all_targets)
    else:
        auroc_score = auroc(all_scores, all_targets)

    print(f'classification task {args.city}, test acc = {test_acc}, f1_micro = {f1_micro_score}, f1_macro = {f1_macro_score}, f1_weighted = {f1_weighted_score}, auroc = {auroc_score}\n')
    with open(args.pre_path + '/val_result/val_classification.log', 'a') as file:
        file.write(f'classification task {args.city}, test acc = {test_acc}, f1_micro = {f1_micro_score}, f1_macro = {f1_macro_score}, f1_weighted = {f1_weighted_score}, auroc = {auroc_score}\n')
