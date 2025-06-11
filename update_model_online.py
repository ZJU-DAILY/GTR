import torch.nn
import torch.utils.data.dataset
from torch.utils.data import DataLoader
from Model.model import TrajBERT, VITwithGAT
from datasets import *
from train import init_adj_feature, init_adj_feature_beijing
import os
from torch.nn.utils import clip_grad_norm_


def update_model_online(args, model, task, valLoader, testLoader, train_type):
    print('------- online learning start ----------')

    '''
    task name:
    1. classification
    2. time_estimate
    3. similarity
    4. simplify
    5. imputation
    6. generation_rel
    7. destination prediction
    '''

    '''frozen the model transformer half layer'''
    # for i, transformer_block in enumerate(model.model.transformer_blocks):
    #     if i < 6:  # 前6层
    #         for param in transformer_block.parameters():
    #             param.requires_grad = False
    # Freeze parameters of the first 6 layers

    for block in model.model.blocks[:6]:  # Freeze first 6 layers
        for param in block.parameters():
            param.requires_grad = False

    model.model.ModelEmbedding.spatial_temporal_fusion.requires_grad_(True)

    # for param in model.model.ModelEmbedding.parameters():
    #     param.requires_grad = False
    '''---------------------------------------'''

    print("Available GPU count:", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion_CEL = nn.CrossEntropyLoss(ignore_index=0).to(device)

    criterion_Simplify = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    criterion_MSE = nn.SmoothL1Loss().to(device)
    creterion_TP = nn.TripletMarginLoss(margin=1, p=2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-5)

    if task == 'classification':
        best_loss = 10000
        for epoch in range(args.epochs):
            train_epochLoss = 0
            model = model.train()
            for i, (inputToken, daytime, weekday, cls_labels, attention_mask, year, grid, poi, task_info) in tqdm(enumerate(valLoader)):
                optimizer.zero_grad()
                out = model(inputToken, daytime, weekday, year, grid, poi, task_info, task='classification')
                cls_labels = cls_labels.squeeze(1)
                loss = criterion_Simplify(out.squeeze(), cls_labels)
                loss.backward()
                optimizer.step()
                train_epochLoss += loss.item()
            train_epochLoss /= len(valLoader)
            if train_epochLoss < best_loss:
                best_loss = min(best_loss, train_epochLoss)
                savePath = args.pre_path + '/output'
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                torch.save(model.state_dict(), savePath + '/classification_VIT_' + args.city + '_update.pth')

    elif task == 'time_estimate':
        best_loss = 10000
        for epoch in range(args.epochs):
            train_epochLoss = 0
            model = model.train()
            for i, (inputToken, daytime, weekday, time_labels, mask_attention, year, grid, poi, task_info) in tqdm(enumerate(valLoader), ncols=80):
                optimizer.zero_grad()
                out = model(inputToken, daytime, weekday, year, grid, poi, task_info, task='time_estimate')
                out = out.squeeze()
                loss = criterion_MSE(out, time_labels)
                loss.backward()
                optimizer.step()
                train_epochLoss += loss.item()
            test_mae = 0
            test_mse = 0
            test_mape = 0
            if args.city == 'porto':
                min_val = 1.25
                max_val = 14.75
            else:
                min_val = 1.0
                max_val = 60.0
            model.eval()
            for inputToken, daytime, weekday, time_labels, mask_attention, year, grid, poi, task_info in tqdm(
                    testLoader, desc=f'test for tte', ncols=80):
                out = model(inputToken, daytime, weekday, year, grid, poi, task_info, task='time_estimate')
                out = out.squeeze()
                # loss = criterion(out, time_labels)
                # test_epochLoss += loss.item()
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
            print(test_mae, test_mse, test_mape)

            train_epochLoss = train_epochLoss / len(valLoader)
            if train_epochLoss < best_loss:
                best_loss = min(best_loss, train_epochLoss)
                savePath = args.pre_path + '/output'
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                torch.save(model.state_dict(), savePath + '/time_estimate_VIT_' + args.city + '_update.pth')
                # if args.bert_type == 0:
                #     torch.save(model.state_dict(), savePath + '/time_estimate_update.pth')
                # else:
                #     torch.save(model.state_dict(), savePath + '/time_estimate_GAT_update.pth')

    elif task == 'similarity':
        print('----------------------similarity task update start---------------------------')
        best_loss = 10000
        for epoch in range(args.epochs):
            model = model.train()
            train_epochLoss = 0
            for i, (trj_a, trj_p, trj_n, a_day, a_week, p_day, p_week, n_day, n_week, att_a, att_p, att_n) in tqdm(enumerate(valLoader)):
                optimizer.zero_grad()
                emb_a = model(trj_a, a_day, a_week, task='similarity')
                emb_p = model(trj_p, p_day, p_week, task='similarity')
                emb_n = model(trj_n, n_day, n_week, task='similarity')
                att_a, att_p, att_n = att_a.unsqueeze(-1), att_p.unsqueeze(-1), att_n.unsqueeze(-1)
                emb_a, emb_p, emb_n = emb_a * att_a, emb_p * att_p, emb_n * att_n
                loss = creterion_TP(emb_a, emb_p, emb_n)
                loss.backward()
                optimizer.step()
                train_epochLoss += loss.item()
            train_epochLoss /= len(valLoader)
            print(train_epochLoss)
            if train_epochLoss < best_loss:
                best_loss = min(best_loss, train_epochLoss)
                savePath = args.pre_path + '/output'
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                bestmodel = model
                # if args.bert_type == 0:
                #     torch.save(model.state_dict(), savePath + '/similarity_update.pth')
                # else:
                #     torch.save(model.state_dict(), savePath + '/similarity_GAT_update.pth')

    elif task == 'simplify':
        best_loss = 10000
        for epoch in range(args.epochs):
            train_epochLoss = 0
            model = model.train()
            for i, (inputToken, daytime, weekday, simple_labels, a_mask, year, grid, poi, task_info) in tqdm(enumerate(valLoader)):
                optimizer.zero_grad()
                out = model(inputToken, daytime, weekday, year, grid, poi, task_info, task='simplify')
                out = out * a_mask.unsqueeze(-1)
                out = out.view(-1, 2)
                simple_labels = simple_labels.view(-1)
                loss = criterion_Simplify(out, simple_labels)
                loss.backward()
                optimizer.step()
                train_epochLoss += loss.item()
            if train_epochLoss < best_loss:
                best_loss = min(best_loss, train_epochLoss)
                savePath = args.pre_path + '/output'
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                torch.save(model.state_dict(), savePath + '/simplify_VIT_' + args.city + '_update.pth')
                # if args.bert_type == 0:
                #     torch.save(model.state_dict(), savePath + '/simplify_update.pth')
                # else:
                #     torch.save(model.state_dict(), savePath + '/simplify_GAT_update.pth')

    elif task == 'imputation':
        best_loss = 10000
        for epoch in range(args.epochs):
            model = model.train()
            train_epochLoss = 0
            for i, (inputToken, daytime, weekday, token_labels, mask_index, year, grid, poi, task_info) in tqdm(enumerate(valLoader)):
                optimizer.zero_grad()
                out = model(inputToken, daytime, weekday, year, grid, poi, task_info, task='imputation')
                out = out * mask_index.unsqueeze(-1)
                out = out.view(-1, args.vocab_size)
                token_labels = token_labels.view(-1)
                loss = criterion_CEL(out, token_labels)
                loss.backward()
                optimizer.step()
                train_epochLoss += loss.item()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
            # if train_epochLoss < best_loss:
            #     best_loss = min(best_loss, train_epochLoss)
        savePath = args.pre_path + '/output'
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        torch.save(model.state_dict(), savePath + '/imputation_VIT_' + args.city + '_update.pth')

    elif task == 'generation_predict':
        best_loss = 10000
        for epoch in range(args.epochs):
            model = model.train()
            train_epochLoss = 0
            for i, (inputToken, daytime, weekday, token_labels, mask_index, year, grid, poi, task_info) in tqdm(enumerate(valLoader)):
                optimizer.zero_grad()
                out1 = model(inputToken, daytime, weekday, year, grid, poi, task_info, 'trj_predict')
                out1 = out1 * mask_index.unsqueeze(-1)
                out1 = out1.view(-1, args.vocab_size)
                token_labels = token_labels.view(-1)
                lossCEL = criterion_CEL(out1, token_labels)
                lossCEL.backward()
                optimizer.step()
                train_epochLoss += lossCEL.item()
            train_epochLoss /= len(valLoader)
            if train_epochLoss < best_loss:
                best_loss = min(best_loss, train_epochLoss)
                savePath = args.pre_path + '/output'
                if not os.path.exists(savePath):
                    os.mkdir(savePath)
                torch.save(model.state_dict(), savePath + '/generation_predict_VIT_' + args.city + '_update.pth')

    elif task == 'destination prediction':
        for epoch in range(args.epochs):
            model = model.train()
            for i, (trj_token, min_list, weekday_list, label) in enumerate(valLoader):

                optimizer.zero_grad()
                output = model(trj_token, min_list, weekday_list, task)
                loss = criterion_MSE(output, label)

    '''hot the model transformer half layer'''
    # for i, transformer_block in enumerate(model.model.transformer_blocks):
    #     if i < 6:  # 前6层
    #         for param in transformer_block.parameters():
    #             param.requires_grad = True

    for block in model.model.blocks[:6]:  # Freeze first 6 layers
        for param in block.parameters():
            param.requires_grad = True

    # for param in model.model.ModelEmbedding.parameters():
    #     param.requires_grad = True
    '''---------------------------------------'''
    # torch.save(bestmodel.state_dict(), args.pre_path + '/output/' + train_type[args.update_type] + '_VIT_update.pth')
    '''save the model'''

def start_update(args):
    print('------------trajectory generation task start-------------------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.city == 'beijing':
        trainFilePath = args.pre_path + '/data/beijing/val_bj.csv'
        testFilePath = args.pre_path + '/data/beijing/test_bj.csv'
        adj, feature = init_adj_feature_beijing(args)
    else:
        trainFilePath = args.pre_path + '/data/porto_edge1/process_data/val.csv'
        testFilePath = args.pre_path + '/data/porto_edge1/process_data/test.csv'
        adj, feature = init_adj_feature(args)
    '''
        1. classification
        2. time_estimate
        3. similarity
        4. simplify
        5. imputation
        6. generation_predict
        7. destination_prediction
    '''
    train_type = ['classification', 'time_estimate', 'similarity', 'simplify', 'imputation', 'generation_predict', 'destination_prediction']
    if args.update_type == 1:
        trainData = ClassificationDataset(args)
        testData = ClassificationDataset(args)
    elif args.update_type == 2:
        trainData = time_estimate_Dataset(args)
        testData = time_estimate_Dataset(args)
    elif args.update_type == 3:
        trainData = EdgeClusterDataset(args)
        testData = EdgeClusterDataset(args)
    elif args.update_type == 4:
        trainData = simplifyDataset(args)
        testData = simplifyDataset(args)
    elif args.update_type == 5:
        trainData = imputationDataset(args)
        testData = imputationDataset(args)
    elif args.update_type == 6:
        trainData = generator_for_predict_Dataset(args)
        testData = generator_for_predict_Dataset(args)
    else:
        print('wrong update type input')
        return -1

    trainData.load(trainFilePath, device, args)
    testData.load(testFilePath, device, args)

    trainLoader = DataLoader(trainData, batch_size=args.batch_size)
    testLoader = DataLoader(testData, batch_size=args.batch_size)

    vocab_size = trainData.vocab_size
    bert = VITwithGAT(args=args, vocab_size=vocab_size, adj=adj, feature=feature).to(device)
    model = TrajBERT(args=args, bert=bert, vocab_size=trainData.vocab_size).to(device)

    modelpath = args.pre_path + '/output/' + train_type[args.update_type - 1] + '_VIT_' + args.city + '.pth'
    print(modelpath)
    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
    else:
        print('this is no checkpoint model')
        return -1

    update_model_online(args, model, train_type[args.update_type - 1], trainLoader, testLoader, train_type)


