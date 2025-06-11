import argparse
from train import *
from validate import *
from update_model_online import start_update
pre_path = os.path.dirname(os.path.abspath(__file__))

args = argparse.ArgumentParser()
args.add_argument('-hidden', type=int, default=3072, help='hidden')
args.add_argument('-vocab_size', type=int, default=40311, help='vocab_size beijing = 40311, porto = 30000')
args.add_argument('-max_len', type=int, default=70, help='the length of the input')
args.add_argument('-d_model', type=int, default=768, help='the dimension of the Model')
args.add_argument('-dropout', type=float, default=0.2, help='the value of dropout')
args.add_argument('-n_layers', type=int, default=12, help='the numbers of encoderLayers')
args.add_argument('-n_heads', type=int, default=12, help='the numbers of Attention Heads')
args.add_argument('-output_size', type=int, default=12, help='output size')
args.add_argument('-n_clusters', type=int, default=12, help='the numbers of cluster')

args.add_argument('-grid_size', type=int, default=10000, help='grid size')
args.add_argument('-num_class', type=int, default=3, help='class for classification')
args.add_argument('-settings', type=str, default="/data/porto_edge1/setting.json", help='lng and vocab_size')
args.add_argument('-vocab_file1', type=str, default="/data/porto_edge1/process_data/vocab.json", help='vocab')
args.add_argument('-vocab_file', type=str, default="/data/porto_edge1/setting.json", help='setting')
args.add_argument('-epochs', type=int, default=30, help='epochs of train')
args.add_argument('-lr', type=float, default=0.00001, help='learning rate')
args.add_argument('-batch_size', type=int, default=16, help='batch size')
args.add_argument('-pre_path', type=str, default=pre_path, help='cur_path')
args.add_argument('-bert_type', type=int, default=1, help='0: bert, 1: bert with GAT, 2:BERT with GAT2')
args.add_argument('-update_type', type=int, default=3, help='1: classification, 2: tte, 3: similarity, 4:simplify, 5:imputation, 6:generation')

args.add_argument('-task', type=str, default='pretrain_mlm', help='')
args.add_argument('-feature_dim', type=int, default=4, help='feature_dim')
args.add_argument('-train_file', type=str, default="/data/porto/data/porto_train.csv", help='setting')
args.add_argument('-test_file', type=str, default="/data/porto/data/porto_test.csv", help='setting')
args.add_argument('-val_file', type=str, default="/data/porto/data/porto_val.csv", help='setting')

args.add_argument('-train', type=int, default=3, help='0 for train, 1 for validate, 2 for update online learning ')
args.add_argument('-val', type=int, default=4, help='0 - 10 switch the task')
args.add_argument('-city', type=str, default='beijing', help='beijing or porto')
args.add_argument('-top_k', type=int, default=0)
args.add_argument('-task_emb', type=int, default=0)
args.add_argument('-device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Training device to use: "cpu" or "cuda".')

if __name__ == '__main__':
    args = args.parse_args()
    if args.task == 'pretrain_mlm':
        args.task_emb = 1
    elif args.task == 'classification':
        args.task_emb = 2
    elif args.task == 'time_estimate':
        args.task_emb = 3
    elif args.task == 'simplify':
        args.task_emb = 4
    elif args.task == 'imputation':
        args.task_emb = 5
    elif args.task == 'trj_predict':
        args.task_emb = 6

    if args.city == 'beijing':
        args.num_class = 2
    elif args.city == 'porto':
        args.num_class = 3

    for k, v in args._get_kwargs():
        print("{0} = {1}".format(k, v))
    print("-" * 10 + "start training" + "-" * 10)
    if args.train == 0:
        if args.val == 0:
            pretrain_mlm_triplet(args)     # pretrain_model
        elif args.val == 1:
            train_time_estimate_model(args)  # time estimate
        elif args.val == 2:
            train_simplify_model(args)  # trajectory simplify
        elif args.val == 3:
            train_classification_model(args)  # trajectory classification
        elif args.val == 4:
            train_imputation_model(args)  # trajectory imputation
        elif args.val == 5:
            train_generation_predict_model(args)  # trajectory generation
    elif args.train == 1:
        if args.val == 1:
            print('similarity task validation')
            val_similarity_compute(args)
        elif args.val == 2:
            print('imputation task validation')
            val_imputation_compute(args)
        elif args.val == 3:
            print('simplify task validation')
            val_simplify_model(args)
        elif args.val == 4:
            print('generation task validation')
            val_generation_compute(args)
        elif args.val == 5:
            print('classification task validation')
            val_classification(args)
        elif args.val == 6:
            print('time estimate task validation')
            val_time_estimate(args)
    elif args.train == 2:
        if args.val == 1:
            start_update(args)
        elif args.val == 2:
            print('imputation task validation')
            val_imputation_compute(args)
        elif args.val == 3:
            print('simplify task validation')
            val_simplify_model(args)
        elif args.val == 4:
            print('generation task validation')
            val_generation_compute(args)
        elif args.val == 5:
            print('classification task validation')
            val_classification(args)
        elif args.val == 6:
            print('time estimate task validation')
            val_time_estimate(args)
    else:
        print(f'wrong train: {args.train}, val: {args.val}')


