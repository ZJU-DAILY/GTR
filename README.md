# GTR 
This is a PyTorch implementation of the paper: GTR: General Multi-View Trajectory Representation Learning via Pre-train and Fine-tune

![Overview](./framework.png)

## Environment Preparation
    -Python 3.8.18
    -CentOS 7.0
    -Pytorch 1.13.1
    -A Nvidia GPU with cuda 11.7
    -Please refer to the 'requirement.txt' to install all required packages

## Datasets Description & Preprocessing
You can access our data [here](https://pan.quark.cn/s/e92cc7ffe980)
The folder contains a total of two datasets, Beijing and porto

### Pre-training

```shell
python main.py -batch_size 64 -train 0 -val 0 -epochs 10 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'pretrain_mlm'

python main.py -batch_size 64 -train 0 -val 0 -epochs 10 -bert_type 2 -max_len 70 -vocab_size 44646 -city 'porto' -task 'pretrain_mlm'
```

### Fine-tuning

You can fine-tune the model with the following commands. If you want to run the model in porto, please change the vocab_size and city name.
```shell
python main.py -batch_size 64 -train 0 -val 1 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'time_estimate'

python main.py -batch_size 64 -train 0 -val 2 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'simplify'

python main.py -batch_size 64 -train 0 -val 3 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'classification'

python main.py -batch_size 64 -train 0 -val 4 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'imputation'

python main.py -batch_size 64 -train 0 -val 5 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'trj_predict'
```

### Online Updating
You can update the model with the following commands. If you want to run the model in porto, please change the vocab_size and city name.

```shell
#imputation
python main.py -batch_size 64 -train 2 -val 1 -epochs 20 -bert_type 2 -max_len 70 -vocab_size 40311 -grid_size 10000 -city 'beijing' -num_class 2 -task 'imputation' -update_type 5

#time_estimate
python main.py -batch_size 64 -train 2 -val 1 -epochs 20 -bert_type 2 -max_len 70 -vocab_size 40311 -grid_size 10000 -city 'beijing' -num_class 2 -task 'time_estimate' -update_type 2

#classification
python main.py -batch_size 64 -train 2 -val 1 -epochs 20 -bert_type 2 -max_len 70 -vocab_size 40311 -grid_size 10000 -city 'beijing' -num_class 2 -task 'classification' -update_type 1

#simplify
python main.py -batch_size 64 -train 2 -val 1 -epochs 20 -bert_type 2 -max_len 70 -vocab_size 40311 -grid_size 10000 -city 'beijing' -num_class 2 -task 'simplify' -update_type 4

#generation
python main.py -batch_size 64 -train 2 -val 1 -epochs 20 -bert_type 2 -max_len 70 -vocab_size 40311 -grid_size 10000 -city 'beijing' -num_class 2 -task 'trj_predict' -update_type 6
```


### Evaluation
You can evaluate the model performance with the following commands. If you want to run the model in porto, please change the vocab_size and city name.

```shell
python main.py -batch_size 64 -train 1 -val 1 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'similarity'
python main.py -batch_size 64 -train 1 -val 2 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'imputation'
python main.py -batch_size 64 -train 1 -val 3 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'simplify'
python main.py -batch_size 64 -train 1 -val 4 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'trj_predict'
python main.py -batch_size 64 -train 1 -val 5 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'classification'
python main.py -batch_size 64 -train 1 -val 6 -epochs 50 -bert_type 2 -max_len 70 -vocab_size 40311 -city 'beijing' -task 'time_estimate'
```

## Citation
If you use our code for reseach work, please cite our paper as below:
```
@article{,
  title={GTR: General Multi-View Trajectory Representation Learning via Pre-train and Fine-tune},
  author={Xiangheng Wang, Ziquan Fang, Danlei Hu, Chenglong Huang, Lu Chen, Yunjun Gao},
  year={2025},
}
```
