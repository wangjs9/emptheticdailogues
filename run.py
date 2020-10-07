# coding: UTF-8
import time
import torch
import numpy as np
from classifier import train, test, predict
import classifier_model
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Emotion Classification')
parser.add_argument('--do_train', type=bool, default=True, required=False, help='Train model')
parser.add_argument('--do_test', type=bool, default=False, required=False, help='Test model')
parser.add_argument('--do_predict', type=bool, default=False, required=False, help='Predict model')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'C:/Users/csjwang/Documents/.csjwang/$empatheticdialogues/'  # 数据集

    do_train = args.do_train
    do_test = args.do_test
    do_predict = args.do_predict
    if (do_train or do_test or do_predict) == False:
        raise ValueError('At least one of `do_train` or `do_test` or `do_predict` must be True.')
    config = classifier_model.Config(dataset, dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    train_data, valid_data, test_data = build_dataset(config, do_train, do_test or do_predict)

    if do_train:
        train_iter = build_iterator(train_data, config)
        valid_iter = build_iterator(valid_data, config)
    if do_test:
        test_iter = build_iterator(test_data, config)
    if do_predict:
        data_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = classifier_model.Model(config).to(config.device)
    if do_train:
        train(config, model, train_iter, valid_iter)
    if do_test:
        test(config, model, test_iter)
    if do_predict:
        predict(config, model, data_iter)