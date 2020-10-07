# coding: UTF-8
import numpy as np
import torch, time, os
import torch.nn as nn
from torch import optim
from sklearn import metrics
from utils import get_time_dif
from transformers import get_linear_schedule_with_warmup

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter):
    start_time = time.time()

    if os.path.exists(config.save_path):
        model.load_state_dict(torch.load(config.save_path)['model_state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    total_steps = len(train_iter) * config.learning_rate
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    total_batch = 0
    dev_best_loss = float('inf')
    dev_last_loss = float('inf')
    no_improve = 0
    flag = False
    model.train()

    for epoch in range(config.epoch_num):
        print('Epoch [{}/{}]'.format(epoch + 1, config.epoch_num))
        for i, (trains, labels) in enumerate(train_iter):
            model.zero_grad()
            loss, logits = model(trains, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(logits.data, -1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                train_loss = loss.item()
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    state = {
                        'model_state_dict': model.state_dict(),
                    }
                    dev_best_loss = dev_loss

                    torch.save(state, config.save_path)
                    improve = '*'
                    del state
                else:
                    improve = ''

                if dev_last_loss > dev_loss:
                    no_improve = 0
                elif no_improve % 2 == 0:
                    no_improve += 1
                    scheduler.step()
                else:
                    no_improve += 1

                dev_last_loss = dev_loss

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, train_loss, train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if no_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break

        scheduler.step()

def test(config, model, test_iter):
    # test
    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    with open(config.save_dic + 'report.txt', 'w') as f:
        f.write(msg.format(test_loss, test_acc))
        f.write('\n')
        f.write("Precision, Recall and F1-Score...")
        f.write(str(test_report))
        f.write('\n')
        f.write("Confusion Matrix...\n")
        f.write(str(test_confusion))

def predict(model, data_iter):
    logits_list = torch.FloatTensor(0,8).to(data_iter.device)
    with torch.no_grad():
        for texts, labels in data_iter:
            logits = model(texts, labels, pred=True)
            logits_list = torch.cat((logits_list, logits), dim=0)
    return logits_list

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in data_iter:
            loss, logits = model(texts, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(logits.data, -1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

