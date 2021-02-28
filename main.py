import argparse
import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report

from bert_sent_emb import get_data
from model import Model

start_time = time.time()
criterion = nn.CrossEntropyLoss()

max_accuracy_epoch = 0
max_val_accuracy = 0


def train(epoch, model, optimizer, train_x, train_y, device, batch_size):
    model.train()

    train_size = len(train_x)
    train_idx = np.random.permutation(train_size)
    loss_accum = 0
    for i in range(0, train_size, batch_size):
        idx = train_idx[i: i+batch_size]
        output = model([train_x[i] for i in idx])
        label = train_y[idx]
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    print('Epoch : ', epoch, 'loss_accum training: ', loss_accum, 'Time : ', int(time.time() - start_time))


def pass_data_iteratively(model, x, batch_size=32):
    outputs = []
    total_size = len(x)
    ixd = np.arange(total_size)
    for i in range(0, total_size, batch_size):
        idx_tmp = ixd[i: i+batch_size]

        with torch.no_grad():
            output = model([x[i] for i in idx_tmp])
        outputs.append(output)

    return torch.cat(outputs)


def evaluate(epoch, model, scheduler, train_x, train_y, dev_x, dev_y, test_x, test_y, device):
    model.eval()

    output = pass_data_iteratively(model, train_x)
    pred_train = output.max(1, keepdim=True)[1]
    # correct = pred_train.eq(train_y.view_as(pred_train)).sum().cpu().item()
    # acc_train = correct / float(len(train_y))
    train_y = train_y.detach().cpu().numpy()
    pred_train = pred_train.squeeze().detach().cpu().numpy()
    acc_train = accuracy_score(train_y, pred_train)

    output = pass_data_iteratively(model, dev_x)
    pred_val = output.max(1, keepdim=True)[1]
    # correct = pred_val.eq(dev_y.view_as(pred_val)).sum().cpu().item()
    # acc_val = correct / float(len(dev_y))
    dev_y = dev_y.detach().cpu().numpy()
    pred_val = pred_val.squeeze().detach().cpu().numpy()
    acc_val = accuracy_score(dev_y, pred_val)
    # print(classification_report(dev_y, pred_val))

    global max_accuracy_epoch, max_val_accuracy
    if acc_val > max_val_accuracy:
        max_accuracy_epoch = epoch
        max_val_accuracy = acc_val
        if test_x:
            output = pass_data_iteratively(model, test_x)
            test_y = output.max(1, keepdim=True)[1].squeeze().detach().cpu().numpy()
    else:
        scheduler.step()

    print("accuracy train: %f val: %f, max val: %f" % (acc_train, acc_val, max_val_accuracy), flush=True)

    return test_y


def main():

    parser = argparse.ArgumentParser(description='Pytorch for RTER')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--hidden_dim', type=int, default=300, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout (default: 0.3)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay (default: 0.3)')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')

    args = parser.parse_args()

    print(args, flush=True)
    
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print('device : ', device, flush=True)

    data = get_data()

    labels_list = ['pos', 'neg']
    num_classes = len(labels_list)

    train_x, train_y = data['train']
    dev_x, dev_y = data['dev']

    flag = False
    if flag:
        dev_prop = .2
        train_dev_x = train_x + dev_x
        train_dev_y = train_y + dev_y

        train_x = []
        train_y = []
        dev_x = []
        dev_y = []

        train_dev_x_size = len(train_dev_x)
        train_dev_idx = np.random.permutation(train_dev_x_size)
        dev_size = int(train_dev_x_size * dev_prop)
        train_size = train_dev_x_size - dev_size
        train_idx = train_dev_idx[:train_size]
        dev_idx = train_dev_idx[train_size:]

        for i in train_idx:
            train_x.append(train_dev_x[i])
            train_y.append(train_dev_y[i])

        for i in dev_idx:
            dev_x.append(train_dev_x[i])
            dev_y.append(train_dev_y[i])

    print('train size :', len(train_x), len(train_y))
    print('dev size :', len(dev_x), len(dev_y))

    test_x = None
    if 'test' in data:
        test_x = data['test'][0]
        print('test size :', len(test_x))

    input_dim = train_x[0][0].shape[1]

    model = Model(args, input_dim, device, num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    print(model)
    print(optimizer, flush=True)

    train_y = torch.tensor(train_y).long().to(device)
    dev_y = torch.tensor(dev_y).long().to(device)

    test_y = None
    test_y = evaluate(0, model, scheduler, train_x, train_y, dev_x, dev_y, test_x, test_y, device)
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, train_x, train_y, device, args.batch_size)
        test_y = evaluate(epoch, model, scheduler, train_x, train_y, dev_x, dev_y, test_x, test_y, device)
        print('')
        if epoch > (max_accuracy_epoch + args.early_stop):
            break

    test_labels = []
    if test_y is not None:
        for i in test_y:
            test_labels.append(labels_list[i])
        test_labels = np.array(test_labels)
        np.savetxt('output/' + str(max_val_accuracy) + '.txt', test_labels, fmt="%s")
        print('output saved')


if __name__ == '__main__':
    main()
