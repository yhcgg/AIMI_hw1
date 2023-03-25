import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import shutil
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from dataAugumentation import generateDataset

def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn

def plot_accuracy(train_acc_list, val_acc_list, savePath):
    # TODO plot training and testing accuracy curve
    plt.figure(figsize=(8, 4))

    # train
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list)
    plt.title("Train_Accuracy")

    # test
    plt.subplot(1, 2, 2)
    plt.plot(val_acc_list)
    plt.title("Test_Accuracy")

    plt.savefig(savePath + '/plot_accuracy.png')
    plt.show()
    pass

def plot_f1_score(f1_score_list, savePath):
    # TODO plot testing f1 score curve
    plt.figure(figsize=(5, 3))

    plt.plot(f1_score_list)
    plt.title("Test_F1_score")

    plt.savefig(savePath + '/f1_score.png')
    plt.show()
    pass

def plot_confusion_matrix(confusion_matrix, savePath):
    # TODO plot confusion matrix
    plt.figure(figsize=(6, 4))

    sns.heatmap(confusion_matrix, annot=True, fmt=".0f",
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                yticklabels=['Actual Normal', 'Actual Pneumonia'])
    plt.savefig(savePath + '/confusion_matrix.png')
    plt.show()
    pass

def train(device, train_loader, model, criterion, optimizer):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    f1_score_list = []
    best_c_matrix = []
    best_epoch_info = {}

    for epoch in range(1, args.num_epochs+1):

        with torch.set_grad_enabled(True):		# 開啟梯度計算
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0
            for _, data in enumerate(tqdm(train_loader)):		# 顯示進度條
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()		# 把梯度置零
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')

        val_acc, f1_score, c_matrix, recall, precision = test(test_loader, model)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        f1_score_list.append(f1_score)

        if val_acc > best_acc:
            best_acc = val_acc
            best_c_matrix = c_matrix
            best_epoch_info = {'epoch_num': epoch, 'train_loss': avg_loss, 'train_acc': train_acc,
                               'recall': recall, 'precision': precision,
                               'f1_score': f1_score, 'test_acc': val_acc}
            torch.save(model.state_dict(), 'best_model_weights.pt')

    return train_acc_list, val_acc_list, f1_score_list, best_c_matrix, best_epoch_info

def test(test_loader, model):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]

        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print(f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print(f'↳ Test Acc.(%): {val_acc:.2f}%')

    return val_acc, f1_score, c_matrix, recall, precision

def getTrainImageFolder(dataset_root, doAug, resize, degree):
    if doAug:
        trans = transforms.Compose([transforms.Resize((resize, resize)),
                                    transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.Resize((resize, resize)),
                                    transforms.RandomRotation(degree, resample=False),
                                    transforms.ToTensor()])

    train_dataset = ImageFolder(root=os.path.join(dataset_root, 'train'),
                                transform=trans)
    if doAug:
        # generate augmentation dataset
        generateDataset(dataset_root)
        aug_dataset = ImageFolder(root=os.path.join(dataset_root, 'augmentation'),
                                  transform=transforms.Compose([transforms.Resize((resize, resize)),
                                                                transforms.ToTensor()]))
        train_dataset.classes.extend(aug_dataset.classes)
        train_dataset.classes = sorted(list(set(train_dataset.classes)))
        train_dataset.class_to_idx.update(aug_dataset.class_to_idx)
        train_dataset.samples.extend(aug_dataset.samples)
        train_dataset.targets.extend(aug_dataset.targets)
    return train_dataset

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--model_name', type=str, required=False, default='resnet50')

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=30)
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--augmentation', type=bool, default=False)
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader
    train_dataset = getTrainImageFolder(args.dataset, args.augmentation, args.resize, args.degree)
    test_dataset = ImageFolder(root=os.path.join(args.dataset, 'test'),
                               transform=transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                             transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # define model
    if args.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_neurons = model.fc.in_features  # 最後fc層的輸入
        model.fc = nn.Linear(num_neurons, args.num_classes)
    elif args.model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_neurons = model.fc.in_features
        model.fc = nn.Linear(num_neurons, args.num_classes)
    elif args.model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, args.num_classes)
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training
    train_acc_list, val_acc_list, f1_score_list, best_c_matrix, best_epoch_info = train(device, train_loader, model, criterion, optimizer)

    # create plot save file
    from datetime import datetime
    path = datetime.now().strftime("%m%d_%H%M%S_result")
    os.mkdir(path)

    # plot
    plot_accuracy(train_acc_list, val_acc_list, path)
    plot_f1_score(f1_score_list, path)
    plot_confusion_matrix(best_c_matrix, path)

    # print best test accuracy
    print("")
    print("Best Test Accuracy -----------------------------------------------")
    print(f'Epoch: {best_epoch_info["epoch_num"]}')
    print(f'↳ Loss: {best_epoch_info["train_loss"]}')
    print(f'↳ Training Acc.(%): {best_epoch_info["train_acc"]:.2f}%')
    print(f'↳ Recall: {best_epoch_info["recall"]:.4f}, Precision: {best_epoch_info["precision"]:.4f}, F1-score: {best_epoch_info["f1_score"]:.4f}')
    print(f'↳ Test Acc.(%): {best_epoch_info["test_acc"]:.2f}%')

    # del augmentation dataset
    if args.augmentation:
        augFile = args.dataset + '/augmentation'
        shutil.rmtree(augFile)