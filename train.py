# -*- coding: utf-8 -*-
import data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm
import time
import gc
import os
import math


class resnext50d_32x4d(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class tf_efficientnet_b0_ns(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class tf_efficientnet_b3_ns(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean", classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1

        # cross entropy
        ce_loss = nn.functional.cross_entropy(y_pred, y_true)

        # # LabelSmoothing
        # smoothing = 0.1
        # confidence = 1.0 - smoothing
        # logprobs = F.log_softmax(y_pred, dim=-1)
        # nll_loss = -logprobs.gather(dim=-1, index=y_true.unsqueeze(1))
        # nll_loss = nll_loss.squeeze(1)
        # smooth_loss = -logprobs.mean(dim=-1)
        # loss = confidence * nll_loss + smoothing * smooth_loss

        # f1 score
        y_true = torch.nn.functional.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)

        f1_loss = 1 - f1.mean()

        return ce_loss * 0.3 + f1_loss * 0.7


def train_one_epoch(
    epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None,
):
    model.train()

    running_loss = None

    pbar = tqdm(
        enumerate(train_loader), total=len(train_loader), position=0, leave=True
    )
    running_loss = 0.0
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        image_preds = model(imgs.float())
        loss = loss_fn(image_preds, image_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 80 == 79:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 100))
            running_loss = 0.0


def valid_one_epoch(epoch, model, loss_fn, val_loader, device):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs)  # output = model(input)
        # print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % 100 == 99) or ((step + 1) == len(val_loader)):
            description = f"epoch {epoch} loss: {loss_sum/sample_num:.4f}"
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    accuracy = (image_preds_all == image_targets_all).mean()
    print("validation multi-class accuracy = {:.4f}".format(accuracy))
    # f1score = f1_score(image_preds_all, image_targets_all, average='micro')

    return accuracy


def make_folder(path):
    # os.mkdir(path)
    try:
        os.mkdir(path)
    except:
        pass


def male_info(epochs, fold=1):
    train_loader, validation_loader = data_loader.main(
        "/opt/ml/input/data/train/data_path/male_info.csv", label_index=2, fold=fold
    )
    device = torch.device("cuda")
    # model = resnext50d_32x4d("resnext50d_32x4d", n_class=3, pretrained=True).to(device)
    model = tf_efficientnet_b3_ns(
        "tf_efficientnet_b3_ns", n_class=3, pretrained=True
    ).to(device)

    loss_fn = CustomLoss(classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=0.0000001
    )
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.01,  T_up=3, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min", threshold=0.00001
    # )

    save_path = "/opt/ml/input/data/model/male_info/"
    save_path += model.__class__.__name__
    make_folder(save_path)

    max_f1_score = 0.5

    for epoch in range(epochs):
        train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader, device, scheduler=scheduler
        )

        with torch.no_grad():
            f1score = valid_one_epoch(epoch, model, loss_fn, validation_loader, device)

        if f1score > max_f1_score:
            now = time.localtime()
            temp_time = "/{0:02d}{1:02d}_{2:02d}{3:02d}".format(
                now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
            )
            torch.save(
                model.state_dict(),
                save_path
                + temp_time
                + "_f1score_{}_epoch_{}.pth".format(f1score, epoch),
            )
            max_f1_score = f1score


def female_info(epochs, fold=1):
    train_loader, validation_loader = data_loader.main(
        "/opt/ml/input/data/train/data_path/female_info.csv", label_index=2, fold=fold
    )
    device = torch.device("cuda")
    # model = resnext50d_32x4d("resnext50d_32x4d", n_class=3, pretrained=True).to(device)
    model = tf_efficientnet_b3_ns(
        "tf_efficientnet_b3_ns", n_class=3, pretrained=True
    ).to(device)

    loss_fn = CustomLoss(classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=0.0000001
    )
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.01,  T_up=3, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min", threshold=0.00001
    # )

    save_path = "/opt/ml/input/data/model/female_info/"
    save_path += model.__class__.__name__
    make_folder(save_path)

    max_f1_score = 0.5

    for epoch in range(epochs):
        train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader, device, scheduler=scheduler
        )

        with torch.no_grad():
            f1score = valid_one_epoch(epoch, model, loss_fn, validation_loader, device)

        if f1score > max_f1_score:
            now = time.localtime()
            temp_time = "/{0:02d}{1:02d}_{2:02d}{3:02d}".format(
                now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
            )
            torch.save(
                model.state_dict(),
                save_path
                + temp_time
                + "_f1score_{}_epoch_{}.pth".format(f1score, epoch),
            )
            max_f1_score = f1score


def is_wear_mask(epochs):
    train_loader, validation_loader = data_loader.main(
        "/opt/ml/input/data/train/data_path/is_wear_mask.csv"
    )
    device = torch.device("cuda")
    model = resnext50d_32x4d("resnext50d_32x4d", n_class=3, pretrained=True).to(device)

    loss_fn = CustomLoss(classes=5).to(device)
    optimizer = optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=0.0001
    )
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.01,  T_up=3, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min", threshold=0.00001
    # )

    save_path = "/opt/ml/input/data/model/is_wear_mask/"
    save_path += model.__class__.__name__
    make_folder(save_path)

    max_f1_score = 0.5

    for epoch in range(epochs):
        train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader, device, scheduler=scheduler
        )

        with torch.no_grad():
            f1score = valid_one_epoch(epoch, model, loss_fn, validation_loader, device)

        if f1score > max_f1_score:
            now = time.localtime()
            temp_time = "/{0:02d}{1:02d}_{2:02d}{3:02d}".format(
                now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
            )
            torch.save(
                model.state_dict(),
                save_path
                + temp_time
                + "_f1score_{}_epoch_{}.pth".format(f1score, epoch),
            )
            max_f1_score = f1score


def age_and_gender(epochs, fold):
    train_loader, validation_loader = data_loader.main(
        "/opt/ml/input/data/train/data_path/age_and_gender.csv", fold=fold
    )
    device = torch.device("cuda")
    # model = resnext50d_32x4d("resnext50d_32x4d", n_class=2, pretrained=True).to(device)
    model = tf_efficientnet_b3_ns(
        "tf_efficientnet_b3_ns", n_class=2, pretrained=True
    ).to(device)

    loss_fn = CustomLoss(classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=0.0000001
    )
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=1, eta_max=0.01,  T_up=3, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min", threshold=0.00001
    # )

    save_path = "/opt/ml/input/data/model/age_and_gender/"
    save_path += model.__class__.__name__
    make_folder(save_path)

    max_f1_score = 0.9

    for epoch in range(epochs):
        train_one_epoch(
            epoch, model, loss_fn, optimizer, train_loader, device, scheduler=scheduler
        )

        with torch.no_grad():
            f1score = valid_one_epoch(epoch, model, loss_fn, validation_loader, device)

        if f1score > max_f1_score:
            now = time.localtime()
            temp_time = "/{0:02d}{1:02d}_{2:02d}{3:02d}".format(
                now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min
            )
            torch.save(
                model.state_dict(),
                save_path
                + temp_time
                + "_f1score_{}_epoch_{}.pth".format(f1score, epoch),
            )
            max_f1_score = f1score


if __name__ == "__main__":
    # gc.collect()
    # torch.cuda.empty_cache()
    # epochs = 10
    # is_wear_mask(epochs)

    for i in range(1, 4):
        gc.collect()
        torch.cuda.empty_cache()
        epochs = 3
        age_and_gender(epochs, fold=i)

        gc.collect()
        torch.cuda.empty_cache()
        epochs = 3
        male_info(epochs, fold=i)

        gc.collect()
        torch.cuda.empty_cache()
        epochs = 3
        female_info(epochs, fold=i)
