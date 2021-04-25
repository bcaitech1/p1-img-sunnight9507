# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from albumentations import (
    RandomGamma,
    HorizontalFlip,
    VerticalFlip,
    IAAPerspective,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    RandomCrop,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    Flip,
    OneOf,
    Compose,
    Normalize,
    Cutout,
    ShiftScaleRotate,
    CenterCrop,
    Resize,
    GridDropout,
    ColorJitter,
    MultiplicativeNoise,
)


def get_img(path):
    img_bgr = cv2.imread(path)  # .astype('float32')
    img_rgb = img_bgr[:, :, ::-1]
    return img_rgb


class MyDataset(Dataset):
    def __init__(self, df, transforms=None, output_label=True, label_index=1):
        super().__init__()
        self.df = df.copy()
        self.transforms = transforms
        self.output_label = output_label
        self.label_index = label_index

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        if self.output_label:
            target = self.df.iloc[index][self.label_index]
        path = "{}".format(self.df.iloc[index][0])

        image = Image.open(path)
        image = image.convert("RGB")
        image = np_array = np.asarray(image)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if self.output_label:
            return image, target
        else:
            return image


def train_data_loader(df, label_index):
    train_ds = MyDataset(
        df,
        transforms=get_train_transforms(),
        output_label=True,
        label_index=label_index,
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    return train_loader


def validation_data_loader(df, label_index):
    validation_ds = MyDataset(
        df,
        transforms=get_valid_transforms(),
        output_label=True,
        label_index=label_index,
    )

    validation_loader = DataLoader(validation_ds, batch_size=64, shuffle=True)
    return validation_loader


def test_data_loader(df):
    test_ds = MyDataset(df, transforms=get_valid_transforms(), output_label=False,)

    test_loader = DataLoader(test_ds, batch_size=64)
    return test_loader


def get_train_transforms():
    return Compose(
        [
            Resize(512, 384),
            CenterCrop(height=380, width=350),
            # HorizontalFlip(p=0.5),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            CLAHE(p=0.5),
            # OneOf(
            #     [
            #         # IAAPerspective()
            #     ],
            #     p=1,
            # ),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
        ],
    )


def get_valid_transforms():
    return Compose(
        [
            Resize(512, 384),
            CenterCrop(height=380, width=350),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def main(data_path, label_index=1, fold=1):
    dataset = pd.read_csv(data_path)

    # train_split = 0.7
    # validation_split = 0.3
    # train_index = int(dataset_size * train_split)

    dataset_size = len(dataset)
    split_ratio = 0.4
    index = int(dataset_size * split_ratio)

    if fold == 1:
        train_data, validation_data = dataset[: index * 2], dataset[index * 2 :]
    elif fold == 2:
        train_data, validation_data = (
            dataset.iloc[[*range(index), *range(int(index * (1.5)), dataset_size)], :],
            dataset[index : int(index * 1.5)],
        )
    else:
        new_index = int(index / 2)
        train_data, validation_data = dataset[new_index:], dataset[:new_index]

    # print(train_data.shape)
    # print(validation_data.shape)

    return (
        train_data_loader(train_data, label_index),
        validation_data_loader(validation_data, label_index),
    )


# main("/opt/ml/input/data/train/data_path/is_wear_mask.csv", fold=1)
# main("/opt/ml/input/data/train/data_path/is_wear_mask.csv", fold=2)
# main("/opt/ml/input/data/train/data_path/is_wear_mask.csv", fold=3)
