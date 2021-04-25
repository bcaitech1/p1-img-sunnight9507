# -*- coding: utf-8 -*-
from glob import glob
import numpy as np
import pandas as pd
import os


def is_wear_mask(file_list: list):
    """
    마스크 착용 유무 확인하는 dataset 생성

    mask1-5: "m"
    incorrect_mask: "i"
    normal: "n"

    Args:
        file_list (list)
    """
    to_categorical_data = {"m": 0, "i": 1, "n": 2}
    file_list_length = len(file_list)

    to_csv_file = pd.DataFrame(
        np.array(
            [
                [file_list[i], to_categorical_data[file_list[i].split("/")[-1][:-4][0]]]
                for i in range(file_list_length)
            ]
        )
    )

    to_csv_file.to_csv(
        "/opt/ml/input/data/train/data_path/is_wear_mask.csv", index=False, header=False
    )


def age2categorical(age):
    if age <= 20:
        return 0
    elif age <= 29:
        return 0
    elif age <= 49:
        return 1
    elif age <= 58:
        return 1
    else:
        return 2


def age_and_gender(file_list: list):
    """
    성별과 나이 dataset

    Args:
        file_list (list): [description]
    """
    file_list_length = len(file_list)
    to_csv_file = []
    gender2categorical = {"male": 0, "female": 1}

    for i in range(file_list_length):
        num, gender, nationality, age = file_list[i].split("/")[-2].split("_")
        if gender == "female":
            to_csv_file.append(
                [file_list[i], gender2categorical[gender], age2categorical(int(age))]
            )

    to_csv_file = pd.DataFrame(np.array(to_csv_file))
    to_csv_file.to_csv(
        "/opt/ml/input/data/train/data_path/female_info.csv", index=False, header=False,
    )


if __name__ == "__main__":
    train_image_path = "/opt/ml/input/data/train/images"
    file_list = glob(os.path.join(train_image_path, "*/*.[jp]*[pne]g"))  # jpg, png
    print(len(file_list))
    is_wear_mask(file_list)
    age_and_gender(file_list)
