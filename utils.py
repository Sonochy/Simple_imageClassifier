# -*- coding: utf-8 -*-
from __future__ import print_function

import random
import os
import cv2
import numpy as np

IMG_SIZE = 128

#要素から'.DS_Store'を削除する
def remove_DS_store(array):
    if '.DS_Store' in array:
        array.remove('.DS_Store')
    return array


# checking existence of the class, train and test lists
def exist_list(list_dir):
    # os.path.exists() return True or False
    # if class.lst, train.lst and test.lst exist, return exists as True
    # '\'は行の途中で折り返す時に使う
    exists = os.path.exists(os.path.join('.', list_dir, 'class.lst')) \
             and os.path.exists(os.path.join('.', list_dir, 'train.lst')) \
             and os.path.exists(os.path.join('.', list_dir, 'test.lst'))
    return exists

# create and return the class, train and test lists
def create_list(data_dir, list_dir, slash):
#　!bag 解決 ".DS_Store"が存在する場合、data_listに".DS_Store"が混入する
    classes = os.listdir(os.path.join('.', data_dir))
    ## https://qiita.com/clarinet758/items/43fdc786685e7c13abf5
    # data以下に".DS_Store"が存在する場合classesから消去　→　記事にしていいかも
    remove_DS_store(classes)
    #
    data_list = []
# enumerate():インデックスとともにループ
# https://qiita.com/wwacky/items/27402b5aa27b34423ec0
# i:0           i:1
# cls:Cargo     cls:Tanker
    for i, cls in enumerate(classes):
# files に cls(e.g.'Cargo')以下のファイル一覧を代入
# e.g. files == ['1.jpg', '3.jpg',.....]
# !bag 解決　e.g. : ./data/Cargo/.DS_store を除外したい
        files = os.listdir(os.path.join('.', data_dir, cls))
        #
        remove_DS_store(files)
        #
# e.g. f == '1.jpg'
# data_list == ['./data/Cargo/1.jpg','....']
        for f in files:
            data_list.append(os.path.join('.', data_dir, cls, f))

    split_index = int(len(data_list) * slash)
    random.shuffle(data_list)
    train_list = data_list[split_index:]
    test_list = data_list[:split_index]
    try:
        os.mkdir(list_dir)
    except OSError:
        print('Directory ./{0} already exists.'.format(list_dir))
    f = open(os.path.join('.', list_dir, 'class.lst'), 'w')
    f.write('\n'.join(classes))
    f.close()
    f = open(os.path.join('.', list_dir, 'train.lst'), 'w')
    f.write('\n'.join(train_list))
    f.close()
    f = open(os.path.join('.', list_dir, 'test.lst'), 'w')
    f.write('\n'.join(test_list))
    f.close()
    return classes, train_list, test_list

# load the class, train and test lists
def load_lists(list_dir):
    # os.path.join : パスの結合を行う
    f = open(os.path.join('.', list_dir, 'class.lst'), 'r')
    # sprit():改行を区切りにして分ける
    # f.read().split() : 区切り毎にread
    # e.g.:classes={Cargo,Tanker}
    classes = f.read().split()
    f.close()
    f = open(os.path.join('.', list_dir, 'train.lst'), 'r')
    train_list = f.read().split()
    f.close()
    f = open(os.path.join('.', list_dir, 'test.lst'), 'r')
    test_list = f.read().split()
    f.close()
    return classes, train_list, test_list

# load images and add labels
def load_images(classes, data_list):
    images = []
    labels = []
    num_classes = len(classes)
    # data_list内のファイルパスを拾う
    for data in data_list:
        # .DS_Store を除外する for mac
        if data == ".DS_Store":
            continue
        # OpenCVを用いて画像dataを読み込む
        img = cv2.imread(data)
        # check imread
        if not img is None:
            # IMG_SIZEで定義されたサイズに画像をリサイズ
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # 中身を0~1にする for ニューラルネットワーク用
            img = img/255.0
            # .append()配列の要素追加...[img]
            images.append(img)
            # np.zeros:要素を0とする配列を分類するクラスの数だけの生成
            lbl = np.zeros(num_classes)
            # dirname(data):データのディレクトリパスを返す
            # basename(path):pathの末尾を返す
            # e.g.:data == "./data/Cargo/1.jpg"
            # dirname(data) == "./data/Cargo"
            # basename(dirname(data))=="Cargo"
            lbl[classes.index(os.path.basename(os.path.dirname(data)))] = 1
            labels.append(lbl)
    return images, labels
