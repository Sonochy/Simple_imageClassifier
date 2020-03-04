# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
import os
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import utils

#trainData:testData = 8:2
SLASH = 0.2 # percentage of test(validation) data
# BATCH:勾配更新毎のサンプル数を示す
BATCH_SIZE = 20
# EPOCH:訓練データ配列の反復回数,一度に処理する学習回数
EPOCH = 20
# Patiemce:EarlyStopping参照
PATIENCE = 100
MONITOR = 'val_acc'
# model.compileの最適化アルゴリズム指定
OPTIMIZER = 'rmsprop'

# get datetime
date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# parsing arguments
def parse_args():
    #コマンドラインオプションの定義
    # https://docs.python.org/ja/3/library/argparse.html
    parser = argparse.ArgumentParser(description='image classifier')
    # --dataを定義,
    # dest:parse_args() が返すオブジェクトに追加される属性名。e.g.:args.data_dir
    # default:コマンドラインに引数がなかった場合に生成される値。
    parser.add_argument('--data', dest='data_dir', default='data')
    parser.add_argument('--list', dest='list_dir', default='list')
    parser.add_argument('--model_name', dest='model_name', default = date_str)
    parser.add_argument('--epoch', dest='epoch', default = 20)
    args = parser.parse_args()
    return args

args = parse_args()
# class.lst, train.lst and test.lst existが存在するかチェック
EPOCH = args.epoch

if utils.exist_list(args.list_dir):
    # format():変数の文字列への埋め込み, e.g.:''...{0}..'.format()
    # {0}は変数0番目の意味
    # 各リストが存在していた場合、リストから分類クラスの
    print('Lists already exist in ./{0}. Use these lists.'.format(args.list_dir))
    classes, train_list, test_list = utils.load_lists(args.list_dir)
else:
    print('Lists do not exist. Create list from ./{0}.'.format(args.data_dir))
    # いずれかのリストが存在しない場合create_list()を行い作成する
    # utilsで読み込んだdata以下のファイルツリーを classes,train_list, test_listに代入
    # e.g:classes={Cargo, Tanker}, data_list={./data/Cargo/3.jpg,.....}, ...
    classes, train_list, test_list = utils.create_list(args.data_dir, args.list_dir, SLASH)
# training用の画像とその画像のラベルを与える
train_image, train_label = utils.load_images(classes, train_list)
# test用の画像とその画像のラベルを与える
test_image, test_label = utils.load_images(classes, test_list)

# convert to numpy.array
x_train = np.asarray(train_image)
y_train = np.asarray(train_label)
x_test = np.asarray(test_image)
y_test = np.asarray(test_label)

print('train samples: ', len(x_train))
print('test samples: ', len(x_test))

NUM_CLASSES = len(classes)


# building the model
print('building the model ...')

model = Sequential()

# Keras1.x → 2.x : Convert2D(32,3,3) → Conv2D(32,(3,3)) , border_mode → padding
# Conv2D:2次元入力をフィルターする畳み込み層．
# 3*3フィルターで32
# .shape:配列の大きさを返す e.g.a={2,3} →a.shape==(2,3)
# .shape[1:]: e.g.a={2,3} →a.shape[1:] == (3,)
model.add(Conv2D(32, (3, 3), padding='valid',
                        input_shape=x_train.shape[1:]))
# ReLU(Rectified Linear Unit) 活性化関数の一つ
# ReLU:入力した値が0以下のとき0になり、1より大きいとき入力をそのまま出力
model.add(Activation('relu'))
#
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# Max_Pooling2D : 画像から2*2領域を切り出してその範囲の最大値を利用する
model.add(MaxPooling2D(pool_size=(2, 2)))
# https://deepage.net/deep_learning/2016/10/17/deeplearning_dropout.html
# Dropout:過学習を防ぐ。一定の確立でニューロンを無視して学習を進める正規化の一種
# CNNにおける正規化:モデルの複雑さにペナルティを与える
model.add(Dropout(0.02))

model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.02))

# Flatten:入力の平滑化
model.add(Flatten())
# Dense:全結合
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(NUM_CLASSES))
# softmax: 要素中の最大値を1としてそれを基準に各値を変換する。e.g. {10,2,1}→{1,0.0003,0.00...}
model.add(Activation('softmax'))

# RMSprop:勾配降下法(Optimizer)の手法の一つ。
# 一次元で説明すると、波形の最大値最小値となる部分を探す i.e. でっぱりを探す
#rmsplop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# save model image
plot_model(model, to_file='model.png')

# EarlyStopping:Callback patience:判断回数,monitor:判断対象の値,mode:max,min,auto
es_cb = EarlyStopping(monitor=MONITOR, patience=PATIENCE, verbose=1, mode='auto')

# ModelCheckpoint:Callback for save best model
os.mkdir(date_str)
mc_cb = ModelCheckpoint(os.path.join('.',date_str,'acc_best_' + args.model_name + '.model'),
                                        monitor='val_acc',
                                        verbose = 1,
                                        save_best_only = True)

# TensorBoard Callback
tb_cb = TensorBoard(log_dir=os.path.join('.',date_str),
histogram_freq=1, write_graph=True, write_images=True)

# training
hist = model.fit(x_train, y_train,
                 batch_size=BATCH_SIZE,
                 verbose=1,
                 #Keras1.x → 2.x : nb_epoch → epochs
                 epochs=EPOCH,
                 validation_data=(x_test, y_test),
                 callbacks = [es_cb, mc_cb, tb_cb])




# save model
model.save(os.path.join('.', date_str, 'IJS_' + args.model_name + '.model'))

# plot learning cuarv
print(hist.history.keys())
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

len_epochs = len(loss)
fig, ax1 = plt.subplots()
ax1.plot(range(len_epochs), loss, label='loss', color='b')
ax1.plot(range(len_epochs), val_loss, label='val_loss', color='g')
leg = plt.legend(loc='upper left', fontsize=10)
leg.get_frame().set_alpha(0.5)
ax2 = ax1.twinx()
ax2.plot(range(len_epochs), acc, label='acc', color='r')
ax2.plot(range(len_epochs), val_acc, label='val_acc', color='m')
leg = plt.legend(loc='upper right', fontsize=10)
leg.get_frame().set_alpha(0.5)
plt.grid()
plt.title('learinig cuarv')
plt.xlabel('epoch')
plt.savefig(os.path.join('.', date_str,'graph_' + args.model_name + '.png'))
plt.show()
