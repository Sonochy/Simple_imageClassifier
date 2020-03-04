# -*- coding: utf-8 -*-
# __future__ is use python2 lib if cannot use python3 lib
from __future__ import print_function
# for load_model
from keras.models import load_model
# コマンドライン引数の解析モジュール
import argparse
# utils is loading image units
import utils
# file操作
import os
# numpy 数値計算用ライブラリ 主に配列計算用
import numpy as np
# parsing arguments
# https://qiita.com/dodo5522/items/6ec2b6d83287add6c185
def parse_args():
    parser = argparse.ArgumentParser(description='This script is image classifier')
# --data　用途不明　不要?
    # parser.add_argument('--data', dest='data_dir', default='data')
    parser.add_argument('--list', dest='list_dir', default='list')
    parser.add_argument('--model', dest='model_name', required=True)
    args = parser.parse_args()
    return args

args = parse_args()
if utils.exist_list(args.list_dir):
    print('Lists exist in ./{0}. Use the test list.'.format(args.list_dir))
    classes, _, test_list = utils.load_lists(args.list_dir)
else:
    print('Lists do not exist.')
    exit(0)

test_image, test_label = utils.load_images(classes, test_list)

# convert to numpy.array
x_test = np.asarray(test_image)
y_test = np.asarray(test_label)

print('test samples: ', len(x_test))

model = load_model(args.model_name)

#入力サンプルに対する予測値の出力を生成する。
pred = model.predict(x_test, batch_size=32, verbose=0)

correct = 0
incorrect = 0
incorrects = []
# enumerate() インデックスとともにループ
for i, test in enumerate(test_list):
    print(test)
    answer = os.path.basename(os.path.dirname(test))
    predict = classes[np.argmax(pred[i])]
    if answer == predict:
        print('Correct!!!')
        #
        correct += 1
        #
    else:
        print('Incorrect...')
        #
        incorrect += 1
        incorrects.append(os.path.join(os.path.basename(os.path.dirname(test)),os.path.basename(test)))
        #
    print('answer:', os.path.basename(os.path.dirname(test)))
    print('predict:', classes[np.argmax(pred[i])])
    for j in range(len(classes)):
        print('{0}: {1:4.2f} '.format(classes[j], pred[i][j]), end='')
    print('\n------------------------------------\n')
##
print('| Num_of_Correct_data:',correct)
print('| Num_of_Incorrect_data:',incorrect)
if len(incorrects)==0:
    print('\n| Incorrect test data are None!!')
else:
    incorrects.sort()
    print('\n| Incorrect_data_list:',incorrects)

print('\n| Accuracy:{0} %'.format(correct/(correct+incorrect)*100))
print('\n')
    ##
