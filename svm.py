#coding:utf-8
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import sys
from mfcc import *
import glob
import csv
import random
import itertools
import numpy as np

if __name__ == "__main__":

    bell_nums = range(1,6)
    pairs = list(itertools.combinations(bell_nums,2))

    for pair in pairs:

        bell_num1=int(pair[0])
        bell_num2=int(pair[1])
        train_data = np.empty((0,12),float)
        train_label = np.array([])
        test_data = np.empty((0,12),float)
        test_label = np.array([])
        noise_nums = range(1,2)
        level_nums = range(0,9)
        random.shuffle(noise_nums)

        nfft = 2048  # FFTのサンプル数
        nceps = 12   # MFCCの次元数

        #鈴の音1
        for noise_num in noise_nums[0:10]:
            random.shuffle(level_nums)
            #学習用データを作成
            for level_num in level_nums[0:10]:
                files_name = glob.glob("learning_sample/%d_%d_%d.wav" % (bell_num1,noise_num,level_num))
                for file_name in files_name:
                    feature = get_feature(file_name,nfft,nceps)
                    if len(train_data) == 0:
                        train_data=feature
                    else:
                        train_data=np.vstack((train_data,feature))
                    train_label=np.append(train_label,bell_num1)
            #テストデータを作成
            file_name = "learning_sample/%d_%d_%d.wav" % (bell_num1,noise_num,level_nums[8])
            feature = get_feature(file_name,nfft,nceps)
            if len(test_data) == 0:
                test_data=feature
            else:
                test_data=np.vstack((test_data,feature))
            test_label=np.append(test_label,bell_num1)

        #鈴の音2
        for noise_num in noise_nums[0:10]:
            random.shuffle(level_nums)
            #学習用データを作成
            for level_num in level_nums[0:10]:
                files_name = glob.glob("learning_sample/%d_%d_%d.wav" % (bell_num2,noise_num,level_num))
                for file_name in files_name:
                    feature = get_feature(file_name,nfft,nceps)
                    if len(train_data) == 0:
                        train_data=feature
                    else:
                        train_data=np.vstack((train_data,feature))
                    train_label=np.append(train_label,bell_num2)
            #テストデータを作成
            file_name = "learning_sample/%d_%d_%d.wav" % (bell_num2,noise_num,level_nums[8])
            feature = get_feature(file_name,nfft,nceps)
            if len(test_data) == 0:
                test_data=feature
            else:
                test_data=np.vstack((test_data,feature))
            test_label=np.append(test_label,bell_num2)


        #特徴データをテキストに出力
        feature_train_data=np.hstack((train_label.reshape(len(train_label),1),train_data))
        feature_test_data=np.hstack((test_label.reshape(len(test_label),1),test_data))
        with open("feature_data/train_data.txt","w") as f:
            writer=csv.writer(f)
            writer.writerows(feature_train_data)
        with open("feature_data/test_data.txt","w") as f:
            writer=csv.writer(f)
            writer.writerows(feature_test_data)

        #識別機学習
        clf = svm.SVC()
        clf.fit(train_data,train_label)
        #推定
        test_pred = clf.predict(test_data)
        #print np.hstack((test_label.reshape(len(test_label),1),(test_pred.reshape(len(test_pred),1))))

        #結果算出
        score=accuracy_score(test_label, test_pred)
        print (pair,score)
