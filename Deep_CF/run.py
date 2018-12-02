# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 11/25/2018
'''
import tensorflow as tf
from Deep_CF.utils import save_movie_feature, save_user_feature
from Deep_CF.preprocess import load_data
from Deep_CF.train import train
from Deep_CF.eval import eval
from sklearn.model_selection import KFold
from Deep_CF.Config import *
import numpy as np

def main(argv = None):
    title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    # train_X, test_X, train_y, test_y = train_test_split(features, targets_values, test_size=0.2, random_state=1)
    # reg_loss = {}
    # for reg_rate in reg_list:
    test_loss = []
    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets_values[train_index], targets_values[test_index]
        train(X_train, y_train)
        test_loss.append(eval(X_test, y_test))
    print("At K: %d, reg: %g, batch_size: %d, embedding_size: %d, layer 1 : %d, layer 2: %d, average test loss: %g" % (K, reg_rate,
                                                                                            batch_size,
                                                                                            embed_dim,
                                                                                            fc_layer1_num,
                                                                                            fc_layer2_num,
                                                                                            np.mean(test_loss)))
    #save_user_feature(users)
    #save_movie_feature(movies)

if __name__ == "__main__":
    tf.app.run()