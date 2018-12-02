# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 11/11/2018
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import sparse
from sklearn.metrics import mean_squared_error

def cos_similarity(matrix1, matrix2, epsilon=1e-9):
    sim = matrix1.dot(matrix2.T)
    norms1 = np.array([np.sqrt(np.diagonal(matrix1.dot(matrix1.T)+epsilon))]).T
    norms2 = np.array([np.sqrt(np.diagonal(matrix2.dot(matrix2.T)+epsilon))])
    return (sim / norms1/ norms2)

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def predict(ratings, similarity, kind, epsilon=1e-9):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for j in range(ratings.shape[1]):
            index = ratings[:, j].nonzero()[0]
            pred[:, j] = similarity.dot(ratings[:, j]) / (np.abs(similarity[:, index]).sum(axis=1) + epsilon)
    if kind == 'item':
        for i in range(ratings.shape[0]):
            index = ratings[i, :].nonzero()[0]
            pred[i, :] = ratings[i, :].dot(similarity) / (np.abs(similarity[index, :]).sum(axis=0) + epsilon)
    return pred

def main():
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('../ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    M = np.max(ratings["UserID"])
    N = np.max(ratings["MovieID"])
    random_state = [1,100,666,1111,2222]
    for rs in random_state:
        error_user = []
        error_item = []
        Y_train, Y_test = train_test_split(ratings, test_size=0.2, random_state=rs)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        rating_mat = sparse.coo_matrix((Y_train[:,2], (Y_train[:,0]-1, Y_train[:,1]-1)), shape=(M, N)).toarray()
        test_mat = sparse.coo_matrix((Y_test[:,2], (Y_test[:,0]-1, Y_test[:,1]-1)), shape=(M, N)).toarray()
        sim_user = cos_similarity(rating_mat, rating_mat)
        predict_user = predict(rating_mat, sim_user, kind="user")
        error_user.append(get_mse(predict_user, test_mat))
        sim_item = cos_similarity(rating_mat.T, rating_mat.T)
        predict_item = predict(rating_mat, sim_item, kind = "item")
        error_item.append(get_mse(predict_item, test_mat))
    print("User CF average loss :", np.average(np.array(error_user)))
    print("Item CF average loss :", np.average(np.array(error_item)))
if __name__ == "__main__":
    main()