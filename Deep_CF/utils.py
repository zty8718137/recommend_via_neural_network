# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 11/15/2018
'''
import tensorflow as tf
import numpy as np
import pickle
from Deep_CF.Config import *
import pandas as pd
import json

def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]

def get_tensors(loaded_graph):
    uid = loaded_graph.get_tensor_by_name("uid:0")
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")
    movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
    movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
    movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    if user_item_concat == "concat":
        # Method 1
        inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
    elif user_item_concat == "mf":
        # Method 2
        inference = loaded_graph.get_tensor_by_name("inference/ExpandDims:0")

    movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
    loss = loaded_graph.get_tensor_by_name("loss:0")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, inference, movie_combine_layer_flat, user_combine_layer_flat, loss

def rating_movie(userid, movieid, users, movies):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(save_dir + model_name + '.meta')
        loader.restore(sess, save_dir + model_name)
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, inference, movie_combine_layer_flat, user_combine_layer_flat = get_tensors(loaded_graph)  # loaded_graph
        user_info = users[users["UserID"] == userid].values
        movie_info = movies[movies["MovieID"] == movieid].values

        feed = {
            uid: np.reshape(userid, [1, 1]),
            user_gender: np.reshape(user_info[1], [1, 1]),
            user_age: np.reshape(user_info[2], [1, 1]),
            user_job: np.reshape(user_info[3], [1, 1]),
            movie_id: np.reshape(movieid, [1, 1]),
            movie_categories: movie_info[2],  # x.take(6,1)
            movie_titles: movie_info[1],  # x.take(5,1)
        }
        inf = sess.run([inference], feed)
        return inf

def save_movie_feature(movies):
    loaded_graph = tf.Graph()
    movie_matrix = np.zeros([movie_id_max, K])
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(save_dir + model_name + '.meta')
        loader.restore(sess, save_dir + model_name)

        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, inference, movie_combine_layer_flat, user_combine_layer_flat, loss = get_tensors(loaded_graph)  # loaded_graph
        for item in movies.values:
            feed = {
                movie_id: np.reshape(item.take(0), [1, 1]),
                movie_categories: np.reshape(item.take(2),[-1,18]),  # x.take(6,1)
                movie_titles: np.reshape(item.take(1), [-1,15]),  # x.take(5,1)
            }
            movie_vec = sess.run([movie_combine_layer_flat], feed)
            movie_matrix[np.reshape(item.take(0), [1, 1]) - 1] = movie_vec
    pickle.dump((np.array(movie_matrix).reshape(-1, K)), open(data_dir + 'movie_matrix.p', 'wb'))

def save_user_feature(users):
    loaded_graph = tf.Graph()
    user_matrix = np.zeros([uid_max, K])
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(save_dir + model_name + '.meta')
        loader.restore(sess, save_dir + model_name)

        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, inference, movie_combine_layer_flat, user_combine_layer_flat, loss = get_tensors(loaded_graph)  # loaded_graph

        for item in users.values:
            feed = {
                uid: np.reshape(item.take(0), [1, 1]),
                user_gender: np.reshape(item.take(1), [1, 1]),
                user_age: np.reshape(item.take(2), [1, 1]),
                user_job: np.reshape(item.take(3), [1, 1]),
            }
            user_vec = sess.run([user_combine_layer_flat], feed)
            user_matrix[np.reshape(item.take(0), [1, 1]) - 1] = user_vec
    pickle.dump((np.array(user_matrix).reshape(-1, K)), open(data_dir + 'user_matrix.p', 'wb'))

def save_rating_history():
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('../ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    history = {}
    for line in np.array(ratings):
        uid = line[0].astype(str)
        itemid = int(line[1])
        if uid not in history.keys():
            history[uid] = []
        history[uid].append(itemid)
    with open(data_dir + "history.json", "w+") as file:
        file.write(json.dumps(history) + "\n")

