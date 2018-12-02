# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 11/25/2018
'''
import tensorflow as tf
from Deep_CF.utils import *
import numpy as np
from Deep_CF.Config import save_dir, model_name


def eval(X, y):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(save_dir + model_name + '.meta')
        loader.restore(sess, save_dir + model_name)
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, inference, movie_combine_layer_flat, user_combine_layer_flat, loss = get_tensors(loaded_graph)  # loaded_graph
        test_batch = get_batches(X, y, batch_size)
        test_loss = []
        for bch in range(X.shape[0] // batch_size):
            xs, ys = next(test_batch)
            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = xs.take(6, 1)[i]
            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = xs.take(5, 1)[i]
            feed = {
                uid: np.reshape(xs.take(0, 1), [xs.shape[0], 1]),
                user_gender: np.reshape(xs.take(2, 1), [xs.shape[0], 1]),
                user_age: np.reshape(xs.take(3, 1), [xs.shape[0], 1]),
                user_job: np.reshape(xs.take(4, 1), [xs.shape[0], 1]),
                movie_id: np.reshape(xs.take(1, 1), [xs.shape[0], 1]),
                movie_categories: categories,
                movie_titles: titles,  # x.take(5,1)
                targets: np.reshape(ys, [ys.shape[0], 1]),
            }
            training_loss = sess.run([loss], feed_dict=feed)
            test_loss.append(training_loss)
        print("Loss on test dataset : %g" %(np.mean(test_loss)))
        return np.mean(test_loss)