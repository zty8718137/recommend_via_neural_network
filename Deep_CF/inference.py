# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 11/15/2018
'''

import tensorflow as tf
from Deep_CF.Config import *

def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
    movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets


def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name="uid_embed_matrix")
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1),
                                          name="gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")

        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), name="job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope("user_fc"):
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="uid_fc_layer", activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name="gender_fc_layer", activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name="age_fc_layer", activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name="job_fc_layer", activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))

        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  # (?, 1, 128)
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, K, tf.tanh)  # (?, 1, 200)

        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, K])
    return user_combine_layer, user_combine_layer_flat

def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1), name = "movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name = "movie_id_embed_layer")
    return movie_id_embed_layer

def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1), name = "movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories, name = "movie_categories_embed_layer")
        movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
    return movie_categories_embed_layer


def get_movie_cnn_layer(movie_titles):
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1),
                                               name="movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name="movie_title_embed_layer")
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                         name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                      name="conv_layer")
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1],
                                           padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # Dropout
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
        max_num = len(window_sizes) * filter_num
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")
        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer


def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    with tf.name_scope("movie_fc"):
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name="movie_id_fc_layer",
                                            activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim,
                                                    name="movie_categories_fc_layer", activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))

        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, K, tf.tanh)  # (?, 1, 200)

        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, K])
    return movie_combine_layer, movie_combine_layer_flat

def graph_inf(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles):
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender,
                                                                                               user_age, user_job)
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer,
                                                                         age_embed_layer, job_embed_layer)
    movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
    movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
    pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles)
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                            movie_categories_embed_layer,
                                                                            dropout_layer)
    # Method 1 : Concatenate two features into one long vec
    # with tf.name_scope("inference"):
    #     inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)
    #     inference_1 = tf.layers.dense(inference_layer, fc_layer1_num,
    #                                   activation=tf.nn.relu,
    #                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    #                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate),
    #                                 name="fc_1")
    #     inference_2 = tf.layers.dense(inference_1, fc_layer2_num,
    #                                   activation=tf.nn.relu,
    #                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    #                                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate),
    #                                name="fc_2")
    #     inference = tf.layers.dense(inference_2, 1,
    #                                 activation=tf.nn.relu,
    #                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    #                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate),
    #                                 name="inference")

    # Method 2 : Implement matrix multiplication
    with tf.name_scope("inference"):
        inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
        inference = tf.expand_dims(inference, axis=1)
    return inference