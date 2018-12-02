# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 11/15/2018
'''
import tensorflow as tf
from Deep_CF.inference import *
from Deep_CF.utils import *
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
from Deep_CF.Config import *

def train(X, y):
    train_graph = tf.Graph()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=1)
    with train_graph.as_default():
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets = get_inputs()
        inference = graph_inf(uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles)
        cost = tf.losses.mean_squared_error(targets, inference)
        loss_no_reg = tf.reduce_mean(cost, name="loss")
        loss_reg = loss_no_reg + tf.losses.get_regularization_loss()
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_reg, global_step=global_step)
        saver = tf.train.Saver()
    prev_loss = float("inf")
    with tf.Session(graph=train_graph) as sess:
        tf.global_variables_initializer().run()
        for epoch_i in range(num_epochs):
            train_batch = get_batches(train_X, train_y, batch_size)
            for bch in range(len(train_X)//batch_size):
                xs, ys = next(train_batch)
                categories = np.zeros([batch_size, 18])
                for i in range(batch_size):
                    categories[i] = xs.take(6, 1)[i]
                titles = np.zeros([batch_size, sentences_size])
                for i in range(batch_size):
                    titles[i] = xs.take(5, 1)[i]

                feed = {
                    uid: np.reshape(xs.take(0, 1), [batch_size, 1]),
                    user_gender: np.reshape(xs.take(2, 1), [batch_size, 1]),
                    user_age: np.reshape(xs.take(3, 1), [batch_size, 1]),
                    user_job: np.reshape(xs.take(4, 1), [batch_size, 1]),
                    movie_id: np.reshape(xs.take(1, 1), [batch_size, 1]),
                    movie_categories: categories,  # x.take(6,1)
                    movie_titles: titles,  # x.take(5,1)
                    targets: np.reshape(ys, [batch_size, 1]),
                }
                _, step, training_loss = sess.run([train_op, global_step, loss_reg], feed_dict=feed)
                if (epoch_i * (len(train_X) // batch_size) + bch) % show_every_n_batches == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        time_str,
                        epoch_i,
                        bch,
                        (len(train_X) // batch_size),
                        training_loss))

            # Get test loss for each epoch
            test_batch = get_batches(test_X, test_y, batch_size)
            for bch_test in range(test_X.shape[0]//batch_size):
                test_loss = []
                x_test, y_test = next(test_batch)
                categories_test = np.zeros([x_test.shape[0], 18])
                for i in range(x_test.shape[0]):
                    categories_test[i] = x_test.take(6, 1)[i]
                titles_test = np.zeros([x_test.shape[0], sentences_size])
                for i in range(x_test.shape[0]):
                    titles_test[i] = x_test.take(5, 1)[i]

                feed_test = {
                    uid: np.reshape(x_test.take(0, 1), [x_test.shape[0], 1]),
                    user_gender: np.reshape(x_test.take(2, 1), [x_test.shape[0], 1]),
                    user_age: np.reshape(x_test.take(3, 1), [x_test.shape[0], 1]),
                    user_job: np.reshape(x_test.take(4, 1), [x_test.shape[0], 1]),
                    movie_id: np.reshape(x_test.take(1, 1), [x_test.shape[0], 1]),
                    movie_categories: categories_test,  # x.take(6,1)
                    movie_titles: titles_test,  # x.take(5,1)
                    targets: np.reshape(y_test, [x_test.shape[0], 1]),
                }
                _, step, training_loss = sess.run([train_op, global_step, loss_no_reg], feed_dict=feed_test)
                test_loss.append(training_loss)
            print("At epoch %d, validation loss is %g" % (epoch_i, np.mean(test_loss)))
            if prev_loss > np.mean(test_loss) or epoch_i <= 5:
                prev_loss = np.mean(test_loss)
                saver.save(sess, save_dir + model_name)
            else : break
        print('Model Trained and Saved')

