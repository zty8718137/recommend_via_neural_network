# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 11/26/2018
'''
# Config for inference
embed_dim = 32
uid_max = 6040
gender_max = 2
age_max = 7
job_max = 21
movie_id_max = 3952
movie_categories_max = 19
movie_title_max = 5216
sentences_size = 15
window_sizes = {2, 3, 4, 5}
filter_num = 8
dropout_keep_prob = 0.5
reg_rate = 0.00001
K = 256
user_item_concat = "mf"

#Config for training
num_epochs = 10
batch_size = 256
learning_rate = 0.001
show_every_n_batches = 20
fc_layer1_num = 128
#fc_layer2_num = 16
save_dir = "../save/model/"
model_name = "model.ckpt"
data_dir = "../save/matrix/"
top_k = 20