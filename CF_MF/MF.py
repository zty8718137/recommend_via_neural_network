# -*- coding:utf-8 -*-

'''
   author: Tianyu Zhong
   Created on 11/8/2018
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CF_MF.MF_skeleton import train_model, get_err
from sklearn.model_selection import train_test_split

def main():
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    Y_train, Y_test = train_test_split(ratings, test_size=0.2, random_state=1)

    M = max(np.max(Y_train["UserID"]), np.max(Y_test["UserID"])).astype(int)  # users
    N = max(np.max(Y_train["MovieID"]), np.max(Y_test["MovieID"])).astype(int)  # movies
    Ks = [10, 20, 30, 50, 100]
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    regs = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1]
    eta = 0.03  # learning rate
    E_ins = []
    E_outs = []

    # Use to compute Ein and Eout
    for reg in regs:
        E_ins_for_lambda = []
        E_outs_for_lambda = []

        for k in Ks:
            print("Training model with M = %s, N = %s, k = %s, eta = %s, reg = %s" % (M, N, k, eta, reg))
            U, V, e_in = train_model(M, N, k, eta, reg, Y_train)
            E_ins_for_lambda.append(e_in)
            eout = get_err(U, V, Y_test)
            E_outs_for_lambda.append(eout)

        E_ins.append(E_ins_for_lambda)
        E_outs.append(E_outs_for_lambda)

    # Plot values of E_in across k for each value of lambda
    for i in range(len(regs)):
        plt.plot(Ks, E_ins[i], label='$E_{in}, \lambda=$' + str(regs[i]))
    plt.title('$E_{in}$ vs. K')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('2e_ein.png')
    plt.clf()

    # Plot values of E_out across k for each value of lambda
    for i in range(len(regs)):
        plt.plot(Ks, E_outs[i], label='$E_{out}, \lambda=$' + str(regs[i]))
    plt.title('$E_{out}$ vs. K')
    plt.xlabel('K')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('2e_eout.png')


if __name__ == "__main__":
    #main()
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('../ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')
    Y_train, Y_test = train_test_split(ratings, test_size=0.2, random_state=1)
    M = max(np.max(Y_train["UserID"]), np.max(Y_test["UserID"])).astype(int)  # users
    N = max(np.max(Y_train["MovieID"]), np.max(Y_test["MovieID"])).astype(int)  # movies
    U, V, e_in = train_model(M, N, 40, 0.03, 0.1, Y_train.values)
    print(get_err(U, V, Y_test.values))
