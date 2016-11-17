from __future__ import division
import matplotlib.pyplot as plt
import random, time
import math
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from numpy import linalg
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_moons
from sklearn.datasets.samples_generator import make_circles
from sklearn.neighbors import KNeighborsClassifier
from itertools import cycle
import urllib2
import json
import argparse
import logging
import threading
from random import randint
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import SVC
from sslimport.frameworks.SelfLearning import SelfLearningModel
from sklearn.svm import SVC
from sslimport.methods import scikitTSVM
from sklearn.cross_validation import train_test_split
from time import time

# this parameter instructs RPC to stop fitting if the difference between
# the new iteration error rate and the old one is this big
_EXIT_ERROR_THRESHOLD = 0.5
# the number of points in the region of uncertainty when
# the model stops iterating
_MIN_BETA_LENGTH = 5

# API is not 24 hours available, this data is kinda fake, but inspired by real;
# we handle it as coming in real-time which makes this experiment quite close to
# the real setup
_FAKE_TEMPERATURE = [20 for _ in range(15)] + \
                    [21 for _ in range(10)] + \
                    [22 for _ in range(10)] + \
                    [23, 24, 25, 26, 27] + \
                    [27 for _ in range(10)] + \
                    [26, 25, 24, 24, 23] + \
                    [22 for _ in range(10)]

_T1_TEMPERATURE = [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
_T1_TEMPERATURE_LABELS = [0,0,0,1,1,1,2,2,2,2,3,3,3,4,4,4]
_T1 = {'co2':[0.01,0.05,0.3, 1, 2, 3, 4, 5, 6, 7], 'temp':[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], 'humidity':[5,10,15,20,25,30,40,50,60,70,80,90,100]}
_T1_LABELS = {'co2':[2,2,2,3,3,3,4,4,4,4], 'temp':[0,0,0,1,1,1,2,2,2,2,3,3,3,4,4,4], 'humidity':[0,0,1,1,2,2,2,3,3,3,4,4,4]}

# parmeters names for real data
_PARAMS = ['co2', 'temp', 'humidity']



class RPC_Classifier:
    """
    main class implementing RPC classifier
    """

    def __init__(self, T1=np.array(()), T1Labels=np.array(()), T2=np.array(()), proto_init_type='dataset', loglevel='info', lmbd=1000, iter_num = 10):
        self.T1 = T1
        self.T2 = T2
        self.T1Labels = T1Labels
        self.T2Labels = np.zeros(len(T2))
        self.log = self.set_logging(loglevel)
        self.proto_init_type = proto_init_type
        self.alpha, self.W_LABELS = self.initialize_prototypes(self.proto_init_type)
        self.D = []
        self.D2 = []
        self.fig = 0
        self.Beta = []
        self.iter_count = 0
        self.lmbd = lmbd
        self.iter_num = iter_num

    def set_logging(self, loglevel):
        """
        set RPC logger
        """
        logging.basicConfig(format="%(levelname)s:%(message)s")
        numeric_level = getattr(logging, loglevel.upper(), None)
        logger = logging.getLogger('RPC_Logger')
        logger.setLevel(numeric_level)
        return logger

    def get_pairwise_euc(self):
        """
        pairwise distance matrix calculation function
        """
        D = squareform(pdist(self.T1, metric='euclidean'))
        return D

    def get_pairwise_euc_new(self):
        """
        helper function to calculate pairwise distances matrix for new data
        """
        lengd = len(self.T1)
        lengn = len(self.T2)
        D=np.zeros(shape=(lengn, lengd))
        for j in range(lengn):
            for i in range(lengd):
                euc_sum = 0
                for k in range(len(self.T1[i])):
                    #euclidean sum
                    euc_sum += (self.T2[j][k] - self.T1[i][k])**2
                D[j][i] = math.sqrt(euc_sum)
        return D

    def Fun(self, arg):
        """
        helper function calculating F for the model
        """
        return (1+math.exp(-arg))**(-1)

    def Fun_d(self, arg):
        """
        helper function calculating derivative F' for the model
        """
        if np.abs(arg)>100:
            # this tweaking is made to deal with exp() when arg > 100
            # otherwise it leads to infinity
            arg = 100
        return math.exp(-arg)/(math.exp(-arg)+1)**2

    def alpha_prot_plus(self, i):
        """
        helper function to calculate distance to the closest (+) prototype
        """
        a_plus = np.zeros(len(self.D))
        dist_temp = []
        plus_indexes = []
        for j in range (0, len(self.alpha)):
            if(self.T1Labels[i] == self.W_LABELS[j]):
                dist_temp.append(self.distance_toprot(self.alpha[j], i))
                plus_indexes.append(j)
        sorted_distances_toprot = np.array(dist_temp).argsort(axis=0)
        ss = plus_indexes[sorted_distances_toprot.argmin()]
        a_plus = self.alpha[ss]
        return a_plus, ss

    def alpha_prot_minus(self, i):
        """
        helper function to calculate distance to the closest (-) prototype
        """
        a_minus = np.zeros(len(self.D))
        dist_temp = []
        minus_indexes = []
        for j in range (0, len(self.alpha)):
            if(self.T1Labels[i] != self.W_LABELS[j]):
                dist_temp.append(self.distance_toprot(self.alpha[j], i))
                minus_indexes.append(j)
        sorted_distances_toprot = np.array(dist_temp).argsort(axis=0)
        ss = minus_indexes[sorted_distances_toprot.argmin()]
        a_minus = self.alpha[ss]
        return a_minus, ss

    def alpha_prot_plus_new(self, DATA_LABEL, i, Dx):
        """
        helper function to calculate distance to the closest (+) prototype for new data
        """
        a_plus = np.zeros(len(self.D))
        dist_temp = []
        plus_indexes = []
        for j in range (0,len(self.alpha)):
            if(DATA_LABEL == self.W_LABELS[j]):
                dist_temp.append(self.distance_toprot_new(self.alpha[j], Dx))
                plus_indexes.append(j)
        sorted_distances_toprot = np.array(dist_temp).argsort(axis=0)
        ss = plus_indexes[sorted_distances_toprot.argmin()]
        a_plus = self.alpha[ss]
        return a_plus, ss


    def alpha_prot_minus_new(self, DATA_LABEL, i, Dx):
        """
        helper function to calculate distance to the closest (-) prototype for new data
        """
        a_minus = np.zeros(len(self.D))
        dist_temp = []
        minus_indexes = []
        for j in range (0, len(self.alpha)):
            if(DATA_LABEL != self.W_LABELS[j]):
                dist_temp.append(self.distance_toprot_new(self.alpha[j], Dx))
                minus_indexes.append(j)
        sorted_distances_toprot = np.array(dist_temp).argsort(axis=0)
        ss = minus_indexes[sorted_distances_toprot.argmin()]
        a_minus = self.alpha[ss]
        return a_minus, ss


    def distance_toprot(self, alpha_row, i):
        """
        function calculating distance to prototype
        """
        return np.dot(self.D, alpha_row)[i] - np.dot(np.dot(0.5*alpha_row.T, self.D), alpha_row)


    def distance_toprot_new(self, alpha_row, Dx):
        """
        function calculating distance to prototype on unseen data dissimilarity matrix
        """
        return np.dot(Dx.T, alpha_row) - np.dot(np.dot(0.5*alpha_row.T, self.D), alpha_row)


    def fit_RPC(self):
        """
        RPC fitting function
        """
        self.log.debug("Training RPC")
        E = 0
        E_best = 100000
        E_old = 100000
        rec_num = len(self.D)
        errs = []
        for iter in range(self.iter_num):
            E = 0
            alpha_old = self.alpha.copy()
            for i in range(rec_num):
                alpha_plus, plus_index = self.alpha_prot_plus(i)
                alpha_minus, minus_index = self.alpha_prot_minus(i)
                dp_aplus = self.distance_toprot(alpha_plus, i)
                dp_aminus = self.distance_toprot(alpha_minus, i)
                mu_v = (dp_aplus - dp_aminus)/(dp_aplus + dp_aminus)
                mu_v_plus = 2 * dp_aminus/(dp_aplus + dp_aminus)**2
                mu_v_minus = 2 * dp_aplus/(dp_aplus + dp_aminus)**2
                d_alpha_plus = np.zeros(rec_num)
                d_alpha_minus = np.zeros(rec_num)
                mu_v_Fun_d = self.Fun_d(mu_v)
                for kk in range(rec_num):
                    d_alpha_plus[kk] = -mu_v_Fun_d * mu_v_plus * (self.D[i][kk] - (sum([self.D[l][kk]*alpha_plus[l] for l in range(rec_num)])))
                    d_alpha_minus[kk]= mu_v_Fun_d * mu_v_minus * (self.D[i][kk] - (sum([self.D[l][kk]*alpha_minus[l] for l in range(rec_num)])))
                alpha_plus += d_alpha_plus/self.lmbd
                alpha_minus += d_alpha_minus/self.lmbd

                self.alpha[plus_index] = alpha_plus.copy()
                self.alpha[minus_index] = alpha_minus.copy()

            for kk in range(len(self.alpha)):
                self.alpha[kk] = self.alpha[kk].copy()/sum(self.alpha[kk])

            E = 0
            for ii in range(rec_num):
                alpha_plus, _ = self.alpha_prot_plus(ii)
                alpha_minus, _ = self.alpha_prot_minus(ii)
                dp_aplus = self.distance_toprot(alpha_plus, ii)
                dp_aminus = self.distance_toprot(alpha_minus, ii)
                E += (self.Fun((dp_aplus - dp_aminus)/(dp_aplus + dp_aminus)))
            self.log.debug("E = %s" % E)

            if ((E-E_old) > _EXIT_ERROR_THRESHOLD):
                self.log.debug("Breaking after %s iterations" % iter)
                self.alpha = alpha_best.copy()
                break
            else:
                E_old = E
                if E < E_best:
                    alpha_best = self.alpha.copy()
                    E_best = E
                errs.append(E)
        self.log.debug("RPC training done after total %s iterations" % iter)

        plt.figure(25)
        plt.plot(errs)
        w = []
        plt.figure(self.fig)
        candidate_row = np.array(self.T1)
        colors = cycle('rbm')
        for k, col in zip(range(3), colors):
            my_members = self.T1Labels == k
            plt.plot(candidate_row[my_members, 0], candidate_row[my_members, 1], col + '.')

        colors = ['r','b','m']
        # plot prototype positions
        # each prototype marker is scaled basing on how long ago it was created
        for lab in range(len(np.unique(self.W_LABELS))):
            w = []
            w_count = 0
            for i in range(len(self.alpha)):
                if lab == self.W_LABELS[i]:
                    w.append(np.dot(self.alpha[i], self.T1))
                    #plt.scatter(w[w_count][0], w[w_count][1], c = colors[int(self.W_LABELS[i])], marker = 'o', s = 15*3**w_count, alpha=0.7)
                    w_count += 1

        self.fig += 1


    def conformal_prediction(self):
        """
        conformal prediction for RPC
        """
        self.log.debug("Conformal prediction")
        rl = []
        alpha_plus = []
        alpha_minus = []
        self.Beta = []
        for N in range(len(self.T2)):
            rr = np.zeros(len(np.unique(self.T1Labels)), dtype=float)
            for l in range(len(np.unique(self.T1Labels))):
            #N - index in T2
                alpha_plus, plus_index = self.alpha_prot_plus_new(l, N, self.D2[N])
                alpha_minus, minus_index = self.alpha_prot_minus_new(l, N, self.D2[N])
                n_mi = self.distance_toprot_new(alpha_plus, self.D2[N])/self.distance_toprot_new(alpha_minus, self.D2[N])
                N_len = 0.0
                for i in range(len(self.T1)):
                    #compute mu_i for T1 and remeber it
                    alpha_plus, plus_index = self.alpha_prot_plus(i)
                    alpha_minus, minus_index = self.alpha_prot_minus(i)
                    t1_mi = self.distance_toprot(alpha_plus, i)/self.distance_toprot(alpha_minus, i)
                    if ((t1_mi) >= (n_mi)):
                        N_len += 1.0
                #N_len +=1.0
                rr[l] = float(N_len)/float(len(self.T1)+1)
            rl.append([rr])
            label_index = np.argsort(rr)[len(np.unique(self.T1Labels))-1]
            point_label = np.unique(self.T1Labels)[label_index]
            self.T2Labels[N] = point_label.copy()
            sorted_rr = np.sort(rr)[::-1]
            conf = 1 - sorted_rr[1]
            cred = sorted_rr[0]
            if((conf <= (1 - 1/(len(self.T1)))) or (cred <= 1/(len(self.T1)))) and (self.T2[N].tolist() not in self.T1.tolist()):
                self.Beta.append(self.T2[N])

    def get_new_prototypes_T1_only(self, this_cluster_beta, this_cluster_label):
        """
        method for getting new prototypes
        """
        self.log.debug("Getting new prototypes from T1")
        data_dim = self.T1[0].shape[0]
        alphas_of_betas = []
        beta_median = []

        for i in range(data_dim):
            median_index = 0
            temp_list = []
            element_index = 0
            for j in range(len(this_cluster_beta)):
                temp_list.append(this_cluster_beta[j][i])
            #odd number case
            if ((len(temp_list)%2) == 1):
                index_of_mid_el = (len(temp_list)-1)/2
                sorted_temp_list_indexes = np.argsort(np.array(temp_list).argsort(axis=0))

                for k in range(len(sorted_temp_list_indexes)):
                    if (sorted_temp_list_indexes[k] == index_of_mid_el):
                        element_index = k
            # even number case - get two middle elements
            # return 1 to the closest to the real median
            else:
                index_of_mid_el = len(temp_list)/2 - 1
                sorted_temp_list_indexes = np.argsort(np.array(temp_list).argsort(axis=0))

                for k in range(len(sorted_temp_list_indexes)):
                    if (sorted_temp_list_indexes[k] == int(index_of_mid_el)):
                        element_index_1 = k

                    if (sorted_temp_list_indexes[k] == (int(index_of_mid_el)+1)):
                        element_index_2 = k
                the_goal_median = np.median(temp_list)

                if (np.abs(the_goal_median - temp_list[element_index_1]) <= np.abs(the_goal_median - temp_list[element_index_2])):
                    element_index = element_index_1
                else:
                    element_index = element_index_2

            beta_median.append(temp_list[element_index])
            temp_list = np.zeros(len(temp_list))
            temp_list[element_index] = 1
            alphas_of_betas.append(temp_list)

        label_new_prot = this_cluster_label
        self.T1 = np.append(self.T1, this_cluster_beta, axis=0)

        for i in range(len(this_cluster_beta)):
            self.T1Labels = np.append(self.T1Labels, label_new_prot)

        for i in range(data_dim):
            self.W_LABELS= np.append(self.W_LABELS, label_new_prot)

        bottom_zeros = np.zeros((data_dim, self.alpha.shape[1]))
        right_zeros = np.zeros((self.alpha.shape[0], len(this_cluster_beta)))
        self.alpha = np.concatenate((self.alpha, bottom_zeros))
        right_zeros = np.concatenate((right_zeros, np.array(alphas_of_betas)))
        self.alpha = np.concatenate((self.alpha, right_zeros),axis=1)

    def initialize_prototypes(self, proto_init_type):
        """
        this method enables different techniques for initializing prototypes
        """
        num_clusters = len(np.unique(self.T1Labels))
        rec_num = self.T1.shape[0]
        _alpha = []
        _W_LABELS = []
        self.log.debug("Prototype init type: %s" % proto_init_type)
        if proto_init_type == 'random':
            for counter in range(num_clusters):
                r = [random.random() for i in range(rec_num)]
                s = sum(r)
                r = [ i/s for i in r ]
                _alpha.append(r)
                _W_LABELS.append(counter)

        # NN-initialization of initial prototypes
        if proto_init_type == 'NN':
            flag = True
            while flag == True:
                    no_conv = False
                    n_neighbors = 3
                    clf = KNeighborsClassifier(n_neighbors)
                    clf.fit(self.T1, self.T1Labels)
                    _W_LABELS = []
                    while len(np.unique(_W_LABELS)) != len(np.unique(self.T1Labels)):
                        _W_LABELS = []
                        for counter in range(num_clusters):
                            s = 0
                            r = [random.uniform(-1,1) for i in range(0, rec_num)]
                            s = sum(r)
                            r = [ i/s for i in r ]
                            _alpha.append(r)
                            _W_LABELS.append(clf.predict(np.dot(_alpha[counter], self.T1))[0])
                    for lab in range(0,int(np.max(_W_LABELS))+1):
                        if _W_LABELS.count(lab) == 1:
                           continue
                        else:
                            no_conv = True
                            break
                    if no_conv == False:
                        break

        # here we set initial W to a point in a cluster
        if proto_init_type == 'dataset':
            _W_LABELS = [i for i in range(num_clusters)]
            for k in range(len(_W_LABELS)):
                _alpha.append(np.zeros(len(self.T1)))
                for i in range(len(self.T1Labels)):
                    if self.T1Labels[i] == k:
                        _alpha[k][i] = 1.0
                        break

        self.log.debug("W_LABELS:\n%s" % _W_LABELS)
        self.log.debug(_alpha)
        return np.array(_alpha), np.array(_W_LABELS)


    def nearest_n(self, new_prot):
        """
        helper to find closest point label frrom T1
        """
        min_label = 0
        min = 1000
        for i in range(len(self.T1)):
            dst = distance.euclidean(new_prot, self.T1[i])
            if(dst < min):
                min = dst
                min_label = self.T1Labels[i]
        return min_label


    def model_do_fitting(self):
        """
        Scikit-learn notation for fitting
        """
        self.RPC_iteration()

    def model_do_predicting(self):
        """
        Scikit-learn notation for predicting
        """
        s_index_best = -1
        _alpha_best = []
        _W_LABELS_BEST = []
        _T1_best = []
        _T1_LABELS_BEST = []
        _T2LabelsBest = self.T2Labels.copy()
        while (len(self.Beta)>_MIN_BETA_LENGTH) and (self.iter_count<self.iter_num):
            clusters_list = []
            for i in range(len(self.Beta)):
                clusters_list = np.append(clusters_list, self.nearest_n(self.Beta[i]))
            for ii in np.unique(clusters_list):
                beta_of_this_cluster = []
                for iii in range(len(self.Beta)):
                    if clusters_list[iii] == ii:
                        beta_of_this_cluster.append(self.Beta[iii])
                this_cluster_label = ii
                self.get_new_prototypes_T1_only(beta_of_this_cluster, this_cluster_label)
            self.log.info("%s iteration alpha" % str(self.iter_count))
            self.RPC_iteration()
            s_index = metrics.silhouette_score(np.concatenate((self.T1, self.T2), axis=0), np.concatenate((self.T1Labels,self.T2Labels),axis=0), metric='euclidean')
            self.log.debug("Silhouette index = %s" % s_index)
            self.log.debug("T2 Predicted Labels:\n%s" % self.T2Labels)
            if s_index > s_index_best:
                _T2LabelsBest = self.T2Labels.copy()
                _alpha_best = self.alpha
                _W_LABELS_BEST = self.W_LABELS.copy()
                _T1_best = self.T1
                _T1_LABELS_BEST = self.T1Labels
                self.log.debug("NEW BEST LABELS:\n%s" % _T2LabelsBest)
                s_index_best = s_index

        return _T1_best, _T1_LABELS_BEST, _T2LabelsBest, _alpha_best, _W_LABELS_BEST

    def fit(self, T1, T2, T1Labels):
        """
        scikit-learn notation
        """
        # initialization partially overlaps with constructor to keep
        # compatiblity to both scikit- and non-scikit-learn initializations
        self.T1 = T1
        self.T2 = T2
        self.T1Labels = T1Labels
        self.T2Labels = np.zeros(len(self.T2))
        self.alpha, self.W_LABELS = self.initialize_prototypes(self.proto_init_type)
        self.log.info("Initialized RPC classifier")
        self.log.info("Training RPC classifier")
        self.model_do_fitting()

    def predict(self):
        """
        scikit-learn notation
        """
        _, _, self.T2Labels, _, _ = self.model_do_predicting()
        return self.T2Labels

    def score(self, real_labels, score_type='accuracy'):
        """
        get assessment of the result
        """
        if score_type == 'accuracy':
            num_correct = 0
            for i in range(len(real_labels)):
                if real_labels[i] == self.T2Labels[i]:
                    num_correct += 1

            return num_correct/len(real_labels)
        else:
            return "Unknown accuracy measure"

    def RPC_iteration(self):
        """
        helper function for performing one RPC iteration
        """
        self.D = self.get_pairwise_euc()
        self.D2 = self.get_pairwise_euc_new()
        self.iter_count += 1
        self.fit_RPC()
        self.conformal_prediction()

    def model_do_training(self):
        """
        function defining main model mechanics
        """
        s_index_best = -1
        _alpha_best = []
        _W_LABELS_BEST = []
        _T1_best = []
        _T1_LABELS_BEST = []
        self.RPC_iteration()
        _T2LabelsBest = self.T2Labels.copy()
        while (len(self.Beta)>_MIN_BETA_LENGTH) and (self.iter_count<self.iter_num):
            clusters_list = []
            for i in range(len(self.Beta)):
                clusters_list = np.append(clusters_list, self.nearest_n(self.Beta[i]))
            for ii in np.unique(clusters_list):
                beta_of_this_cluster = []
                for iii in range(len(self.Beta)):
                    if clusters_list[iii] == ii:
                        beta_of_this_cluster.append(self.Beta[iii])
                this_cluster_label = ii
                self.get_new_prototypes_T1_only(beta_of_this_cluster, this_cluster_label)
            self.log.info("%s iteration alpha" % str(self.iter_count))
            self.RPC_iteration()
            s_index = metrics.silhouette_score(np.concatenate((self.T1, self.T2), axis=0), np.concatenate((self.T1Labels,self.T2Labels),axis=0), metric='euclidean')
            self.log.debug("Silhouette index = %s" % s_index)
            self.log.debug("T2 Predicted Labels:\n%s" % self.T2Labels)
            if s_index > s_index_best:
                _T2LabelsBest = self.T2Labels.copy()
                _alpha_best = self.alpha
                _W_LABELS_BEST = self.W_LABELS.copy()
                _T1_best = self.T1
                _T1_LABELS_BEST = self.T1Labels
                self.log.debug("NEW BEST LABELS:\n%s" % _T2LabelsBest)
                s_index_best = s_index

        return _T1_best, _T1_LABELS_BEST, _T2LabelsBest, _alpha_best, _W_LABELS_BEST

    def get_summary(self, T2LabelsBest):
        right_labels = 0
        for i in range(len(self.T2Labels)):
            if int(T2LabelsBest[i]==self.T2Labels[i]):
                right_labels += 1
        accuracy = right_labels/len(self.T2Labels)
        self.log.info("Training Accuracy: %s" % accuracy)
        #plt.show()
        return accuracy

    def cluster_map(self):
        """
        draw a clustermap for the report
        """
        self.log.debug("Plotting Clustermap")
        colors = cycle('rbm')
        k = len(self.T1[0])
        realT1 = self.T1
        realT2 = self.T2
        realT1Labels = self.T1Labels
        realT2Labels = self.T2Labels
        realAlpha = self.alpha
        realW_LABELS = self.W_LABELS
        realD = self.D
        realD2 = self.D2
        if k == 2:
            candidate_x = np.arange(np.min([x[0] for x in self.T1])-1, np.max([x[0] for x in self.T1])+1, 0.1)
            candidate_y = np.arange(np.min([x[1] for x in self.T1])-1, np.max([x[1] for x in self.T1])+1, 0.1)
            plt.figure(20)
            for i in range(len(candidate_x)):
                #candidate_row = []
                for j in range(len(candidate_y)):
                    candidate_row = []
                    candidate_row.append([candidate_x[i], candidate_y[j]])
                    Y_LABELS = np.zeros(len(candidate_row))
                    self.T2 = np.array(candidate_row)
                    self.T2Labels = np.zeros(len(candidate_row))
                    self.D2 = self.get_pairwise_euc_new()
                    self.conformal_prediction()
                    candidate_row = np.array(candidate_row)
                    for k, col in zip(range(3), colors):
                        my_members = self.T2Labels == k
                        plt.scatter(candidate_row[my_members, 0], candidate_row[my_members, 1], marker = '.', color = col, s = 300, alpha = 0.3)

            self.T1 = realT1
            self.T2 = realT2
            self.T1Labels = realT1Labels
            self.T2Labels = realT2Labels
            self.alpha = realAlpha
            self.W_LABELS = realW_LABELS
            self.D = realD
            self.D2 = realD2

            X = self.T1
            for k, col in zip(range(len(np.unique(self.T1Labels))), colors):
                my_members = self.T1Labels == k
                plt.scatter(X[my_members, 0], X[my_members, 1], marker = '.', color = col, s = 150, alpha=0.8)

            X = self.T2
            for k, col in zip(range(len(np.unique(self.T1Labels))), colors):
                my_members = self.T2Labels == k
                plt.scatter(X[my_members, 0], X[my_members, 1], marker = '*', color = col, s = 200, alpha=0.8)

            colors = ['r','b','m']
            # plot prototype positions
            # each prototype marker is scaled basing on how long ago it was created
            for lab in range(len(np.unique(self.W_LABELS))):
                w = []
                w_count = 0
                for i in range(len(self.alpha)):
                    if lab == self.W_LABELS[i]:
                        w.append(np.dot(self.alpha[i], self.T1))
                        plt.scatter(w[w_count][0], w[w_count][1],c = colors[int(self.W_LABELS[i])], marker = 'o', s = 10*10*i, alpha = 0.5)
                        w_count += 1

class BaseAPI:
    def __init__(self, freq, v_len, type='real'):
        self.frequency = freq
        self.vector_length = v_len
        self.data = []
        self.stop_it = False
        self.data_fetched = False
        self.stored_data = []
        self.index_t = 0
        self.params = enumerate(['co2', 'temp', 'humidity'])
        self.prev_point = {'co2':'0.05', 'temp':'23', 'humidity':'20'}
        if type=='fake':
            self.threadSender = threading.Thread(
                name='DataSender',
                target = self.produce_datapoint).start()
        if type=='real':
            self.threadSender = threading.Thread(
                name='DataSender',
                target = self.get_data_from_server).start()


    def produce_datapoint(self):
        while not self.stop_it:
            if self.data_fetched == True:
                self.data = []
                self.data_fetched = False

            time.sleep(self.frequency)
            # check if the latest data was fetched
            # and empty the data

            if self.data_fetched == True:
                self.data = []
                self.data_fetched = False

            if self.stop_it:
                return 0

            # CO2, humidity, temperature
            print "index: %s \n" % self.index_t
            self.stored_data.append([randint(700,800),
                                     random.random(),
                                     _FAKE_TEMPERATURE[self.index_t]])
            self.index_t += 1
            # attempt to send data to Listener
            if len(self.stored_data) == int(self.vector_length):
                self.send_data()

    def get_data_from_server(self):
        from time import gmtime, strftime
        from datetime import datetime, timedelta
        while not self.stop_it:
            if self.data_fetched == True:
                self.data = []
                self.data_fetched = False

            time.sleep(self.frequency)

            if self.data_fetched == True:
                self.data = []
                self.data_fetched = False

            if self.stop_it:
                return 0

            now = strftime("%Y/%m/%d-%H:%M:%S", gmtime())
            prev_time = now - timedelta(seconds=self.frequency)
            response = ''
            response_dict = {'co2':'', 'temp':'', 'humidity':''}
            for param in self.params:
                request = '{API_base_address}?start={time_previous_interval}&end={time_now}&number={room_number}&type={parameter_name}&token=aaltootakaari4'.format(
                API_base_address='http://121.78.237.162:8000/otakaari4',
                time_previous_interval=prev_time,
                room_number='4201',
                parameter_name=param)
                response = urllib2.urlopen(request).read()
                if response:
                    prev_point[param] = response[-1]
                else:
                    response = prev_point[param]

                response_dict[param] = response[-1]

            self.stored_data.append(response_dict)
            self.index_t += 1
            # attempt to send data to Listener
            if len(self.stored_data) == int(self.vector_length):
                self.send_data()

    def send_data(self):
            self.data = self.stored_data
            self.stored_data = []
            # clean data object

    def stop_api(self):
        print "Stopping API thread!"
        self.stop = True
        self.threadSender.join()
        #self.threadSender.stop()

class StoppableThread(threading.Thread):
    """
    Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.
    """

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

def training_RPC_threaded(data_fetched):

    for _, param in enumerate(_PARAMS):
        T1_T, T1_T_LABELS, T2_T = transform_real_data(_T1[param],
                                                      _T1_LABELS[param],
                                                      data_fetched[param])
        T1_T, T2_T = normalize_data(T1_T, T2_T)
        RPC = RPC_Classifier(T1_T, T1_T_LABELS, T2_T, 'dataset', 'debug')
        T1_best, T1_LABELS_BEST, T2LabelsBest, alpha_best, W_LABELS_BEST = RPC.model_do_training()
        RPC.log.debug("Fetched data for %s: %s" % (param, T2_T))
        RPC.log.debug("Final labels: %s" % T2LabelsBest)

class DataListener:
    def __init__(self, call_freq, endpoint):
        self.call_freq = call_freq
        self.endpoint = endpoint
        self.stop_it = False
        self.threadListener = threading.Thread(name='DataListener',
                                               target = self.listen,
                                               args=(self.endpoint,)).start()
        self.exposed_data = []
        self.data_dump = []

    def listen(self, API):
        i = 0
        while not self.stop_it:
            time.sleep(self.call_freq)
            if self.stop_it:
                return 0
            if (len(API.data) > i) and (API.data_fetched == False):
                print "Listener: data arrived"
                for i in range(len(API.data)):
                    self.exposed_data.append(API.data[i])
                print "Exposed data"
                if len(self.exposed_data) > 0:
                    self.data_dump.append(self.exposed_data)
                print self.exposed_data
                API.data_fetched = True
                # get temperature only
                T2_TEMP_fetched = np.array(self.exposed_data)[:,2:]
                threading.Thread(name='Training RPC threaded',
                                 target = training_RPC_threaded,
                                 args=(T2_TEMP_fetched,)).run()
                self.exposed_data = []
                i = 0

            else:
                print "Listener: no data to parse"

    def stop_listener(self):
        self.stop_it = True
        print "Stopping Listening Thread!"
        print self.data_dump
        self.threadListener.join()
        plt.show()
        sys.exit(0)
        #self.threadListener.stop()


def get_args():
    """
    getting various arguments from command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=['toy', 'real', 't1t2experiment'], default='toy')
    parser.add_argument("--toy-type", choices = ['gaussians', 'moons', 'circles'], default='gaussians')
    parser.add_argument("--clustermap", action="store_true")
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--proto-init-type", choices=['random', 'NN', 'dataset'], default='dataset')
    parser.add_argument("--plott1t2", action="store_true")
    parser.add_argument("--loglevel", choices=['debug', 'info', 'warning'], default='info')
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--other-models", action="store_true")
    parser.add_argument("--svm", action="store_true")
    parser.add_argument("--timexperiment", action="store_true")
    parser.add_argument("--timedimexperiment", action="store_true")
    parser.add_argument("--inittimeexperiment", action="store_true")
    parser.add_argument("--othermodelsplot", action="store_true")
    parser.add_argument("--knowndataset", choices=['haberman', 'wdbc', 'ecoli'], default=None)
    parser.add_argument("--toydataset", action="store_true")
    parser.add_argument("--lambdatest", action="store_true")
    parser.add_argument("--lambdatestplot", action="store_true")
    parser.add_argument("--scikit", action="store_true")
    parser.add_argument("--server-data", action="store_true")
    args = parser.parse_args()
    return args


def get_data_from_server(data_type='json'):
    """
    not always possible to use due to unstable API on the Korean side
    """
    print "Getting data from the remote server"
    s_win = []
    if data_type == 'json':
        response = urllib2.urlopen('http://121.78.237.162:8000/otakaari4?start=2014/09/25-05:00:00&end=2014/09/25-09:00:00&number=4201&type=co2&token=aaltootakaari4')
        data_raw = json.load(response)
        data = pd.DataFrame.from_dict(data_raw.values()[1], orient='index')
        data.index = pd.to_datetime(data.index)
        data[0] = data.values.astype('int')
        data = data.sort()
        data = data.asfreq('30S', method='pad')
        for i in range(1, len(data)-1):
            s_win.append((data.values[i-1][0], data.values[i][0], data.values[i+1][0]))
        return np.array(s_win)
    elif data_type == 'raw':
        response = urllib2.urlopen('http://121.78.237.162:8000/otakaari4?start=2014/09/25-05:00:00&end=2014/09/25-09:00:00&number=4201&type=co2&token=aaltootakaari4').read()
        response = response.splitlines()
        for i in range(len(response)-1):
            s_win.append((response[i-1].split(' ')[2], response[i].split(' ')[2], response[i+1].split(' ')[2]))
        return np.array(s_win)


def transform_real_data(T1, LABELS_T1, T2):
    #sliding window
    data_window_t1 = []
    data_window_t2 = []
    labels_t1 = []
    T2 = np.array(T2.ravel())
    for i in range(len(T2)):
        data_window_t2.append((T2[i-1], T2[i], T2[i+1]))
    for i in range(len(T1)):
        data_window_t1.append((T1[i-1], T1[i], T1[i+1]))
        labels_t1.append(LABELS_T1[i])
    return np.array(data_window_t1), \
           np.array(labels_t1),\
           np.array(data_window_t2)


def construct_data(data_type, toy_type="None", n_samples=60, num_clusters=2, t1_div_t2_rate=2, k=2):
    """
    function for constructing synthetic datasets
    """
    if data_type == 'real':
        # first cluster
        # this is a kind of a real data:
        for i in range(10):
            DATA  = np.append(DATA,np.array(np.random.uniform(700, 750)))

        for i in range(10):
            DATA  = np.append(DATA,np.array(np.random.uniform(410, 450)))

        for i in range(7):
            DATA  = np.append(DATA,np.array(np.random.uniform(700, 750)))

        for i in range(10):
            DATA  = np.append(DATA,np.array(np.random.uniform(410, 450)))

        for i in range(10):
            DATA  = np.append(DATA,np.array(np.random.uniform(200, 250)))

        for i in range(10):
            DATA  = np.append(DATA,np.array(np.random.uniform(410, 450)))

        d_win = []
        label_win = []
        for i in range(len(DATA)):
           d_win.append((DATA[i-1], DATA[i], DATA[i+1]))
           # note: labels must start from 0 and contain only consecutive numbers!
           if ((DATA[i-1] and DATA[i]) or (DATA[i] and DATA[i+1]))>600:
               label_win.append(0)
           else:
               label_win.append(1)

        DATA = np.array(d_win)
        DATA_LABELS = np.array(label_win).astype('int')
        DATA2 = get_data_from_server('raw')
        DATA2 = DATA2[-150:-70]
        LABELS = DATA_LABELS.copy()
        DATA3 = np.array(((200, 250, 250), (250, 250, 250),
                          (250, 250, 250), (250, 250, 250),
                          (250, 250, 250), (250, 200, 200)))
        DATA2 = np.append(DATA2, DATA3, axis = 0)
        k = 3
        return DATA1, DATA2, DATA_LABELS,[]

    if data_type == 'toy':
        num_samples_t1 = n_samples
        print num_samples_t1
        num_samples_t2 = n_samples/t1_div_t2_rate
        print num_samples_t2
        print "T2:%s" % t1_div_t2_rate
        data_type = toy_type
        log = logging.getLogger('RPC_Logger')
        log.debug("Toy data initialization")
        log.debug("Num clusters: %s" % num_clusters)
        log.debug("Num samples in T1 %s" % n_samples)
        log.debug("Toy data type: %s" % data_type)
        if data_type == 'gaussians':
            if num_clusters == 2:
                DATA, LABELS = make_blobs(n_samples=int(num_samples_t1), centers=[[2 for x in range(k)],[0 for x in range(k)]], n_features=k)
                DATA2, TARGET_LABELS = make_blobs(n_samples=int(round(num_samples_t2)), centers=[[2 for x in range(k)],[0 for x in range(k)]], n_features=k)

            if num_clusters == 3 and k == 2:
                DATA, LABELS = make_blobs(n_samples=int(num_samples_t1), centers=[[8,8], [5,5], [2,2]], n_features=k)
                DATA2, TARGET_LABELS = make_blobs(n_samples=int(round(num_samples_t2)), centers=[[8,8], [5,5], [2,2]], n_features=k)

        if  data_type == 'moons':
            DATA, LABELS = make_moons(n_samples=200, shuffle=True)
            DATA2, TARGET_LABELS = make_moons(n_samples=30, shuffle=True)
        if  data_type == 'circles':
            DATA, LABELS = make_circles(n_samples=200)
            DATA2, TARGET_LABELS = make_circles(n_samples=30)
        DATA_LABELS = np.array(LABELS)
    return np.array(DATA), np.array(DATA2), np.array(LABELS), TARGET_LABELS


def normalize_data(T1, T2, args=''):
    """
    data scaling helper
    """
    T1T2 = np.concatenate((T1, T2), axis=0)
    if hasattr(args, 'knowndataset'):
        if args.knowndataset== 'ecoli':
            T1T2 = preprocessing.scale(T1T2, with_std=False)
    else:
        T1T2 = preprocessing.scale(T1T2)
    T1 = T1T2[0:len(T1)]
    T2 = T1T2[len(T1):]
    return np.array(T1), np.array(T2)


def t1t2experiment(args):
    """
    T1T2 experiment for the report
    """
    t1_start_samples = 2
    t2_samples = 100
    num_runs = 25
    num_samples_t1_list = [t1_start_samples*i for i in range(1, num_runs+1)]
    num_samples_t2_list = [t2_samples for t in range(1, num_runs+1)]
    t1t2_for_plot = []
    print "Starting T1T2"
    print num_samples_t1_list
    print num_samples_t2_list
    for t in range(len(num_samples_t1_list)):
        t1t2_for_plot.append(float(num_samples_t1_list[t])/float(num_samples_t2_list[t]))
    t1_div_t2_rate_list =  np.array(num_samples_t1_list)/np.array(num_samples_t2_list)
    print("Number of samples in T1:\n%s" % num_samples_t1_list)
    print("Number of samples in T2:\n%s" % num_samples_t2_list)
    print("T1 / T2 rate:\n%s" % t1_div_t2_rate_list)
    num_runs_per_chunk = 2
    k = 2
    accuracy_list = []
    av_acc_other = []
    for tt in range(num_runs):
        print "Iteration %i of the T1T2 experiment" % tt
        chunk_accuracy_list = []
        other_models_chunk_accuracy_list = []
        for i in range(num_runs_per_chunk):
            T1, T2, T1Labels, T2Labels = construct_data("toy", "gaussians", n_samples=num_samples_t1_list[tt], t1_div_t2_rate=t1_div_t2_rate_list[tt])
            T1, T2 = normalize_data(T1, T2, args)
            acc_run = full_iterate(T1, T2, T1Labels, T2Labels, args)
            print "Accuracy for this SSL-RPC run: %s" % acc_run
            chunk_accuracy_list.append(acc_run)
            if args.other_models:
                other_models_chunk_accuracy_list.append(other_models_experiment(T1, T2, T1Labels, T2Labels))
        average_other_models_chunk_accuracy = []
        average_chunk_accuracy = sum(chunk_accuracy_list)/len(chunk_accuracy_list)
        for ii in range(len(other_models_chunk_accuracy_list[0])):
            av_acc = sum(np.array(other_models_chunk_accuracy_list)[:, ii])/len(np.array(other_models_chunk_accuracy_list)[:, ii])
            average_other_models_chunk_accuracy.append(av_acc)
        av_acc_other.append(average_other_models_chunk_accuracy)
        accuracy_list.append(average_chunk_accuracy)
        print "Accuracy for this chunk:\n%s" % chunk_accuracy_list
        print "Accuracy for this chunk, OTHER MODELS:\n%s" % average_other_models_chunk_accuracy
    print "T1T2 experiment results:\n%s" % accuracy_list
    print "T1T2 experiment results, OTHER MODELS:\n%s" % av_acc_other
    sys.exit(0)


def plott1t2():
    accuracies_list = [0.9946666666666667, 0.9971428571428571, 1.0, 1.0, 0.9933333333333334, 0.9888333333333333, 0.9949999999999999, 0.998, 0.9955555555555555, 1.0, 0.9894999999999999, 1.0, 0.984, 1.0, 1.0, 1.0, 1.0, 0.998, 0.9867857142857144, 0.9977777777777778, 0.9925, 0.9960000000000001, 0.9687777777777778, 0.9971428571428571, 0.9883333333333333, 0.9833333333333334, 0.998, 0.9925, 0.9971428571428571, 1.0]
    accuracies_list2 = [0.9844444444444445, 1.0, 0.9685555555555556, 0.9435, 0.9730000000000001, 0.9608571428571429, 0.8992857142857144, 0.9738095238095239, 0.9879999999999999, 0.9507936507936507, 0.9855555555555556, 0.9933333333333334, 0.9355, 0.9597222222222224, 0.9535873015873018, 0.9722857142857144, 0.986111111111111, 0.994, 0.9784761904761906, 0.9949999999999999, 0.9946666666666667, 0.994, 0.944, 0.9702380952380952, 1.0, 0.9460000000000001, 0.9949999999999999, 0.9518888888888888, 0.9364761904761906, 0.9834285714285714]
    t1_start_samples = 50
    num_runs = 30
    num_samples_t1_list = [t1_start_samples for i in range(num_runs)]
    num_samples_t2_list = [int(round(t1_start_samples*i/10)) for i in range(1, num_runs+1)]
    final_list = []
    for i in range(len(num_samples_t1_list)):
        final_list.append(num_samples_t2_list[i]/num_samples_t1_list[i])

    t1t2_for_plot = []
    for t in range(len(num_samples_t1_list)):
        t1t2_for_plot.append(float(num_samples_t2_list[t])/float(num_samples_t1_list[t]))
    t1_div_t2_rate_list = [float(10/i) for i in range(1, num_runs+1)]

    print "Average accuracy 1: " + str(sum(accuracies_list)/len(accuracies_list))
    print "Average accuracy 2: " + str(sum(accuracies_list2)/len(accuracies_list2))
    plt.yticks([t for t in np.arange(0.7, 1.0, 0.05)])
    plt.ylim((0.85, 1.05))
    plt.xlim((-0.1, 3.1))
    plt.xlabel("Propotion T2/T1")
    plt.ylabel("Accuracy rate")
    plt.plot(final_list, accuracies_list, linestyle='--', marker='o', color='b')
    plt.plot(final_list, accuracies_list2, linestyle='--', marker='o', color='r')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def plot_other_models():
    """
    standalone plotting of other models experiment described in text
    """
    plt.figure(10)
    t1_start_samples = 2
    t2_samples = 100
    num_runs = 25
    num_samples_t1_list = [t1_start_samples*i for i in range(1, num_runs+1)]
    num_samples_t2_list = [t2_samples for t in range(1, num_runs+1)]
    t1t2_for_plot = []
    for t in range(len(num_samples_t1_list)):
        t1t2_for_plot.append(float(num_samples_t1_list[t])/float(num_samples_t2_list[t]))


    plt.xlabel("Propotion T2/T1")
    plt.ylabel("Accuracy rate")

    accuracy_list = np.array([1.0, 1.0, 0.94, 1.0, 1.0, 1.0, 0.96, 0.9550000000000001, 1.0, 1.0, 0.895, 0.925, 0.87, 0.975, 1.0, 0.9450000000000001, 0.9450000000000001, 0.96, 1.0, 0.975, 1.0, 1.0, 0.97, 1.0, 1.0])
    av_acc_other = np.array([[0.88500000000000001, 0.87, 0.72500000000000009, 0.56499999999999995, 0.88500000000000001, 0.56499999999999995, 0.89500000000000002, 0.88500000000000001], [0.89000000000000001, 0.88, 0.86499999999999999, 0.88500000000000001, 0.89000000000000001, 0.48499999999999999, 0.81000000000000005, 0.87], [0.91500000000000004, 0.89000000000000001, 0.91000000000000003, 0.91000000000000003, 0.88, 0.52000000000000002, 0.89500000000000002, 0.89500000000000002], [0.89500000000000002, 0.875, 0.91500000000000004, 0.88500000000000001, 0.90500000000000003, 0.46999999999999997, 0.88500000000000001, 0.875], [0.84999999999999998, 0.84499999999999997, 0.84999999999999998, 0.86499999999999999, 0.84499999999999997, 0.185, 0.87, 0.85499999999999998], [0.90500000000000003, 0.90000000000000002, 0.88500000000000001, 0.89500000000000002, 0.90500000000000003, 0.505, 0.92000000000000004, 0.89500000000000002], [0.89500000000000002, 0.90000000000000002, 0.92500000000000004, 0.88500000000000001, 0.87, 0.875, 0.87, 0.875], [0.90000000000000002, 0.88500000000000001, 0.90000000000000002, 0.90000000000000002, 0.91000000000000003, 0.495, 0.71499999999999997, 0.90500000000000003], [0.80499999999999994, 0.82499999999999996, 0.69999999999999996, 0.82000000000000006, 0.85499999999999998, 0.48999999999999999, 0.80499999999999994, 0.84499999999999997], [0.90500000000000003, 0.90500000000000003, 0.92500000000000004, 0.91500000000000004, 0.90999999999999992, 0.48000000000000004, 0.89000000000000001, 0.90500000000000003], [0.79499999999999993, 0.78000000000000003, 0.76000000000000001, 0.79499999999999993, 0.84000000000000008, 0.57500000000000007, 0.73499999999999999, 0.86499999999999999], [0.87, 0.84999999999999998, 0.85999999999999999, 0.85999999999999999, 0.85499999999999998, 0.92000000000000004, 0.82499999999999996, 0.85999999999999999], [0.875, 0.84999999999999998, 0.88, 0.83999999999999997, 0.85000000000000009, 0.125, 0.78499999999999992, 0.88500000000000001], [0.92500000000000004, 0.90500000000000003, 0.89500000000000002, 0.875, 0.92000000000000004, 0.5, 0.8600000000000001, 0.92500000000000004], [0.88500000000000001, 0.85499999999999998, 0.83499999999999996, 0.85999999999999999, 0.85999999999999999, 0.87, 0.69999999999999996, 0.88], [0.89500000000000002, 0.88500000000000001, 0.89000000000000001, 0.91000000000000003, 0.92999999999999994, 0.47999999999999998, 0.82499999999999996, 0.92000000000000004], [0.91000000000000003, 0.91000000000000003, 0.90000000000000002, 0.91000000000000003, 0.91000000000000003, 0.51000000000000001, 0.875, 0.92000000000000004], [0.875, 0.86499999999999999, 0.90000000000000002, 0.87, 0.89500000000000002, 0.5, 0.84000000000000008, 0.89500000000000002], [0.90500000000000003, 0.92999999999999994, 0.90999999999999992, 0.90999999999999992, 0.91999999999999993, 0.55000000000000004, 0.91500000000000004, 0.91999999999999993], [0.91000000000000003, 0.875, 0.90500000000000003, 0.89000000000000001, 0.91999999999999993, 0.46000000000000002, 0.81999999999999995, 0.91500000000000004], [0.92999999999999994, 0.90999999999999992, 0.93500000000000005, 0.91999999999999993, 0.91999999999999993, 0.47500000000000003, 0.83499999999999996, 0.92500000000000004], [0.86499999999999999, 0.89000000000000001, 0.875, 0.89000000000000001, 0.88, 0.12, 0.87, 0.88500000000000001], [0.89000000000000001, 0.88500000000000001, 0.86499999999999999, 0.88, 0.89500000000000002, 0.115, 0.87, 0.875], [0.90000000000000002, 0.92000000000000004, 0.90000000000000002, 0.90500000000000003, 0.91000000000000003, 0.52500000000000002, 0.90500000000000003, 0.90500000000000003], [0.92500000000000004, 0.90500000000000003, 0.91500000000000004, 0.89500000000000002, 0.91000000000000003, 0.48500000000000004, 0.875, 0.91500000000000004]])

    all_acc = np.c_[accuracy_list,av_acc_other].T
    all_acc_ticks = range(25)
    plt.yticks([t for t in np.arange(0.4, 1.05, 0.05)])
    plt.xticks(np.array(t1t2_for_plot))
    plt.ylim((0.65, 1.02))
    plt.xlim((0, 0.52))

    plt.plot(t1t2_for_plot, accuracy_list, linestyle='-', marker='o', color='b', label="SSL-RPC")
    plt.plot(t1t2_for_plot, np.array(av_acc_other)[:,0], linestyle='-.', marker='v', color='r',label="LP-rbf")
    plt.plot(t1t2_for_plot, np.array(av_acc_other)[:,1], linestyle='--', marker='s', color='g',label="LS-rbf")
    plt.plot(t1t2_for_plot, np.array(av_acc_other)[:,2], linestyle='--', marker='*', color='k',label="LP-KNN-7")
    plt.plot(t1t2_for_plot, np.array(av_acc_other)[:,3], linestyle='-.', marker='d', color='#41BF78',label="LS-KNN-7")
    plt.plot(t1t2_for_plot, np.array(av_acc_other)[:,4], linestyle='--', marker='v', color='#ff00ff',label="SVM-rbf")
    plt.plot(t1t2_for_plot, np.array(av_acc_other)[:,5], linestyle='-.', marker='p', color='#00ff00',label="KMEANS")
    plt.plot(t1t2_for_plot, np.array(av_acc_other)[:,6], linestyle='--', marker='D', color='#ff00ff',label="TSVM")
    plt.plot(t1t2_for_plot, np.array(av_acc_other)[:,7], linestyle='-.', marker='o', color='#ff6600',label="Self-Learning")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.show()

    all_list = []
    for t, i in enumerate([0, 4, 9, 14, 19, 24]):
        print num_samples_t1_list[i]
        pair_list = []
        pair_list.extend([accuracy_list[i]])
        for j in range(len(av_acc_other[0])):
            pair_list.extend([av_acc_other[i][j]])

        all_list.append(pair_list)

    all_list = np.array(all_list)
    for i in range(all_list.shape[1]):
        one_string = ''
        for j in range(all_list[:,i].shape[0]):
           one_string = one_string + "& %s " % all_list[j,i]
        print one_string


def calculate_accuracy(real_labels, predicted_labes):
    num_correct = 0
    for i in range(len(real_labels)):
        if real_labels[i] == predicted_labes[i]:
            num_correct += 1

    return num_correct/len(real_labels)


def calculateKMeans(T1, T2, T1Labels, T2Labels):
    y_pred = KMeans(n_clusters=len(np.unique(T1Labels))).fit_predict(T1)
    print y_pred
    return calculate_accuracy(T1Labels, y_pred)


def other_models_experiment(T1, T2, T1Labels, T2Labels):
    """
    experiment for calculating accuracies for all other models
    in report
    """
    label_prop_model = LabelPropagation()
    label_spr_model = LabelSpreading()
    data = np.append(T1, T2, axis=0)
    data = np.reshape(data,(data.shape[0], data.shape[1]))
    minus_ones = np.ones(len(T2)) * (-1)
    labels = np.append(T1Labels, minus_ones)
    labelsT2_lp = label_prop_model.fit(data, labels).predict(T2)
    labelsT2_ls = label_spr_model.fit(data, labels).predict(T2)
    lp_acc = calculate_accuracy(labelsT2_lp, T2Labels)
    ls_acc = calculate_accuracy(labelsT2_ls, T2Labels)
    i = 0
    lp_acc_knn = np.zeros(1)
    ls_acc_knn = np.zeros(1)
    for kk in enumerate([7]):
        label_prop_model_knn = LabelPropagation(kernel='knn', n_neighbors=kk[1])
        label_spr_model_knn = LabelSpreading(kernel='knn', n_neighbors=kk[1])
        labelsT2_lp = label_prop_model_knn.fit(data, labels).predict(T2)
        labelsT2_ls = label_spr_model_knn.fit(data, labels).predict(T2)
        lp_acc_knn[i] = calculate_accuracy(labelsT2_lp, T2Labels)
        ls_acc_knn[i] = calculate_accuracy(labelsT2_ls, T2Labels)
        i += 1
    # SVM
    clf = SVC()
    clf.set_params(kernel='rbf').fit(T1, T1Labels)
    svm_predicted = clf.predict(T2)
    svm_accuracy = calculate_accuracy(T2Labels, svm_predicted)
    # KMEANS
    kmeans_pred = KMeans(n_clusters=len(np.unique(T1Labels)), n_init=50).fit(T1).predict(T2)
    kmeans_acc = calculate_accuracy(T2Labels, kmeans_pred)

    model = scikitTSVM.SKTSVM(kernel='rbf')
    model.fit(np.concatenate((T1, T2), axis=0), np.concatenate((T1Labels, np.array([-1 for i in range(len(T2Labels))]))).astype(int))
    tsvm_accuracy = model.score(T2, T2Labels)
    print "TSVM: %s" % tsvm_accuracy
    basemodel = SVC(probability=True, kernel='linear')
    ssmodel = SelfLearningModel(basemodel)
    ssmodel.fit(np.concatenate((T1, T2), axis=0), np.concatenate((T1Labels, np.array([-1 for i in range(len(T2Labels))]))).astype(int))
    self_l_accuracy = ssmodel.score(T2, T2Labels)
    print "Self-Learning: %s" % self_l_accuracy
    return [lp_acc, ls_acc, lp_acc_knn[0], ls_acc_knn[0], svm_accuracy, kmeans_acc, tsvm_accuracy, self_l_accuracy]


def time_experiment(args):
    """
    experiment with running time
    """
    times = []
    t1_t2 = [2, 1, 0.66, 0.5, 0.4, 0.33, 0.25, 0.2, 0.1667, 0.1, 0.06667]
    for i, proportion in enumerate(t1_t2):
        T1, T2, T1Labels, T2Labels = construct_data('toy', 'gaussians', 50, 2, t1_div_t2_rate=proportion)
        T1, T2 = normalize_data(T1, T2)
        RPC = RPC_Classifier(T1, T1Labels, T2, args.proto_init_type, args.loglevel)
        t0 = time()
        _, _, _, _, _ = RPC.model_do_training()
        t1 = time()
        t_diff = t1-t0
        times.append(t_diff)
        print "Time Experiment Iteration [%s] completed" % i

    plt.xlabel("T2 length")
    plt.ylabel("Run time, sec")
    plt.figure(99)
    plt.plot(50/np.array(t1_t2), times, linestyle='--', marker='o', color='g')
    print "Time Experiment completed"
    return times


def dimension_time_experiment(args):
    times = []
    data_dim = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    for i, data_dimesnionality in enumerate(data_dim):
        times_dim = []
        for j in range(10):
            T1, T2, T1Labels, T2Labels = construct_data('toy', 'gaussians', 10, 2, t1_div_t2_rate=0.2, k = int(data_dimesnionality))
            T1, T2 = normalize_data(T1, T2)
            RPC = RPC_Classifier(T1, T1Labels, T2, args.proto_init_type, args.loglevel)
            t0 = time()
            _, _, _, _, _ = RPC.model_do_training()
            t1 = time()
            t_diff = t1 - t0
            times_dim.append(t_diff)
        times.append(sum(times_dim)/10)
        print "Time Experiment Iteration [%s] completed, average time was %s times" % (i,sum(times_dim)/10)

    plt.xlabel("Data dimensionality")
    plt.ylabel("Run time, sec")
    plt.xticks([2,3,4,5,6,7,8,9,10,15,20,30])
    plt.figure(98)
    plt.plot(data_dim, times, linestyle='--', marker='o', color='b')
    print "Time Experiment completed"
    return times

def init_time_experiment(args):
    """
    experiment calculating prototype initialization time
    depending on init technique
    """
    number_of_runs = 100
    times = []
    init_type = ['random', 'dataset']
    for i, d_type in enumerate(init_type):
        times_dim = []
        print "INIT Type: %s " % d_type
        for j in range(number_of_runs):
            T1, T2, T1Labels, T2Labels = construct_data('toy', 'gaussians', 10, 2, t1_div_t2_rate=0.2)
            T1, T2 = normalize_data(T1, T2)
            RPC = RPC_Classifier(T1, T1Labels, T2, d_type, args.loglevel)
            t0 = time()
            _, _, _, _, _ = RPC.model_do_training()
            t1 = time()
            t_diff = t1-t0
            times_dim.append(t_diff)
        times.append(sum(times_dim)/number_of_runs)
        print "INIT Time Experiment Iteration [%s] completed, average time was %s" % (i,sum(times_dim)/number_of_runs)
    return times


def load_haberman_data(datafile, random_state, skipped_states):
    """
    load UCI Haberman data
    """
    random_state += skipped_states
    data = np.loadtxt(datafile, delimiter=',')
    all_labels = data[:,3]
    all_points = data[:,:3]
    done = False
    while not done:
        T2, T1, Y2, Y1 = train_test_split(all_points, all_labels, train_size=100, test_size=10, random_state=random_state)
        if len(np.unique(Y2)) == len(np.unique(Y1)):
            done = True
        else:
            random_state += 1
            skipped_states += 1

    Y1 = Y1-1
    Y2 = Y2-1
    return T1, T2, Y1, Y2, skipped_states

def load_ecoli_data(datafile, random_state, skipped_states):
    """
    load UCI Ecoli dataset
    """
    random_state += skipped_states
    all_points = np.genfromtxt(datafile, delimiter='  ', usecols=(range(1,8)), autostrip=True)
    all_labels = np.genfromtxt(datafile, delimiter='  ', usecols=(-1), autostrip=True, dtype='|S5')
    all_labels_modified = all_labels
    for i, label in enumerate(np.unique(all_labels)):
        elements = np.where(all_labels == label)
        for k in range(len(elements)):
            all_labels_modified[elements[k]] = i

    all_labels = np.copy(all_labels_modified).astype(float)
    done = False
    while not done:
        print "random state: %s" % random_state
        T2, T1, Y2, Y1 = train_test_split(all_points, all_labels, train_size=100, test_size = 10, random_state=random_state)
        if (len(np.unique(Y2)) == len(np.unique(all_labels))) and (len(np.unique(Y1)) == len(np.unique(all_labels))) and (random_state != 106) and (random_state != 112) and (random_state != 206):
            done = True
        else:
            random_state += 1
            skipped_states += 1
    return T1, T2, Y1, Y2, skipped_states

def load_wdbc_data(datafile, random_state, skipped_states):
    """
    load UCI WDBC data
    """
    random_state += skipped_states
    all_points = np.genfromtxt(datafile, delimiter=',', usecols=(range(2, 31)))
    all_labels = np.genfromtxt(datafile, delimiter=',', usecols=(1), dtype=[(str, 1)])
    all_labels_new = np.empty((0,0))
    for i in range(len(all_labels)):
        if all_labels[i][0] == 'M':
            all_labels_new = np.append(all_labels_new, float('1.0'))
        if all_labels[i][0] == 'B':
            all_labels_new = np.append(all_labels_new, float('0.0'))
    done = False
    while not done:
        # we don't want all labels to have the same class
        T2, T1, Y2, Y1 = train_test_split(all_points, all_labels_new, train_size=100, test_size = 10, random_state=random_state)
        if len(np.unique(Y2)) == len(np.unique(Y1)):
            done = True
        else:
            random_state += 1
            skipped_states += 1
    return T1, T2, Y1, Y2, skipped_states

def full_iterate(T1, T2, T1Labels, T2Labels, args, lmbd=1000):
    RPC = RPC_Classifier(T1, T1Labels, T2, args.proto_init_type, args.loglevel, lmbd)
    RPC.log.info("Initialized RPC classifier")
    RPC.log.info("Training RPC classifier")
    T1_best, T1_LABELS_BEST, T2LabelsBest, alpha_best, W_LABELS_BEST = RPC.model_do_training()
    RPC.log.info("Training completed")
    RPC.log.debug("Final labels: %s" % T2LabelsBest)
    RPC.log.debug("Target labels: %s" % T2Labels)
    if args.clustermap:
        RPC.cluster_map()
        plt.show()

    return RPC.get_summary(T2LabelsBest)


def main():
    """
    main function body
    """
    args = get_args()

    if args.othermodelsplot:
        plot_other_models()
        sys.exit(0)

    if args.timexperiment:
        times_list = time_experiment(args)

    if args.timedimexperiment:
        times_list = dimension_time_experiment(args)

    if args.inittimeexperiment:
        times_list = init_time_experiment(args)

    if args.data == 't1t2experiment':
        t1t2experiment(args)

    if args.plott1t2:
        print np.sum(np.array([[0.65, 0.65, 0.90000000000000002, 0.95999999999999996, 0.9, 0.97, 1.0], [0.73, 0.73, 0.72999999999999998, 0.95999999999999996, 0.97, 0.02, 0.76], [0.66, 0.66, 0.90000000000000002, 0.98999999999999999, 0.95, 0.8, 1.0], [0.58, 0.58, 0.93999999999999995, 0.92000000000000004, 0.94, 0.93, 1.0], [0.59, 0.59, 0.58999999999999997, 0.93999999999999995, 0.83, 0.93, 1.0], [0.56, 0.56, 0.76000000000000001, 0.94999999999999996, 0.91, 0.94, 1.0], [0.59, 0.59, 0.58999999999999997, 0.93999999999999995, 0.93, 0.93, 0.86], [0.54, 0.54, 0.54000000000000004, 0.81000000000000005, 0.71, 0.5, 1.0], [0.67, 0.67, 0.67000000000000004, 0.76000000000000001, 0.83, 0.67, 0.75], [0.62, 0.62, 0.93000000000000005, 0.91000000000000003, 0.91, 0.65, 0.81], [0.62, 0.62, 0.90000000000000002, 0.94999999999999996, 0.87, 0.62, 1.0], [0.59, 0.59, 0.93999999999999995, 0.91000000000000003, 0.83, 0.76, 0.68], [0.66, 0.66, 0.66000000000000003, 0.66000000000000003, 0.66, 0.73, 1.0], [0.61, 0.61, 0.94999999999999996, 0.94999999999999996, 0.93, 0.61, 0.77], [0.62, 0.62, 0.62, 0.89000000000000001, 0.72, 0.13, 1.0], [0.67, 0.67, 0.67000000000000004, 0.87, 0.87, 0.86, 0.74], [0.63, 0.63, 0.79000000000000004, 0.91000000000000003, 0.88, 0.37, 1.0], [0.54, 0.54, 0.81999999999999995, 0.90000000000000002, 0.85, 0.61, 0.71], [0.62, 0.62, 0.62, 0.78000000000000003, 0.88, 0.91, 1.0], [0.64, 0.64, 0.64000000000000001, 0.94999999999999996, 0.87, 0.91, 1.0], [0.64, 0.64, 0.64000000000000001, 0.81000000000000005, 0.79, 0.85, 0.68], [0.6, 0.6, 0.59999999999999998, 0.87, 0.93, 0.08, 0.85], [0.65, 0.65, 0.65000000000000002, 0.80000000000000004, 0.65, 0.68, 1.0], [0.67, 0.67, 0.67000000000000004, 0.94999999999999996, 0.94, 0.97, 1.0], [0.6, 0.6, 0.59999999999999998, 0.90000000000000002, 0.91, 0.9, 1.0], [0.66, 0.66, 0.66000000000000003, 0.89000000000000001, 0.89, 0.86, 1.0], [0.65, 0.65, 0.34999999999999998, 0.71999999999999997, 0.46, 0.14, 0.61], [0.67, 0.67, 0.67000000000000004, 0.85999999999999999, 0.9, 0.06, 1.0], [0.64, 0.64, 0.89000000000000001, 0.90000000000000002, 0.9, 0.91, 1.0], [0.57, 0.57, 0.56999999999999995, 0.78000000000000003, 0.57, 0.18, 1.0]]), axis = 0)/30

        result = [0.62466667, 0.62466667, 0.71533333,  0.87966667,  0.83933333,  0.64933333, 0.90733333]
        plott1t2()
        sys.exit(0)

    if args.lambdatestplot:
        lmbd_real = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        lmbd = range(len(lmbd_real))
        data = [0.83500000000000008, 0.81466666666666654, 0.84200000000000008, 0.83233333333333326, 0.91566666666666685, 0.86433333333333318, 0.92733333333333345, 0.8773333333333333, 0.877, 0.85499999999999998, 0.88533333333333342, 0.93099999999999994, 0.94333333333333325, 0.94800000000000006, 0.94100000000000006, 0.97166666666666679, 0.93999999999999995, 0.96766666666666667, 0.94833333333333336]
        std = [0.23283398950038783, 0.27467233003870056, 0.20572473518433962, 0.19390175060810794, 0.17867444012940287, 0.20431484418797274, 0.12787841447597356, 0.16202743251958568, 0.16421835057832807, 0.15713582235335985, 0.1801246481992092, 0.15763565586503583, 0.11830281296552314, 0.079052725021553388, 0.099510468461028442, 0.065781625262851487, 0.089628864398325014, 0.073334090905177868, 0.082868704721518499]
        fig, ax = plt.subplots()
        plt.xticks(lmbd, lmbd_real)
        plt.xlim((-1,19))
        plt.xlabel("Lambda value")
        plt.ylabel("Accuracy rate")
        plt.errorbar(lmbd, data, std, linestyle='dotted', marker='o', ecolor='r')
        plt.show()
        sys.exit()

    if args.server_data:
        v_length = 3
        data_accumulator = []
        API = BaseAPI(freq=30, v_len=v_length, type='real')
        Listener = DataListener(v_length, API)
        input = raw_input("Press Enter to stop...\n")
        # TODO: add DB here from the draft version
        if input == "":
            API.stop_api()
            Listener.stop_listener()
            print "Exit!"

    if args.simulate:
        data_accumulator = []
        API = BaseAPI(freq=5, v_len=5, type='fake')
        Listener = DataListener(5, API)
        input = raw_input("Press Enter to stop...\n")
        if input == "":
            API.stop_api()
            Listener.stop_listener()
            print "Exit!"

    if not args.simulate:
        if args.lambdatest:
            num_runs = 30
            acc_result = []
            lmbd_std = []
            for _, lmbd in enumerate([600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]):
                lmbd_acc_list = []
                for i in range(num_runs):
                    T1, T2, T1Labels, T2Labels = construct_data(args.data, args.toy_type, n_samples=10, t1_div_t2_rate=0.1)
                    T1, T2 = normalize_data(T1, T2, args)
                    print "Training RPC with non-const lambda"
                    print "Lambda %s" % lmbd
                    acc_run = full_iterate(T1, T2, T1Labels, T2Labels, args, lmbd=lmbd )
                    print "Accuracy for this SSL-RPC run: %s" % acc_run
                    lmbd_acc_list.append(acc_run)
                print "This lambda result: %s" % lmbd_acc_list
                lmbd_std.append(np.std(np.array(lmbd_acc_list)))
                lmbd_av_acc = sum(np.array(lmbd_acc_list))/num_runs
                acc_result.append(lmbd_av_acc)
                print "Current acc_result %s" % acc_result
            print "Final accuracy for all lambdas is:"
            print acc_result
            print "STD is: %s" % lmbd_std
            sys.exit(0)

        if args.toydataset:
            num_runs = 30
            acc_list = []
            acc_result = []
            for i in range(num_runs):
                # T1 10, T2 100
                T1, T2, T1Labels, T2Labels = construct_data(args.data, args.toy_type, n_samples=10, t1_div_t2_rate=0.1)
                T1, T2 = normalize_data(T1, T2, args)
                print "Training RPC"
                acc_run = full_iterate(T1, T2, T1Labels, T2Labels, args)
                print "Accuracy for this SSL-RPC run: %s" % acc_run
                print "Training RPC NON LAMBDED"
                acc_nonlmbd_run = full_iterate(T1, T2, T1Labels, T2Labels, args, lmbd=1)
                print "Training others"
                acc_list = other_models_experiment(T1, T2, T1Labels, T2Labels)
                acc_list.append(acc_run)
                acc_list.append(acc_nonlmbd_run)
                print "This run result: %s" % acc_list
                acc_result.append(acc_list)
                print "Current acc_result %s" % acc_result
            acc_result = np.sum(np.array(acc_result), axis=0)/num_runs
            print acc_result
            sys.exit(0)
        if args.knowndataset:
            num_runs = 30
            acc_list = []
            skipped = 0
            margin = 52
            acc_result = []
            for i in range(num_runs):
                print args.knowndataset
                if args.knowndataset == 'haberman':
                    T1, T2, T1Labels, T2Labels, skipped=load_haberman_data('haberman.data', random_state=margin+i, skipped_states=skipped)

                if args.knowndataset == 'wdbc':
                    T1, T2, T1Labels, T2Labels, skipped=load_wdbc_data('wdbc.data',random_state=margin+i, skipped_states=skipped)

                if args.knowndataset == 'ecoli':
                    T1, T2, T1Labels, T2Labels, skipped=load_ecoli_data('ecoli.data',random_state=margin+i, skipped_states=skipped)
                T1, T2 = normalize_data(T1, T2, args)
                print "Training RPC"
                acc_run = full_iterate(T1, T2, T1Labels, T2Labels, args)
                print "Accuracy for this SSL-RPC run: %s" % acc_run
                print "Training RPC NON LAMBDED"
                acc_nonlmbd_run = full_iterate(T1, T2, T1Labels, T2Labels, args, lmbd=1)
                print "Training others"
                acc_list = other_models_experiment(T1, T2, T1Labels, T2Labels)
                acc_list.append(acc_run)
                acc_list.append(acc_nonlmbd_run)
                print "This run result: %s" % acc_list
                acc_result.append(acc_list)
                print "Current acc_result %s" % acc_result
            acc_result = np.sum(np.array(acc_result), axis=0)/num_runs
            print acc_result
        else:
            if args.scikit:
                clf = RPC_Classifier(proto_init_type='dataset', loglevel='debug', lmbd=1000, iter_num = 10)
                T1, T2, T1Labels, T2Labels = construct_data(args.data, args.toy_type, args.n_samples, args.num_clusters)
                clf.fit(T1=T1, T2=T2, T1Labels=T1Labels)
                print clf.predict()
                print clf.score(real_labels=T2Labels)
            else:
                T1, T2, T1Labels, T2Labels = construct_data(args.data, args.toy_type, args.n_samples, args.num_clusters)
                T1, T2 = normalize_data(T1, T2)
                full_iterate(T1, T2, T1Labels, T2Labels, args)

    if args.other_models and not args.data == 't1t2experiment':
        print "KMeans accuracy\n%s"%calculateKMeans(T1, T2, T1Labels, T2Labels)
        other_models_experiment(T1, T2, T1Labels, T2Labels)

    if args.timexperiment or args.timedimexperiment or args.inittimeexperiment:
        print "Times list:"
        print times_list


if __name__ == "__main__":
    main()
