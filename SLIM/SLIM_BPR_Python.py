#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 2017
Updated on 28 November 2020

@author: Maurizio Ferrari Dacrema
"""

import time
import numpy as np
import scipy.sparse as sp

#from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Base.Recommender_utils import similarityMatrixTopK


class SLIM_BPR_Python(object):
    """
    This class is a python porting of the BPRSLIM algorithm in MyMediaLite written in C#

    This class does not implement early stopping
    """

    def __init__(self,topK = 100, epochs = 25, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05):
        #super(SLIM_BPR_Python, self).__init__(URM_train)
        self.topK=topK
        self.epochs=epochs
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate

    def fit(self,URM):
        
        self.URM_train = sp.csc_matrix(URM)
        self.n_users = self.URM_train.shape[0]
        self.n_items = self.URM_train.shape[1]
        # Initialize similarity with zero values
        self.item_item_S = np.zeros((self.n_items, self.n_items), dtype = np.float)

        start_time_train = time.time()

        for n_epoch in range(self.epochs):
            self._run_epoch(n_epoch)

        print("Train completed in {:.2f} minutes".format(float(time.time()-start_time_train)/60))

        self.W_sparse = similarityMatrixTopK(self.item_item_S, k=self.topK, verbose=False)
        self.W_sparse = sp.csr_matrix(self.W_sparse)


    def _run_epoch(self, n_epoch):

        start_time = time.time()

        # Uniform user sampling without replacement
        for sample_num in range(self.n_users):

            user_id, pos_item_id, neg_item_id = self._sample_triplet()

            # Calculate current predicted score
            user_seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id+1]]

            # Compute positive and negative item predictions. Assuming implicit interactions.
            x_ui = self.item_item_S[pos_item_id, user_seen_items].sum()
            x_uj = self.item_item_S[neg_item_id, user_seen_items].sum()

            # Gradient
            x_uij = x_ui - x_uj
            sigmoid_gradient = 1 / (1 + np.exp(x_uij))

            # Update
            self.item_item_S[pos_item_id, user_seen_items] += self.learning_rate * (sigmoid_gradient - self.lambda_i * self.item_item_S[pos_item_id, user_seen_items])
            self.item_item_S[pos_item_id, pos_item_id] = 0

            self.item_item_S[neg_item_id, user_seen_items] -= self.learning_rate * (sigmoid_gradient - self.lambda_j * self.item_item_S[neg_item_id, user_seen_items])
            self.item_item_S[neg_item_id, neg_item_id] = 0

            # Print some stats
            if (sample_num + 1) % 150000 == 0 or (sample_num + 1) == self.n_users:
                elapsed_time = time.time() - start_time
                samples_per_second = (sample_num + 1) / elapsed_time
                print("Epoch {}, Iteration {} in {:.2f} seconds. Samples per second {:.2f}".format(n_epoch+1, sample_num+1, elapsed_time, samples_per_second))

                start_time = time.time()



    def _sample_triplet(self):

        non_empty_user = False

        while not non_empty_user:
            user_id = np.random.choice(self.n_users)
            user_seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

            if len(user_seen_items) > 0:
                non_empty_user = True

        pos_item_id = np.random.choice(user_seen_items)

        neg_item_selected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not neg_item_selected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in user_seen_items):
                neg_item_selected = True

        return user_id, pos_item_id, neg_item_id

    def get_expected_ratings(self, user_id):
        user_profile = self.URM_train[user_id]
        expected_ratings = user_profile.dot(self.W_sparse).toarray().ravel()

        # # EDIT
        return expected_ratings
    
    def recommend(self,user_id,urm_train: sp.csr_matrix,at=10):
        # compute the scores using the dot product
        scores = self.get_expected_ratings(user_id)
        user_profile = self.URM_train[user_id].indices
        scores[user_profile] = 0

        # rank items
        recommended_items = np.flip(np.argsort(scores), 0)

        return recommended_items[:at]