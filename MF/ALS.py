import numpy as np
import implicit
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp




class AlternatingLeastSquare:

    def __init__(self, n_factors, regularization, iterations): #Dobbiamo passarglieli come parametri
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, URM):
        self.URM = URM

        sparse_item_user = self.URM.T

        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization, iterations=self.iterations)


        alpha_val = 24
        # Calculate the confidence by multiplying it by our alpha value.

        data_conf = (sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors


    def get_expected_ratings(self, user_id):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)

        return np.squeeze(scores)

    def recommend(self, user_id, urm_train: sp.csr_matrix, at=10):
        
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, urm_train[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]