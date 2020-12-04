import numpy as np
import implicit
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from Base.BaseRecommender import BaseRecommender



class AlternatingLeastSquare(BaseRecommender):

    RECOMMENDER_NAME = "AlternatingLeastSquare"

    def __init__(self,URM):
        #super(AlternatingLeastSquare, self).__init__(URM)
        self.URM_train = URM


    def fit(self,n_factors, regularization, iterations):

        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations

        sparse_item_user = self.URM_train.T

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
        self.RECS = sp.csc_matrix(self.user_factors.dot(self.item_factors.T))
        #scores = np.dot(self.user_factors[user_id], self.item_factors.T)

