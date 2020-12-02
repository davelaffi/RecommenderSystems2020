import time
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from SlimBPR.SlimBPR import SlimBPR
from Base.BaseRecommender import BaseRecommender
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class SlimBPRRec(BaseRecommender):

    RECOMMENDER_NAME = "SlimBPRRecommender"
    
    def __init__(self,URM):
        super(SlimBPRRec, self).__init__(URM)

    def fit(self,learning_rate, epochs, nnz, knn):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.positive_item_regularization = 1.0
        self.negative_item_regularization = 1.0
        self.nnz = nnz
        self.knn = knn
        
        # Compute similarity matrix
        self.SlimSimilarity = SlimBPR(self.URM_train,
                            self.learning_rate,
                            self.epochs,
                            self.positive_item_regularization,
                            self.negative_item_regularization,
                            self.nnz).get_S_SLIM_BPR(self.knn)
        
        self.RECS = self.URM_train.dot(self.SlimSimilarity)
