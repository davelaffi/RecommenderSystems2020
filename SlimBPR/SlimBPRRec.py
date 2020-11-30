import time
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from SlimBPR.SlimBPR import SlimBPR

from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class SlimBPRRec(object):
    
    def __init__(self, learning_rate, epochs, nnz, knn):
      
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.positive_item_regularization = 1.0
        self.negative_item_regularization = 1.0
        self.nnz = nnz
        self.knn = knn

    def fit(self, URM):
        
        self.URM = URM
        

        # Compute similarity matrix
        self.SlimSimilarity = SlimBPR(URM,
                            self.learning_rate,
                            self.epochs,
                            self.positive_item_regularization,
                            self.negative_item_regularization,
                            self.nnz).get_S_SLIM_BPR(self.knn)
        
        self.RECS = self.URM.dot(self.SlimSimilarity)
        
    
        
    def get_expected_ratings(self,user_id):
        expected_ratings = self.RECS[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, user_id,urm_train: sp.csr_matrix, at=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0) #Ordino gli expected ratings

        unseen_items_mask = np.in1d(recommended_items,urm_train[user_id].indices,
                                        assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]