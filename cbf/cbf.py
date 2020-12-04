import numpy as np
import scipy.sparse as sp
from Base.BaseRecommender import BaseRecommender
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class ContentBasedFiltering(BaseRecommender):

    RECOMMENDER_NAME = "ContentBasedRecommender"

    def __init__(self,URM, ICM):
        #super(ContentBasedFiltering, self).__init__(URM)
        self.URM_train = URM
        self.ICM = ICM

    def compute_similarity(self):

        similarity_object = Compute_Similarity_Python(self.ICM.transpose(), topK=self.knn, shrink = self.shrink,
                                                     normalize=True,similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self,knn,shrink,similarity):
        self.URM_train = self.URM_train.tocsr()
        self.knn=knn
        self.shrink=shrink
        self.similarity = similarity
        self.W_sparse = self.compute_similarity()
        self.RECS = self.URM_train.dot(self.W_sparse)
