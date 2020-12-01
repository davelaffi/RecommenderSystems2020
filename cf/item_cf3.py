import numpy as np
import scipy.sparse as sp
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
from Base.BaseRecommender import BaseRecommender


class ItemBasedCollaborativeFiltering(BaseRecommender):

    RECOMMENDER_NAME = "ItemBasedCFRecommender"

    def __init__(self,URM):
        self.URM_train = URM
        self.W_sparse = None

    def generate_similarity_matrix(self):
        similarity_object = Compute_Similarity_Python(self.URM_train, topK =self.knn, shrink = self.shrink, normalize=True, similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self,knn, shrink, similarity):
        self.URM_train = self.URM_train.tocsr()
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.W_sparse = self.generate_similarity_matrix()


    