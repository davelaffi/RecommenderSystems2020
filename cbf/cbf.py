import numpy as np
import scipy.sparse as sp

from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class ContentBasedFiltering(object):

    def __init__(self, knn=100, shrink=2):
        self.knn = knn
        self.shrink = shrink

    def compute_similarity(self, ICM, knn, shrink):
        similarity_object = Compute_Similarity_Python(ICM.transpose(), topK=knn, shrink=shrink,
                                                     normalize=True, similarity="cosine")
        return similarity_object.compute_similarity()

    def fit(self, URM, ICM):

        self.URM = URM
        self.ICM = ICM

        self.SM = self.compute_similarity(self.ICM, self.knn, self.shrink)

    def get_expected_ratings(self, user_id):
        
        items_all = self.URM[user_id] #URM_all
        expected_ratings_all_items = items_all.dot(self.SM).toarray().ravel()
        
        return expected_ratings_all_items

    def recommend(self, user_id, urm_train: sp.csr_matrix, at):
        
        expected_ratings_all_items = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings_all_items), 0)
        return recommended_items[0:at]
