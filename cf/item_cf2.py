import numpy as np
import scipy.sparse as sp
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python


class ItemBasedCollaborativeFiltering(object):

    RECOMMENDER_NAME = "ItemBasedCFRecommender"

    def __init__(self,URM):
        self.URM = URM
        self.SM_item = None

    def generate_similarity_matrix(self):
        similarity_object = Compute_Similarity_Python(self.URM, topK =self.knn, shrink = self.shrink, normalize=True, similarity=self.similarity)
        return similarity_object.compute_similarity()

    def fit(self,knn = 100, shrink = 5, similarity="cosine"):
        self.URM = self.URM.tocsr()
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.SM_item = self.generate_similarity_matrix()
        self.RECS = self.URM.dot(self.SM_item)

    def get_expected_ratings(self,user_id):
        expected_ratings = self.RECS[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, user_id,remove_seen_flag, at=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        
        if remove_seen_flag:
           unseen_items_mask = np.in1d(recommended_items,self.URM[user_id].indices,
                                        assume_unique=True, invert=True)
           recommended_items = recommended_items[unseen_items_mask]

           
        return recommended_items[0:at]
    
    def get_URM_train(self):
        return self.URM