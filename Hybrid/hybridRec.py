import numpy as np
import scipy.sparse as sp
from Base.DataIO import DataIO
from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python

#import recommenders
from SLIM.SLIM_BPR_Python import SLIM_BPR_Python
from SLIM.SlimElasticNet import SLIMElasticNetRecommender
from cf.item_cf3 import ItemBasedCollaborativeFiltering
from cf.user_cf2 import UserBasedCollaborativeFiltering
from MF.ALS import AlternatingLeastSquare
from cbf.cbf import ContentBasedFiltering
from SlimBPR.SlimBPRRec import SlimBPRRec
from SlimBPR.SlimBPR import SlimBPR


class HybridRecommender(object):

    RECOMMENDER_NAME = "HybridRecommender"

    def __init__(self, URM, ICM):
        
        self.URM_train = URM
        
        self.ICM = ICM

        self.userCF = UserBasedCollaborativeFiltering(URM.copy())

        self.itemCF = ItemBasedCollaborativeFiltering(URM.copy())
        
        self.cbf = ContentBasedFiltering(URM.copy(), ICM.copy())
        
#         self.slim_random = SLIM_BPR_Python(URM.copy())
        
#         self.slim_elastic = SLIMElasticNetRecommender()
        
#         self.ALS = AlternatingLeastSquare(URM.copy())
    

    def fit(self, user_cf_param, item_cf_param, cbf_param, slim_param, als_param, w_user, w_item , w_cbf):

        ######## Give weights to recommenders ##########
        #self.w = w
        self.w_user = w_user
        self.w_item = w_item
        self.w_cbf = w_cbf

        ################################################

        ### SUB-FITTING ###
        print("Fitting user cf...")
        self.userCF.fit(knn=user_cf_param["knn"], shrink=user_cf_param["shrink"], similarity="cosine")

        print("Fitting item cf...")
        self.itemCF.fit(knn=item_cf_param["knn"], shrink=item_cf_param["shrink"], similarity="cosine")
        
        print("Fitting cbf...")
        self.cbf.fit(knn=cbf_param["knn"],shrink=cbf_param["shrink"],similarity="cosine")
        
        print("Fitting slim bpr...")
#         self.slim_random.fit(topK=slim_param["topK"],epochs=slim_param["epochs"])
        
        print("Fitting slim elastic...")
#         self.slim_elastic.fit(URM.copy())
        
        print("Fitting ALS...")
#         self.ALS.fit(n_factors=als_param["n_factors"], regularization=als_param["regularization"],iterations=als_param["iterations"])

    
    def get_expected_ratings(self, user_id) :
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.cbf_ratings = self.cbf.get_expected_ratings(user_id)
#         self.slim_ratings = self.slim_random.get_expected_ratings(user_id)
#         self.slim_elastic_ratings = self.slim_elastic.get_expected_ratings(user_id)
#         self.ALS_ratings = self.ALS.get_expected_ratings(user_id)

        self.hybrid_ratings = None 

        self.hybrid_ratings = self.userCF_ratings * self.w_user
        self.hybrid_ratings += self.itemCF_ratings * self.w_item
        self.hybrid_ratings += self.cbf_ratings * self.w_cbf
#         self.hybrid_ratings += self.slim_ratings * self.w["slim"]
#         self.hybrid_ratings += self.ALS_ratings * self.w["als"]
#         self.hybrid_ratings += self.slim_elastic_ratings * self.w["elastic"]

        return self.hybrid_ratings




    def get_URM_train(self):
        return self.URM_train


    # non fatta
    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"Hybrid Ratings": self.get_expected_ratings()}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")

    def _print(self, string):
        print("{}: {}".format(self.RECOMMENDER_NAME, string))

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """
        
        item_scores = np.zeros((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)

        counter = 0

        for user_id in user_id_array :
            
            item_scores[counter] = self.get_expected_ratings(user_id)
            counter += 1

        return item_scores
    
    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

    def recommend2(self,user_id,urm_train: sp.csr_matrix,at=10):
     
        expected_ratings = self.get_expected_ratings(user_id)
        
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items,urm_train[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]

    def recommend(self, user_id_array, cutoff=10, remove_seen_flag=True, items_to_compute = None, remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=True):
        
       # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index,:] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)


        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()



        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]


        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list