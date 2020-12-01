import numpy as np
import scipy.sparse as sps

from Base.DataIO import DataIO

"""
def _clone_dictionary(original_dict):
    clone_dict = {key:value.copy() for key,value in original_dict.items()}
    return clone_dict

def gini_index(array):
   
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.array(array, dtype=np.float)
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

"""

class Dataset(object):

    DATASET_NAME = None

    # Available URM split
    AVAILABLE_URM = {}

    # Available ICM for the given dataset, there might be no ICM, one or many
    AVAILABLE_ICM = {}
    # AVAILABLE_ICM_feature_mapper = {}
    # _HAS_ICM = True

    item_original_ID_to_index = {}
    user_original_ID_to_index = {}

    # additional_data_mapper = {}
    # _HAS_additional_mapper = False

    # _IS_IMPLICIT = False

    # Mappers specific for a given dataset, they might be related to more complex data structures or FEATURE_TOKENs
    # DATASET_SPECIFIC_MAPPER = []


    def __init__(self, dataset_name,
                 URM_train,
                 URM_test,
                 ICM_dictionary
                 #ICM_feature_mapper_dictionary = None,
                 #UCM_dictionary = None,
                 #UCM_feature_mapper_dictionary = None,
                #  user_original_ID_to_index = None,
                #  item_original_ID_to_index = None,
                 #is_implicit = False,
                 #additional_data_mapper = None,
                 ):
        """
        :param URM_dictionary:                      Dictionary of "URM_name":URM_object
        :param ICM_dictionary:                      Dictionary of "ICM_name":ICM_object
        :param ICM_feature_mapper_dictionary:       Dictionary of "ICM_name":feature_original_id_to_index
        :param UCM_dictionary:                      Dictionary of "UCM_name":UCM_object
        :param UCM_feature_mapper_dictionary:       Dictionary of "UCM_name":feature_original_id_to_index
        :param user_original_ID_to_index:           Dictionary of "user_original_id":user_index
        :param item_original_ID_to_index:           Dictionary of "item_original_id":user_index (credo sia item_index)
        """
        super(Dataset, self).__init__()

        self.DATASET_NAME = dataset_name
        self.URM_train = URM_train
        self.URM_test = URM_test

        #if ICM_dictionary is not None and len(ICM_dictionary)>0:
        self.AVAILABLE_ICM = ICM_dictionary
        # self.AVAILABLE_ICM_feature_mapper = ICM_feature_mapper_dictionary
        # self._HAS_ICM = True



    #########################################################################################################
    ##########                                                                                     ##########
    ##########                                LOAD AND SAVE                                        ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def save_data(self, save_folder_path):

        dataIO = DataIO(folder_path = save_folder_path)
        
        print(self.URM_train)
        print(self.URM_test)
        print(self.AVAILABLE_ICM)
        print(self.DATASET_NAME)
        global_attributes_dict = {
            "DATASET_NAME": self.DATASET_NAME
        }

        dataIO.save_data(file_name = "dataset_global_attributes", data_dict_to_save = global_attributes_dict)

        dataIO.save_data(file_name = "dataset_URM_train", data_dict_to_save = self.URM_train)

        dataIO.save_data(data_dict_to_save = self.URM_test,
            file_name = "dataset_URM_test")

        dataIO.save_data(data_dict_to_save = self.AVAILABLE_ICM,
            file_name = "dataset_ICM")
