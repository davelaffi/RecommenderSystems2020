import os, traceback
from Data_manager.Dataset import Dataset


#################################################################################################################
#############################
#############################               DATA READER
#############################
#################################################################################################################



class DataReader(object):

    

    
    def save_data(self, save_folder_path):

        dataIO = DataIO(folder_path = save_folder_path)

        global_attributes_dict = {
            "item_original_ID_to_index": self.item_original_ID_to_index,
            "user_original_ID_to_index": self.user_original_ID_to_index,
            "DATASET_NAME": self.DATASET_NAME,
            "_IS_IMPLICIT": self._IS_IMPLICIT,
            "_HAS_ICM": self._HAS_ICM,
            "_HAS_UCM": self._HAS_UCM,
            "_HAS_additional_mapper": self._HAS_additional_mapper
        }

        dataIO.save_data(data_dict_to_save = global_attributes_dict,
                         file_name = "dataset_global_attributes")

        dataIO.save_data(data_dict_to_save = self.AVAILABLE_URM,
                         file_name = "dataset_URM")

        if self._HAS_ICM:
            dataIO.save_data(data_dict_to_save = self.AVAILABLE_ICM,
                             file_name = "dataset_ICM")

            dataIO.save_data(data_dict_to_save = self.AVAILABLE_ICM_feature_mapper,
                             file_name = "dataset_ICM_mappers")

        if self._HAS_UCM:
            dataIO.save_data(data_dict_to_save = self.AVAILABLE_UCM,
                             file_name = "dataset_UCM")

            dataIO.save_data(data_dict_to_save = self.AVAILABLE_UCM_feature_mapper,
                             file_name = "dataset_UCM_mappers")

        if self._HAS_additional_mapper:
            dataIO.save_data(data_dict_to_save = self.additional_data_mapper,
                             file_name = "dataset_additional_mappers")