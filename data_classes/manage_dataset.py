import os
import pickle
import cv2 as cv


class CASIA_dataset():
    
    __slots__ = ['config', 'rec_test', 'rec_tot', 'data_rec', 'rec_tot2']
    
    def __init__(self, config):
        self.config = config
        self.rec_test = None
        self.rec_tot = None
        self.data_rec = None
        self.rec_tot2 = None
    
    def load_dataset(self):
        dataset_path = 'dataset'
        casia_path = os.path.join(dataset_path, 'CASIA.pkl')
        
        if not os.path.exists(dataset_path):
            print(f"Error: The folder '{dataset_path}' does not exist.")
            return

        if not os.path.isfile(casia_path):
            print(f"Error: The file '{casia_path}' does not exist.")
            return
        
        casia_path = os.path.join('dataset', 'CASIA.pkl')
        
        try:
            with open(casia_path, 'rb') as file:
                casia = pickle.load(file)
        except Exception as e:
            print(f"Error: Failed to open or load the file '{casia_path}'.")
            print(f"Exception details: {str(e)}")
            return

        rec_1 = casia['rec_1']
        rec_2 = casia['rec_2']
        rec_3 = casia['rec_3']
        rec_4 = casia['rec_4']
        rec_5 = casia['rec_5']
        rec_6 = casia['rec_6']
        rec_7 = casia['rec_7']

        self.rec_test = [rec_3, rec_4, rec_7]
        self.rec_tot = [rec_3, rec_4, rec_7, rec_1, rec_2, rec_5, rec_6]
        self.data_rec = [rec_1, rec_2, rec_5, rec_6]
        
        self.rec_tot2 = [rec_1, rec_2, rec_3, rec_4, rec_5, rec_6, rec_7]
        
        return self.rec_test, self.rec_tot, self.data_rec
        
    def get_rec_test(self):
        return self.rec_test
    
    def get_rec_tot(self):
        return self.rec_tot
    
    def get_rec_tot2(self):
        return self.rec_tot2
        
    def get_data_rec(self):
        return self.data_rec
    
    
class Manage_file():
    
    def __init__(self):
        self.trasformed_path = 'trasformed_irises'
        self.segmented_path = 'segmented_irises'
        self.keypoints_path = 'keypoints_irises'
        self.normalized_path = 'normalized_irises'
        
    def create_folder_image(self):
        os.makedirs(self.trasformed_path, exist_ok=True)
        segmented_path = os.path.join(self.trasformed_path, self.segmented_path)
        keypoints_path = os.path.join(self.trasformed_path, self.keypoints_path)
        normalized_path = os.path.join(self.trasformed_path, self.normalized_path)
        os.makedirs(segmented_path, exist_ok=True)
        os.makedirs(keypoints_path, exist_ok=True)
        os.makedirs(normalized_path, exist_ok=True)
        
        return segmented_path, keypoints_path, normalized_path

    def save_image(self, segmented_images, keypoints_images, normalized_images):

        print("    Saving segmented images...")
        for image, path in segmented_images:
            cv.imwrite(path, image)
            
        print("    Saving keypoints images...")
        for image, path in keypoints_images:
            cv.imwrite(path, image)
            
        print("    Saving normalized images...")
        for image, path in normalized_images:
            cv.imwrite(path, image)

        print("    Images saved successfully!\n")
    