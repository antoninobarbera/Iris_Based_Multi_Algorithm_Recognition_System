import os
import numpy as np
import pickle
import shutil
import cv2 as cv
from iris import iris_class
from tools.file_manager import configuration, directory_exists, move_directory

def generate_img(config, rec_tot):   
    
    index = 0
    for rec in rec_tot:
      for i in range (0, 108):
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         iris_obtained.segmentation()
         iris_obtained.feature_extraction()
         iris_obtained.set_iris_code()        
         
         path = os.path.join('original', str(index)+'.jpeg')
         cv.imwrite(path, iris_obtained.get_image())
         path = os.path.join('segmented', str(index)+'.jpeg')
         cv.imwrite(path, iris_obtained.get_segmented_image())
         path = os.path.join('normalized', str(index)+'.jpeg')
         cv.imwrite(path, iris_obtained.get_normalized_image())
         path = os.path.join('keypoint', str(index)+'.jpeg')
         cv.imwrite(path, iris_obtained.get_keypoints_image())
         index += 1


if  __name__ == '__main__':    
   print(' CARICAMENTO IMMAGINI')   
   config = configuration()

   with open('CASIA.pkl', 'rb') as file:
       casia = pickle.load(file)

   rec_1 = casia['rec_1']
   rec_2 = casia['rec_2']
   rec_3 = casia['rec_3']
   rec_4 = casia['rec_4']
   rec_5 = casia['rec_5']
   rec_6 = casia['rec_6']
   rec_7 = casia['rec_7']

   rec_tot = [rec_1, rec_2, rec_3, rec_4, rec_5, rec_6, rec_7]

   if directory_exists('images'):
      shutil.rmtree('images')

   os.mkdir('images')
      

   if not directory_exists('original'):
      os.mkdir('original')

   if not directory_exists('segmented'):
      os.mkdir('segmented')

   if not directory_exists('normalized'):
      os.mkdir('normalized')

   if not directory_exists('keypoint'):
      os.mkdir('keypoint')

   generate_img(config, rec_tot)

   with open('iris_data.pkl', 'wb') as file:
      pickle.dump(dict, file)

   move_directory('original', 'images')
   move_directory('segmented', 'images')
   move_directory('normalized', 'images')
   move_directory('keypoint', 'images')

   print(' CARICAMENTO IMMAGINI COMPLETATO')   
      