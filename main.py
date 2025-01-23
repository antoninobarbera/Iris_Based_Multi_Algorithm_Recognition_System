import os
import pickle
import cv2 as cv
from iris import iris_class
from tools.file_manager import configuration
from data_classes.manage_dataset import CASIA_dataset
from data_classes.manage_dataset import Manage_file
from identification import id_class
import warnings
import gc
import time

warnings.filterwarnings("ignore")

if  __name__ == '__main__':
   
   start_time = time.time()
   
   config = configuration()
   
   casia_dataset = CASIA_dataset(config)
   rec_test, rec_tot, data_rec = casia_dataset.load_dataset()
   
   manage_file = Manage_file()
   segmented_path, keypoints_path = manage_file.create_folder_image()
   
   print('\nLoading images...')
   print('\nSegmentation - Normalization - Feature extraction in progress...\n')

   segmented_images = []
   keypoints_images = []
   irises = []
   
   print('Operation in test image...')
   for rec in rec_test:
      for i in range (0, 100):
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         iris_obtained.segmentation()
         segmented_image = iris_obtained.get_segmented_image()
         segmented_images.append((segmented_image, os.path.join(segmented_path, f'{i}.jpeg')))
         iris_obtained.feature_extraction()
         keypoints_image = iris_obtained.get_keypoints_image()
         iris_obtained.set_iris_code()
         irises.append(iris_obtained)
         keypoints_images.append((keypoints_image, os.path.join(keypoints_path, f'{i}.jpeg')))
   
   manage_file.save_image(segmented_images, keypoints_images)
   
   print('Operation in total image...')
   for rec in rec_tot:
      for i in range (100, 108):
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         iris_obtained.segmentation()
         iris_obtained.feature_extraction()
         new_image = iris_obtained.get_keypoints_image()
         iris_obtained.set_iris_code()
         irises.append(iris_obtained)

   print('Operation in train_image...')
   data_dict = {i : [] for i in range(0, 108)}
   for rec in data_rec:
      for i in range (0, 100):
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         iris_obtained.segmentation()
         iris_obtained.feature_extraction()
         new_image = iris_obtained.get_keypoints_image()
         iris_obtained.set_iris_code()
         data_dict[i].append(iris_obtained)
         
   id = id_class(config, data_dict)
   tp, fp, tn, fn, tot = 0, 0, 0, 0, 0
   
   print('\nIdentification:')
   for iris in irises:
      tot += 1
      flag, label = id.identification(iris)
      if flag:
         if iris.get_idx() == label:
            tp += 1
         else:
            fp += 1
      else:
            if iris.get_idx() < 100:
               fn += 1
            else:
               tn += 1

   print('\tTP ' + str(tp))
   print('\tFP ' + str(fp))
   print('\tTN ' + str(tn))
   print('\tFN ' + str(fn))

   print('\nPerformance achieved (' + str(tot) + ')')
   accuracy = (tp + tn) / tot * 100
   far = fp / (fp + tn) * 100
   frr = fn / (fn + tp) * 100
   
   print('\taccuracy ' + str(round(accuracy, 2)) + " %")
   print('\tFAR ' + str(round(far, 2)) + " %")
   print('\tFRR ' + str(round(frr, 2)) + " %")
   
   end_time = time.time() 
   execution_time = end_time - start_time
   execution_time_minutes = execution_time / 60
   print(f"\nTempo di esecuzione: {execution_time:.2f} minuti")
   