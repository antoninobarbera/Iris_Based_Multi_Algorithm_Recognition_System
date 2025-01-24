import os
from iris import iris_class
from tools.file_manager import configuration
from data_classes.manage_dataset import CASIA_dataset
from data_classes.manage_dataset import Manage_file
from identification import id_class
import warnings
from tools import utils

warnings.filterwarnings("ignore")

if  __name__ == '__main__':
   
   # Load the configuration file
   config = configuration()
   
   # Initialize the CASIA dataset handler
   casia_dataset = CASIA_dataset(config)
   
   # Load the dataset into test, total, and data record splits
   rec_test, rec_tot, data_rec = casia_dataset.load_dataset()
   
   # Initialize the file manager for creating folders
   manage_file = Manage_file()
   segmented_path, keypoints_path, normalized_path = manage_file.create_folder_image()
   
   print('\nLoading images...')
   print('\nSegmentation - Normalization - Feature extraction in progress...\n')

   segmented_images = []
   keypoints_images = []
   normalized_images = []
   irises = []
   
   print('Operation in test image...')
   for rec in rec_test:
      for i in range (0, 100):
         # Retrieve the eye image and process it using the iris class
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         
         # Perform segmentation on the eye image
         iris_obtained.segmentation()
         segmented_image = iris_obtained.get_segmented_image()
         segmented_images.append((segmented_image, os.path.join(segmented_path, f'{i}.jpeg')))
         
         # Perform feature extraction and retrieve the keypoints image
         iris_obtained.feature_extraction()
         keypoints_image = iris_obtained.get_keypoints_image()
         
         # Generate the iris code and store normalized images
         iris_obtained.set_iris_code()
         normalized_image = iris_obtained.get_normalized_image()
         
         irises.append(iris_obtained)
         keypoints_images.append((keypoints_image, os.path.join(keypoints_path, f'{i}.jpeg')))
         normalized_images.append((normalized_image, os.path.join(normalized_path, f'{i}.jpeg')))
   
   # Save processed images to the appropriate directories
   manage_file.save_image(segmented_images, keypoints_images, normalized_images)
   
   print('Operation in total image...')
   for rec in rec_tot:
      for i in range (100, 108):
         
         # Retrieve the eye image and process it using the iris class
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         
         # Perform segmentation and feature extraction
         iris_obtained.segmentation()
         iris_obtained.feature_extraction()
         
         # Retrieve the keypoints image and set the iris code
         iris_obtained.set_iris_code()
         irises.append(iris_obtained)

   print('Operation in train_image...')
   data_dict = {i : [] for i in range(0, 108)}
   for rec in data_rec:
      for i in range (0, 100):
         # Retrieve the eye image and process it using the iris class
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         
         # Perform segmentation and feature extraction
         iris_obtained.segmentation()
         iris_obtained.feature_extraction()
         
         # Retrieve the keypoints image and set the iris code
         iris_obtained.set_iris_code()
         data_dict[i].append(iris_obtained)
               
   # Initialize the identification class with the processed data
   id = id_class(config, data_dict)
   tp, fp, tn, fn, tot = 0, 0, 0, 0, 0
   
   print('\nIdentification:')
   for iris in irises:
      tot += 1
      flag, label = id.identification(iris)
      
      # Evaluate identification results
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

   print('\tTrue Positive ' + str(tp))
   print('\tFalse Positive ' + str(fp))
   print('\tTrue Negative ' + str(tn))
   print('\tFalse Negative ' + str(fn))

   print('\nPerformance achieved (' + str(tot) + ')')
   accuracy = (tp + tn) / tot * 100
   far = fp / (fp + tn) * 100
   frr = fn / (fn + tp) * 100
   
   print('\taccuracy ' + str(round(accuracy, 2)) + " %")
   print('\tFAR ' + str(round(far, 2)) + " %")
   print('\tFRR ' + str(round(frr, 2)) + " %")
   
   utils.identification_performance(tp, fp, tn, fn)
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   '''import numpy as np
   import matplotlib.pyplot as plt
   
   thresholds = np.arange(12, 32, 2)
   frr_values = []
   far_values = []

   print('\nCalculating FRR and FAR for different thresholds...')
   for threshold in thresholds:
      tp, fp, tn, fn = 0, 0, 0, 0

      # Set threshold in the identification class (implement this in id_class)
      id.set_threshold(threshold)

      for iris in irises:
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

      # Calculate FRR and FAR
      frr = fn / (fn + tp) if (fn + tp) > 0 else 0
      far = fp / (fp + tn) if (fp + tn) > 0 else 0

      frr_values.append(frr)
      far_values.append(far)

   # Plot the FRR and FAR curves
   plt.figure(figsize=(10, 6))
   plt.plot(thresholds, frr_values, label="FRR (False Rejection Rate)", color="red")
   plt.plot(thresholds, far_values, label="FAR (False Acceptance Rate)", color="blue")
   plt.xlabel("Threshold", fontsize=14)
   plt.ylabel("Rate", fontsize=14)
   plt.title("FRR / FAR Threshold", fontsize=16)
   plt.legend(fontsize=12)
   plt.grid(alpha=0.7)
   plt.tight_layout()
   plt.show()'''
   