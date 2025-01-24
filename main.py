import os
import shutil
from iris import iris_class
from tools.file_manager import configuration, load_dataset, directory_exists, move_directory
from identification import id_class
import warnings
import cv2 as cv
from tools.utils import iris_code_plot, ROC_curve
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def generate_folder():
   if directory_exists('images'):
      shutil.rmtree('images')

   os.mkdir('images')      

   if not directory_exists('original'):
      os.mkdir('original')
   if not directory_exists('segmented'):
      os.mkdir('segmented')
   if not directory_exists('normalized'):
      os.mkdir('normalized')
   if not directory_exists('iris code'):
      os.mkdir('iris code')
   if not directory_exists('keypoint'):
      os.mkdir('keypoint')

def load_folder():
   move_directory('original', 'images')
   move_directory('segmented', 'images')
   move_directory('normalized', 'images')
   move_directory('iris code', 'images')
   move_directory('keypoint', 'images')

def load_iris(eye, sub_index, image_index):
   iris_obtained = iris_class(eye, sub_index, config)
   iris_obtained.segmentation()
   iris_obtained.feature_extraction()
   iris_obtained.set_iris_code()   
   iris_code = iris_obtained.get_iris_code()      
         
   path = os.path.join('original', str(image_index)+'.jpeg')
   cv.imwrite(path, iris_obtained.get_image())
   path = os.path.join('segmented', str(image_index)+'.jpeg')
   cv.imwrite(path, iris_obtained.get_segmented_image())
   path = os.path.join('normalized', str(image_index)+'.jpeg')
   cv.imwrite(path, iris_obtained.get_normalized_image())
   path = os.path.join('iris code', str(image_index)+'.jpeg')
   iris_code_plot(iris_code, path)
   path = os.path.join('keypoint', str(image_index)+'.jpeg')
   cv.imwrite(path, iris_obtained.get_keypoints_image())
   return iris_obtained


def load_irises(dataset):
   irises = []
   irises_stored = {i: [] for i in range (0, 100)}
   index = 0

   for rec_index, rec in enumerate(dataset):
      for subject in range(0, 108):
         eye = rec[subject]
         if subject >= 100: # Unauthorized subject
            iris_obtained = load_iris(eye, subject, index)
            irises.append(iris_obtained)
         elif rec_index in [0, 1, 4, 5]: # Rilevazione usata per il train
            iris_obtained = load_iris(eye, subject, index)
            irises_stored[subject].append(iris_obtained)
         else: # Rilevazione usata per il test
            iris_obtained = load_iris(eye, subject, index)
            irises.append(iris_obtained)
         index += 1

   return irises, irises_stored


def test(irises, irises_stored, threshold=None):
   print('\n')
   if threshold is not None:
      print(" Threshold : " + str(threshold))
   id = id_class(config, irises_stored)
   tp, fp, tn, fn, tot = 0, 0, 0, 0, 0

   for iris in irises:
      tot += 1
      flag, label = id.identification(iris, threshold)
      
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
   '''
   print('\tTrue Positive ' + str(tp))
   print('\tFalse Positive ' + str(fp))
   print('\tTrue Negative ' + str(tn))
   print('\tFalse Negative ' + str(fn))
   '''
   #print('\nPerformance achieved (' + str(tot) + ')')
   accuracy = (tp + tn) / tot * 100
   far = fp / (fp + tn) * 100
   frr = fn / (fn + tp) * 100
   
   print('\taccuracy ' + str(round(accuracy, 2)) + " %")
   print('\tFAR ' + str(round(far, 2)) + " %")
   print('\tFRR ' + str(round(frr, 2)) + " %")


   return far, frr


def ROC_Curve(far, frr):
   tpr = [100 - value for value in frr]
   plt.figure()
   plt.plot(far, tpr, marker="o", label="ROC Curve")
   plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Riferimento (casuale)")
   plt.xlabel("False Positive Rate (FAR)")
   plt.ylabel("True Positive Rate (1 - FRR)")
   plt.title("Curva ROC da FAR e FRR")
   plt.legend(loc="lower right")
   plt.grid()
   plt.show()



if  __name__ == '__main__':
   
   # Load the configuration file
   config = configuration()
   
   # Load CASIA-v1 dataset
   casia_dataset = load_dataset(config)
   
   #Loading irises
   print('\nLoading images...')
   print('\n Original -Segmentation - Normalization - Iris Code - Keypoints - Images Generation in progress...\n')
   generate_folder()
   irises, irises_stored = load_irises(casia_dataset)
   load_folder()

   # Test system
   thresholds = [6, 14, 22, 30, 38]
   far = []
   frr = []

   for threshold in thresholds:
         far_x, frr_x = test(irises, irises_stored, threshold)
         far.append(far_x)
         frr.append(frr_x)

   if not directory_exists('graph'):
      os.mkdir('graph')

   path = os.path.join('graph', 'ROC_curve.jpeg')
   ROC_curve(far, frr, thresholds, path)