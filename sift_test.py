import os
import pickle
import cv2 as cv
from iris import iris_class
from tools.file_manager import configuration
from identification import id_class
from data_classes.manage_dataset import CASIA_dataset

if  __name__ == '__main__':  
   
   config = configuration()
   
   casia_dataset = CASIA_dataset(config)
   casia_dataset.load_dataset()
   rec_tot = casia_dataset.get_rec_tot2()
   irises = []

   print('\nLoading data...')

   for rec in rec_tot:
      for i in range (0, 108):
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         iris_obtained.segmentation()
         iris_obtained.feature_extraction()
         irises.append(iris_obtained)
    
   print('\nCalculation...')

   rec_1 = irises[:108]
   rec_2 = irises[108:216]
   rec_3 = irises[216:324]
   rec_4 = irises[324:432]
   rec_5 = irises[432:540]
   rec_6 = irises[540:648]
   rec_7 = irises[648:756]

   id = id_class(config)

   tot_rec = [rec_1, rec_2, rec_3, rec_4, rec_5, rec_6, rec_7]
   num_err = 0
   tot = 0

   for i in range(0, 108):
      for j in range(len(tot_rec)):
         k = j + 1
         while(k < len(tot_rec)):
            match_x = id.sift_match(tot_rec[j][i], tot_rec[k][i])
            if not match_x:
               num_err += 1
            k += 1
            tot += 1
    
   print('\nFRR Calculation:')  
   print(' \tNum err : ' + str(num_err))          
   print(' \ttot : ' + str(tot))   
   frr = num_err / tot * 100
   print(' \tFRR : ' + str(round(frr, 4)) + ' %')

   num_err = 0
   tot = 0

   for i in range(len(tot_rec)):
      for j in range(0, 107):
         k = j + 1
         while(k <= 107):
            for l in range(len(tot_rec)):
               match_x = id.sift_match(tot_rec[i][j], tot_rec[l][k])
               if match_x: 
                  num_err += 1
               tot += 1
            k += 1
    
   print('\nFAR Calculation')  
   print('\tNum err : ' + str(num_err))          
   print(' \ttot : ' + str(tot))   
   far = num_err / tot * 100
   print('\tFAR : ' + str(round(far, 4)) + ' %')
