from iris import iris_class
from tools.file_manager import configuration
from identification import id_class
from data_classes.manage_dataset import CASIA_dataset
from tools import utils

'''
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
   
   utils.frr_far_sift(frr, far)
   utils.error_distribution_graph(frr, far)
   #utils.frr_far_sift_graph(frr_values, far_values, thresholds)
   '''
   
   
import matplotlib.pyplot as plt
import numpy as np

# Funzione per calcolare FRR e FAR in base alla soglia
def calculate_frr_far(soglia, tot_rec, id):
    num_err_frr = 0
    num_err_far = 0
    tot_frr = 0
    tot_far = 0

    # Calcolare FRR
    for i in range(0, 108):
      for j in range(len(tot_rec)):
         k = j + 1
         while k < len(tot_rec):
            match_x = id.sift_match2(tot_rec[j][i], tot_rec[k][i], soglia)
            if not match_x:
               num_err_frr += 1
            k += 1
            tot_frr += 1

    # Calcolare FAR
    for i in range(len(tot_rec)):
      for j in range(0, 107):
         k = j + 1
         while k <= 107:
               for l in range(len(tot_rec)):
                  match_x = id.sift_match2(tot_rec[i][j], tot_rec[l][k], soglia)
                  if match_x:
                        num_err_far += 1
                  tot_far += 1

    frr = num_err_frr / tot_frr * 100
    far = num_err_far / tot_far * 100
    return frr, far

# Funzione principale
if __name__ == '__main__':
   config = configuration()
   casia_dataset = CASIA_dataset(config)
   casia_dataset.load_dataset()
   rec_tot = casia_dataset.get_rec_tot2()
   irises = []

   print('\nLoading data...')

   for rec in rec_tot:
      for i in range(0, 108):
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

   soglie = np.arange(16, 32, 6)
   frr_values = []
   far_values = []

   for soglia in soglie:
      frr, far = calculate_frr_far(soglia, tot_rec, id)
      frr_values.append(frr)
      far_values.append(far)

   # Creazione del grafico
   plt.figure(figsize=(10, 6))
   plt.plot(soglie, frr_values, label='FRR', marker='o', color='red')
   plt.plot(soglie, far_values, label='FAR', marker='x', color='blue')
   plt.xlabel('Soglia')
   plt.ylabel('%')
   plt.title('FRR e FAR al variare della soglia')
   plt.legend()
   plt.grid(True)
   plt.show()
