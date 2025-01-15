import os
import pickle
import cv2 as cv
from iris import iris_class
from tools.file_manager import configuration
from identification import id_class

if  __name__ == '__main__':  
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

    
    rec_test = [rec_3, rec_4, rec_7]
    irises = []

    print('Caricamento dati')

    for rec in rec_test:
      for i in range (0, 100):
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         iris_obtained.segmentation()
         iris_obtained.feature_extraction()
         new_image = iris_obtained.get_keypoints_image()
         iris_obtained.set_iris_code()
         irises.append(iris_obtained)
         path = os.path.join('result', str(i)+'.jpeg')
         cv.imwrite(path, new_image)

    rec_tot = [rec_3, rec_4, rec_7, rec_1, rec_2, rec_5, rec_6]
    for rec in rec_tot:
      for i in range (100, 108):
         eye = rec[i]
         iris_obtained = iris_class(eye, i, config)
         iris_obtained.segmentation()
         iris_obtained.feature_extraction()
         new_image = iris_obtained.get_keypoints_image()
         iris_obtained.set_iris_code()
         irises.append(iris_obtained)

    data_rec = [rec_1, rec_2, rec_5, rec_6] 
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
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tot = 0
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

    print('TP ' + str(tp))
    print('FP ' + str(fp))
    print('TN ' + str(tn))
    print('FN ' + str(fn))

    print('tot ' + str(tot))
    accuracy = (tp + tn) / tot * 100
    far = fp / (fp + tn) * 100
    frr = fn / (fn + tp) * 100
    
    print('accuracy ' + str(round(accuracy, 2)) + " %")
    print('FAR ' + str(round(far, 2)) + " %")
    print('FRR ' + str(round(frr, 2)) + " %")
           
         
        


    


    '''

    print('Calcolo')


    rec_1 = irises[:108]
    rec_2 = irises[108:216]
    rec_3 = irises[216:324]

    tot_rec = [rec_1, rec_2, rec_3]
    num_err = 0
    tot = 0


    for i in range(0, 108):
        for j in range(len(tot_rec) - 1):
           k = j + 1
           while(k < len(tot_rec)):
               match_x = matcher.match(tot_rec[j][i], tot_rec[k][i])
               if not match_x:
                 num_err += 1
           k += 1
           tot += 1
    
     
          
    print(' Num err : ' + str(num_err))          
    print(' tot : ' + str(tot))   
    frr = num_err / tot * 100
    print(' FRR : ' + str(round(frr, 4)) + ' %')


    num_err = 0
    tot = 0


    for i in range(len(tot_rec)):
       for j in range(0, 107):
          k = j + 1
          while(k <= 107):
             for l in range(len(tot_rec)):
               match_x = matcher.match(tot_rec[i][j], tot_rec[l][k])
               if match_x: 
                  num_err += 1
               tot += 1
          k += 1
    
    print(' Num err : ' + str(num_err))          
    print(' tot : ' + str(tot))   
    far = num_err / tot * 100
    print(' FAR : ' + str(round(far, 4)) + ' %')

    '''


    