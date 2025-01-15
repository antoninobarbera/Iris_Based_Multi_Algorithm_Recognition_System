import os
import pickle
import joblib
import numpy as np
from iris import iris_class
from sklearn.svm import SVC
from nn_model.nn_iris import iris_network
from sklearn.metrics import accuracy_score
from tools.file_manager import configuration, directory_exists
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding
from nn_model.nn_iris_model import nn_classifier_class



def train_test():
   with open('CASIA.pkl', 'rb') as file:
       casia = pickle.load(file)

   rec_1 = casia['rec_1']
   rec_2 = casia['rec_2']
   rec_3 = casia['rec_3']
   rec_4 = casia['rec_4']
   rec_5 = casia['rec_5']
   rec_6 = casia['rec_6']
   rec_7 = casia['rec_7']

   train = [rec_1, rec_2 ,rec_5, rec_6]
   X_train_temp = []
   y_train = []
   for rec in train:
      for i in range (0, 100):
         eye = rec[i]
         iris_obtained = iris_class(eye, None, config)
         iris_obtained.segmentation()
         code = iris_obtained.set_iris_code()
         code = iris_obtained.get_iris_code()
         X_train_temp.append(code)
         y_train.append(i)

   test = [rec_3 , rec_4 ,rec_7]
   X_test_temp = []
   y_test = []
   for rec in test:
      for i in range (0, 100):
         eye = rec[i]
         iris_obtained = iris_class(eye, None, config)
         iris_obtained.segmentation()
         iris_obtained.set_iris_code()
         code = iris_obtained.get_iris_code()
         X_test_temp.append(code)
         y_test.append(i)

   X_train = np.vstack(X_train_temp)
   X_test = np.vstack(X_test_temp)
   return X_train, y_train, X_test, y_test


if  __name__ == '__main__':       
   config = configuration()
   X_train, y_train, X_test, y_test = train_test()
   
   # scaler
   scaler = StandardScaler()
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # feature reduction
   n_neighbors = config.feature_reduction_lle.n_neighbors
   n_components = config.feature_reduction_lle.n_components
   lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
   lle.fit(X_train_scaled)
   X_train_red = lle.transform(X_train_scaled)
   X_test_red = lle.transform(X_test_scaled)

   # training knn
   knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
   knn.fit(X_train_red, y_train)
   y_pred_knn = knn.predict(X_test_red)

   # training svm
   svm = SVC(kernel='sigmoid')
   svm.fit(X_train_red, y_train)
   y_pred_svm = svm.predict(X_test_red)

   # training neural network
   input_size = X_train_red.shape[1]
   num_classes = 100
   model = iris_network(input_size, num_classes)
   nn = nn_classifier_class(model, config)
   nn.fit(X_train_red, y_train)
   y_pred_nn = nn.predict(X_test_red)

   # test
   accuracy_knn = accuracy_score(y_test, y_pred_knn)
   accuracy_svm = accuracy_score(y_test, y_pred_svm)
   accuracy_nn = accuracy_score(y_test, y_pred_nn)
   
   matched = 0
   for i in range(len(y_test)):
      if y_test[i] == y_pred_knn[i] or y_test[i] == y_pred_svm[i] or y_test[i] == y_pred_nn[i]:
         matched += 1

   merge_accuracy = matched / len(y_test)
   
   print(" TEST")
   print('Accuracy KNN : ' + str(round(accuracy_knn, 2)))
   print('Accuracy SVM : ' + str(round(accuracy_svm, 2)))
   print('Accuracy NN : ' + str(round(accuracy_nn, 2))) 
   print('Merge Accuracy : ' + str(round(merge_accuracy, 2)))


   # save models
   if not directory_exists('checkpoints'):
      os.mkdir('checkpoints')
   
   joblib.dump(scaler, os.path.join('checkpoints', 'standard_scaler.pkl'))
   joblib.dump(lle, os.path.join('checkpoints', 'lle_feature_reduction.pkl'))
   joblib.dump(knn, os.path.join('checkpoints', 'knn_model.pkl'))
   joblib.dump(svm, os.path.join('checkpoints', 'svm_model.pkl'))
   joblib.dump(nn, os.path.join('checkpoints', 'nn_model.pkl'))

   

    















