import os
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
from data_classes.manage_dataset import CASIA_dataset
from tools import utils


def train_test():
   """
   Prepares training and testing datasets for model training and evaluation.

   :return: Scaled and formatted training and testing data with corresponding labels.
   :rtype: tuple (numpy.ndarray, list, numpy.ndarray, list)
   """
   casia_dataset = CASIA_dataset(config)
   casia_dataset.load_dataset()
   train = casia_dataset.get_data_rec()
   
   # Prepare training data
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

   # Prepare testing data
   test = casia_dataset.get_rec_test()
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
   
   print('\nTraining start')
   print('\n    Creating train set and test set...')
   X_train, y_train, X_test, y_test = train_test()
   
   # Standardize the datasets
   scaler = StandardScaler()
   scaler.fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   # Apply feature reduction using Locally Linear Embedding (LLE)
   n_neighbors = config.feature_reduction_lle.n_neighbors
   n_components = config.feature_reduction_lle.n_components
   lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
   lle.fit(X_train_scaled)
   X_train_red = lle.transform(X_train_scaled)
   X_test_red = lle.transform(X_test_scaled)
   utils.LLE_graph(X_train_red, y_train)

   # training knn
   print('\n    Training KNN...')
   knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
   knn.fit(X_train_red, y_train)
   y_pred_knn = knn.predict(X_test_red)

   # training svm
   print('\n    Training SVM...')
   svm = SVC(kernel='sigmoid')
   svm.fit(X_train_red, y_train)
   y_pred_svm = svm.predict(X_test_red)

   # training neural network
   print('\n    Training Neural Network...')
   input_size = X_train_red.shape[1]
   num_classes = 100
   model = iris_network(input_size, num_classes)
   nn = nn_classifier_class(model, config)
   nn.fit(X_train_red, y_train)
   y_pred_nn = nn.predict(X_test_red)

   # Calculate accuracy for each model
   accuracy_knn = accuracy_score(y_test, y_pred_knn)
   accuracy_svm = accuracy_score(y_test, y_pred_svm)
   accuracy_nn = accuracy_score(y_test, y_pred_nn)
   
   # Calculate merged accuracy for ensemble-like evaluation
   print('\nCalculating performance...')
   matched = 0
   for i in range(len(y_test)):
      if y_test[i] == y_pred_knn[i] or y_test[i] == y_pred_svm[i] or y_test[i] == y_pred_nn[i]:
         matched += 1

   merge_accuracy = matched / len(y_test)
   
   utils.accuracy_comparison(accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy)
   
   print("\nTEST performance...")
   print('    Accuracy KNN : ' + str(round(accuracy_knn, 2)))
   print('    Accuracy SVM : ' + str(round(accuracy_svm, 2)))
   print('    Accuracy NN : ' + str(round(accuracy_nn, 2))) 
   print('    Merge Accuracy : ' + str(round(merge_accuracy, 2)))
   print('\n')
   
   # Save the trained models and configurations
   if not directory_exists('checkpoints'):
      os.mkdir('checkpoints')
   
   joblib.dump(scaler, os.path.join('checkpoints', 'standard_scaler.pkl'))
   joblib.dump(lle, os.path.join('checkpoints', 'lle_feature_reduction.pkl'))
   joblib.dump(knn, os.path.join('checkpoints', 'knn_model.pkl'))
   joblib.dump(svm, os.path.join('checkpoints', 'svm_model.pkl'))
   joblib.dump(nn, os.path.join('checkpoints', 'nn_model.pkl'))
