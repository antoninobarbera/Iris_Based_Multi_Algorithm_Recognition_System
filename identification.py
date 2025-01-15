import cv2 as cv
import joblib
import os
from tools.matching_score import matching_score_class

class id_class:

    __slots__ = ['config', 'scaler', 'feature_reduction', 'classifier_1', 'classifier_2', 'classifier_3', 'data_dict']

    def __init__(self, config, data_dict):
        scaler_path = os.path.join('checkpoints', 'standard_scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        feature_reduction_path = os.path.join('checkpoints', 'lle_feature_reduction.pkl')
        self.feature_reduction = joblib.load(feature_reduction_path)
        classifier_1_path = os.path.join('checkpoints', 'knn_model.pkl')
        self.classifier_1 = joblib.load(classifier_1_path)
        classifier_2_path = os.path.join('checkpoints', 'svm_model.pkl')
        self.classifier_2 = joblib.load(classifier_2_path)
        classifier_3_path = os.path.join('checkpoints', 'nn_model.pkl')
        self.classifier_3 = joblib.load(classifier_3_path)
        self.config = config
        self.data_dict = data_dict

    
    def sift_match(self, iris_1, iris_2):
        bf = cv.BFMatcher()
        kp_1 = iris_1.get_keypoints()
        des_1 = iris_1.get_descriptors()
        kp_2 = iris_2.get_keypoints()
        des_2 = iris_2.get_descriptors()
        if not kp_1 or not kp_2: 
            return False
        matches = bf.knnMatch(des_1, des_2, k=2)
        matching_score = matching_score_class(iris_1, iris_2, self.config)
        for m, n in matches:
            if (m.distance / n.distance) > self.config.matching.lowe_filter:
                continue
            x_1, y_1 = kp_1[m.queryIdx].pt
            x_2, y_2 = kp_2[m.trainIdx].pt
            p_1 = (x_1, y_1)
            p_2 = (x_2, y_2) 
            matching_score.__add__(p_1, p_2)
        score = matching_score()
        if score > self.config.matching.threshold:
            flag = True
        else:
            flag = False
        return flag
    
    def identification(self, iris):
       iris_code = [iris.get_iris_code()]
       iris_code_scaled = self.scaler.transform(iris_code)
       iris_code_red = self.feature_reduction.transform(iris_code_scaled)
       
       # CLASSIFIER 1 MATCHING
       label_1 = self.classifier_1.predict(iris_code_red)       
       # CLASSIFIER 2 MATCHING
       label_2 = self.classifier_2.predict(iris_code_red)      
       # CLASSIFIER 3 MATCHING
       label_3 = self.classifier_3.predict(iris_code_red)

       possible = [label_1[0], label_2[0], label_3[0]]
       possible = list(set(possible))

       result = []
       for label in possible:
           for possible_iris in self.data_dict[label]:
              if self.sift_match(iris, possible_iris):
                  result.append(label)
                  break
       result = list(set(result))

       if len(result) == 1:
           return True, result[0]
       else:
           return False, None
              
        

        




       

       flag = classifier_1_match and classifier_2_match and classifier_3_match
       return flag
    

    def match(self, iris_1, iris_2):
       if self.sift_match(iris_1, iris_2):
           return True
       else:
           return self.ml_match(iris_1, iris_2)
      
           