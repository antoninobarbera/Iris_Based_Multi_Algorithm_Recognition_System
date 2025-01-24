import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def distance(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def point_in_circle(centre, radius, point):
    dist = distance(centre, point) 
    return dist <= radius

def draw_keypoints_image(image_iris, keypoints, centre, iris_radius):
    red = (0, 0, 255)
    blue = (255, 0, 0)
    keypoints_image = cv.cvtColor(image_iris.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(keypoints_image, centre, 2, red, thickness=3)
    cv.circle(keypoints_image, centre, iris_radius + 3, red, thickness=3)
    keypoints_image = cv.drawKeypoints(keypoints_image, keypoints, color=blue, flags=0, outImage=None)
    return keypoints_image

def angle(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
    angle_360 = (angle + 360) % 360
    return angle_360

def to_polar(point, pole=(0, 0)):
    r = distance(point, pole)
    theta = angle(point, pole)
    return r, theta

def normalize_r(r, pupil_radius, iris_radius):
    range = iris_radius - pupil_radius
    r_norm = (r - pupil_radius) / range
    return r_norm

def is_within_one_std(value, centre, dev):
    not_in_left_tail = value > centre - dev
    not_in_right_tail = value < centre + dev
    is_on_range = not_in_left_tail and not_in_right_tail
    return is_on_range

def warp_polar(image, x_size, y_size, centre, pupil_radius, iris_radius):
    normalized_iris=np.zeros(shape=(x_size, y_size))
    x_c, y_c = centre
        
    angle= 2.0 * math.pi / y_size
    inner_boundary_x = np.zeros(shape=(1, y_size))
    inner_boundary_y = np.zeros(shape=(1, y_size))
    outer_boundary_x = np.zeros(shape=(1, y_size))
    outer_boundary_y = np.zeros(shape=(1, y_size))

    for i in range(y_size):
        inner_boundary_x[0][i]= x_c + pupil_radius * math.cos(angle*(i))
        inner_boundary_y[0][i]= y_c + pupil_radius * math.sin(angle*(i))
        outer_boundary_x[0][i]= x_c + iris_radius * math.cos(angle*(i))
        outer_boundary_y[0][i]= y_c + iris_radius * math.sin(angle*(i))
        
    for j in range (y_size):
       for i in range (x_size):
            normalized_iris[i][j]= image[min(int(int(inner_boundary_y[0][j]) + (int(outer_boundary_y[0][j]) - int(inner_boundary_y[0][j])) * (i/64.0)),
                                        image.shape[0] - 1)][min(int(int(inner_boundary_x[0][j]) + (int(outer_boundary_x[0][j]) - int(inner_boundary_x[0][j])) * (i/64.0)),
                                        image.shape[1] - 1)]
            
    return normalized_iris

def gabor_filter(theta, config):
    ksize = (config.gabor_filter.x_size, config.gabor_filter.y_size)
    kernel = cv.getGaborKernel(ksize, 
                                config.gabor_filter.gamma, 
                                theta,
                                config.gabor_filter.frequency,
                                config.gabor_filter.sigma,
                                config.gabor_filter.psi, 
                                ktype=cv.CV_64F)
    return kernel

def encoding(image, config):
    vector = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
        gabor = gabor_filter(theta, config)
        filtered_eye = cv.filter2D(image, cv.CV_64F, gabor)
        for i in range(0, image.shape[0], 8):
            for j in range(0, image.shape[1], 8):                
                patch = filtered_eye[i:i+8, j:j+8]
                mean = patch.mean()
                AAD = np.abs(patch - mean).mean()
                vector.extend([mean, AAD]) 
    return np.array(vector)

def manage_best_model_and_metrics(model, evaluation_metric, metrics, best_metric, best_model, lower_is_better):
    if lower_is_better:
        is_best = metrics[evaluation_metric] < best_metric
    else:
        is_best = metrics[evaluation_metric] > best_metric
        
    if is_best:
        best_metric = metrics[evaluation_metric]
        best_model = model

    return best_model, best_metric

def accuracy_comparison(accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy):
    """
    Plots a bar chart comparing the accuracy of KNN, SVM, Neural Network models, and Merge Accuracy, and saves it to the 'graph' folder.

    :param accuracy_knn: Accuracy of the KNN model.
    :type accuracy_knn: float
    :param accuracy_svm: Accuracy of the SVM model.
    :type accuracy_svm: float
    :param accuracy_nn: Accuracy of the Neural Network model.
    :type accuracy_nn: float
    :param merge_accuracy: Combined accuracy of the models.
    :type merge_accuracy: float
    """
    models = ['KNN', 'SVM', 'Neural Network', 'Merge Accuracy']
    accuracies = [accuracy_knn, accuracy_svm, accuracy_nn, merge_accuracy]
    os.makedirs('graph', exist_ok=True)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'purple'], alpha=0.8, width=0.6)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                f'{height:.2%}', ha='center', va='bottom', fontsize=12)

    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Comparison of Model Accuracies', fontsize=16)

    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join('graph', 'model_accuracy_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    
def LLE_graph(X_train_red, y_train, save_path='graph/'):
    """
    Visualizes and saves the reduced features using LLE (Locally Linear Embedding).

    :param X_train_red: Reduced feature data.
    :param y_train: Labels associated with the data points.
    :param save_path: Directory path to save the plot (default is 'graph/').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_red[:, 0], X_train_red[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
    
    plt.colorbar(scatter, label='Label')
    
    plt.title('Reduced Features Using LLE')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    
    plot_filename = os.path.join(save_path, 'reduced_features_lle.png')
    plt.savefig(plot_filename)
    plt.close() 
    
def identification_performance(tp, fp, tn, fn, save_path='graph/'):
    """
    Visualizes the performance of the identification system using a bar chart.

    :param tp: True Positives count.
    :param fp: False Positives count.
    :param tn: True Negatives count.
    :param fn: False Negatives count.
    :param save_path: Directory path to save the plot (default is 'graph/').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    values = [tp, fp, tn, fn]
    
    plt.figure(figsize=(9, 6))
    plt.bar(categories, values, color=['green', 'red', 'blue', 'orange'])

    plt.title('Identification System Performance')
    plt.xlabel('Categories')
    plt.ylabel('Count')

    plot_filename = os.path.join(save_path, 'identification_performance.png')
    plt.savefig(plot_filename)
    plt.close()

def frr_far_sift(frr, far, save_path='graph/'):
    """
    Visualizes the FRR and FAR metrics using bar charts.

    :param frr: False Rejection Rate.
    :param far: False Acceptance Rate.
    :param save_path: Directory path to save the plots (default is 'graph/').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    metrics = ['FRR', 'FAR']
    values = [frr, far]
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['red', 'blue'])

    plt.title('FRR and FAR of SIFT algorithm')
    plt.xlabel('Metrics')
    plt.ylabel('Percentage (%)')

    plot_filename = os.path.join(save_path, 'frr_far_calculation.png')
    plt.savefig(plot_filename)
    plt.close()

def error_distribution_graph(frr, far, save_path='graph/'):
    """
    Visualizes the distribution of FRR and FAR using a pie chart.

    :param frr: False Rejection Rate.
    :param far: False Acceptance Rate.
    :param save_path: Directory path to save the plot (default is 'graph/').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    labels = ['FRR', 'FAR']
    sizes = [frr, far]
    colors = ['red', 'blue']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Error Distribution (FRR vs FAR)')

    plot_filename = os.path.join(save_path, 'error_distribution.png')
    plt.savefig(plot_filename)
    plt.close()
    
def frr_far_threshold(id_class_instance, irises, thresholds):
    """
    Plots FRR and FAR curves as a function of thresholds.

    :param id_class_instance: Instance of the identification class.
    :type id_class_instance: id_class
    :param irises: List of processed iris objects to evaluate.
    :type irises: list
    :param thresholds: List of thresholds to evaluate.
    :type thresholds: list or numpy.ndarray
    """
    frr_values = []
    far_values = []

    for threshold in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0

        # Set the threshold in the identification class
        id_class_instance.set_threshold(threshold)

        # Evaluate the performance for the current threshold
        for iris in irises:
            flag, label = id_class_instance.identification(iris)
            
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
    plt.title("FRR and FAR vs Threshold", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.7)
    plt.tight_layout()
    plt.show()

def iris_code_plot(iris_code, path):
    plt.plot(iris_code)
    plt.title(" IRIS CODE PLOT ")
    plt.xlabel("Index")
    plt.ylabel("Value")    
    plt.savefig(path, format="jpeg", dpi=300)
    plt.close()

def ROC_curve(far, frr, thresholds, path):
    plt.figure()
    plt.plot(far, frr, marker='o', label="ROC Curve")
    for i, threshold in enumerate(thresholds):
       plt.text(far[i], frr[i], f"Threshold:{int(threshold)}", fontsize=9)
    plt.xlabel("FAR %")
    plt.ylabel("FRR %")
    plt.title("ROC CURVE")
    plt.grid()
    plt.legend()
    plt.savefig(path, format="jpeg", dpi=300)
    plt.close()
    
def frr_far_sift_graph(frr_values, far_values, thresholds, save_path='graph/'):
    """
    Plots and saves the FRR and FAR against the threshold.

    :param frr_values: List of FRR values.
    :param far_values: List of FAR values.
    :param thresholds: List of threshold values.
    :param save_path: Directory path to save the plot (default is 'graph/').
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 6))

    # Plot FRR and FAR against threshold
    plt.plot(thresholds, frr_values, label='FRR', color='red', marker='o')
    plt.plot(thresholds, far_values, label='FAR', color='blue', marker='o')

    # Add labels and title
    plt.title('FRR and FAR vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Percentage (%)')

    # Show legend
    plt.legend()

    # Save the plot
    plot_filename = os.path.join(save_path, 'frr_far_vs_threshold.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot