from iris import iris_class
from tools.file_manager import configuration, load_dataset
from identification import id_class


def load_irises(dataset):
   """
   Processes a dataset of eye images to extract iris features.

   Parameters:
   - dataset: A collection of eye images grouped by records.

   Returns:
   - irises: A nested list where each record contains a list of iris_class objects for each subject.
   """
   irises = []

   for rec in dataset:
      subjects = []
      for subject in range(0, 108):
         eye = rec[subject]
         iris_obtained = iris_class(eye, subject, config)
         iris_obtained.segmentation()
         iris_obtained.feature_extraction() 
         subjects.append(iris_obtained)  
      irises.append(subjects)
         
   return irises


def calculate_frr_far(irises, id, threshold):
   """
   Calculates the False Rejection Rate (FRR) and False Acceptance Rate (FAR).

   Parameters:
   - irises: A nested list of iris_class objects.
   - id: An id_class object used for iris matching.
   - threshold: A similarity threshold for matching iris samples.

   Returns:
   - frr: The False Rejection Rate (percentage).
   - far: The False Acceptance Rate (percentage).
   """
   num_err_frr = 0
   num_err_far = 0
   tot_frr = 0
   tot_far = 0

   for i in range(0, 108):
      for j in range(len(irises)):
         k = j + 1
         while(k < len(irises)):
            match_x = id.sift_match(irises[j][i], irises[k][i], threshold)
            if not match_x:
               num_err_frr += 1
            k += 1
            tot_frr += 1
    
   for i in range(len(irises)):
      for j in range(0, 107):
         k = j + 1
         while(k <= 107):
            for l in range(len(irises)):
               match_x = id.sift_match(irises[i][j], irises[l][k], threshold)
               if match_x:
                  num_err_far += 1
               tot_far += 1
            k += 1
    
   frr = num_err_frr / tot_frr * 100
   far = num_err_far / tot_far * 100
   return round(frr, 2), round(far, 2)


if __name__ == '__main__':
   """
    Main execution block to process a dataset, calculate FRR and FAR, and display the results.
   """
   print('\nLoading irises...')
   config = configuration()
   casia_dataset = load_dataset(config)
   irises = load_irises(casia_dataset)

   print(' \nStart Matching... \n')

   id = id_class(config, None)
   frr, far = calculate_frr_far(irises, id, None)
   print(" FAR : " + str(far) + " %")
   print(" FRR : " + str(frr) + " %")
   