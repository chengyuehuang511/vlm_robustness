
import numpy as np 
perf = [62.64, 55.0, 59.33, 63.50]
#LoRA, LP, LP-FT, FTP, DiGraP

#range
joint_shift = [16.17, 18.92, 17.2, 16.58] 

image_shift = [7.99,6.88, 7.49, 8.49]


joint_correlation_matrix = np.corrcoef(perf, joint_shift)
joint_correlation = joint_correlation_matrix[0, 1]  # Extract correlation coefficient

print("Joint perf Correlation:", joint_correlation)


image_correlation_matrix = np.corrcoef(perf, image_shift)
image_correlation = image_correlation_matrix[0, 1]  # Extract correlation coefficient

print("Image perf Correlation:", image_correlation)












