from build.cuda_hist_eq import histogram_equalization

import cv2
import numpy as np
import time

input_image_path = "../beach.png"
output_image_path = "python_equalized_image.png"

original_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
equalized_image = np.zeros_like(original_image)
width = original_image.shape[0]
height = original_image.shape[1]

# Start CPU timing
start_time = time.time()
equalized_image_cpu = cv2.equalizeHist(original_image)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"CPU Elapsed time: {elapsed_time} seconds")

# Start GPU timing
start_time = time.time()
histogram_equalization(original_image, equalized_image, width, height)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"GPU Elapsed time: {elapsed_time} seconds")



cv2.imwrite("gpu_Equalized_beach.png", equalized_image)
cv2.imwrite("cpu_Equalized_beach.png", equalized_image_cpu)