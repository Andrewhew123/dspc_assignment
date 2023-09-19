import cv2
import numpy as np
import os
import time
import threading
import matplotlib.pyplot as plt


# Number of images to process in each thread
images_per_thread = 1

# Directory containing the original images
image_directory = "D:/Bachelor of Software Engineering (TARUC)/Bachelor of SE (7th Sem) Andrewhew/Distributed Systems and Parallel Computing/Assignment/image/sharp/100_images/"

# Directory to save the blurred images using gaussian filter
gaussian_output_directory = os.path.join(image_directory, "D:/output_directory/") 
gaussian_output_directory = image_directory + "gaussian_filter_image/"

# Directory to save the blurred images using bilateral
bilateral_output_directory = os.path.join(image_directory, "D:/output_directory/") 
bilateral_output_directory = image_directory + "bilateral_filter_image/"

# Create the gaussian output directory if it doesn't exist
if not os.path.exists(gaussian_output_directory):
    os.makedirs(gaussian_output_directory)

# Create the bilateral output directory if it doesn't exist
if not os.path.exists(bilateral_output_directory):
    os.makedirs(bilateral_output_directory)

# List all files in the directory
image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Resize parameters
desired_width = 500


# ---------- Gaussian Blur image ----------
def gaussian_blur_image(start_index, end_index):

    for i in range(start_index, end_index):
        image_file = image_files[i]
        image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not open or find the image: {image_path}")
            continue

        # Record the start time
        start_time = time.time()

        # Apply Gaussian blur
        kernel_size = 5  # Must be an odd number
        sigma = 1.0
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.multiply(kx, np.transpose(ky))

        blurred_image_output = cv2.filter2D(image, -1, kernel)

        # Resize the original image for display
        resized_image = cv2.resize(image, (desired_width, int(image.shape[0] * (desired_width / image.shape[1]))))
        # Resize the Gaussian image for display
        resized_gaussian_image = cv2.resize(blurred_image_output, (desired_width, int(blurred_image_output.shape[0] * (desired_width / blurred_image_output.shape[1]))))

        # Record the end time
        end_time = time.time()
        # Calculate the total time
        each_total_time = end_time - start_time

        # Output time taken for each image
        print(f"{i + 1} - [{image_file}] Total time taken: {each_total_time:.4f} seconds")

        # Save the blurred image to the output directory
        output_path = os.path.join(gaussian_output_directory, f"gaussianblur_{image_file}")
        cv2.imwrite(output_path, resized_gaussian_image)

        # Close the windows
        # cv2.destroyAllWindows()

# ---------- Gaussian Blur image ----------


# ---------- Bilateral Filter image ----------
def gaussian(x,sigma):
    return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return np.sqrt(np.abs((x1-x2)**2-(y1-y2)**2))

def bilateral_filter_image(start_index, end_index):

    for i in range(start_index, end_index):
        image_file = image_files[i]
        image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not open or find the image: {image_path}")
            continue

        # Record the start time
        start_time = time.time()

        # Apply bilateral filter
        diameter = 9     # Diameter of each pixel neighborhood to consider
        sigma_i = 75     # Intensity similarity weight
        sigma_s = 75     # Spatial distance weight
        blurred_image_output = cv2.bilateralFilter(image, diameter, sigma_i, sigma_s)

        # Resize the original image for display
        resized_image = cv2.resize(image, (desired_width, int(image.shape[0] * (desired_width / image.shape[1]))))
        # Resize the bilateral image for display
        resized_bilateral_image = cv2.resize(blurred_image_output, (desired_width, int(blurred_image_output.shape[0] * (desired_width / blurred_image_output.shape[1]))))

        # Record the end time
        end_time = time.time()
        # Calculate the total time
        each_total_time = end_time - start_time
        
        # Output time taken for each image
        print(f"{i + 1} - [{image_file}] Total time taken: {each_total_time:.4f} seconds")

        # Save the blurred image to the output directory
        output_path = os.path.join(bilateral_output_directory, f"bilateral_{image_file}")
        cv2.imwrite(output_path, resized_bilateral_image)

        # Close the windows
        # cv2.destroyAllWindows()

# ---------- Bilateral Filter image ----------



# ---------- Apply multithreading (Ver 2) ----------

# Record the start time
total_start_time = time.time()

# Split the image processing among multiple threads
num_threads = (len(image_files) + images_per_thread - 1) // images_per_thread

threads = []
for i in range(num_threads):
    start_index = i * images_per_thread
    end_index = min((i + 1) * images_per_thread, len(image_files))
    #----- Gaussian Filter -----
    thread = threading.Thread(target=gaussian_blur_image, args=(start_index, end_index))
    #----- Bilateral Filter -----
    #thread = threading.Thread(target=bilateral_filter_image, args=(start_index, end_index))
    threads.append(thread)

# Start the threads
for thread in threads:
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All threads have finished processing.")

# Record the end time
total_end_time = time.time()

# Calculate the total time
total_time = total_end_time - total_start_time
print(f"\nTotal time taken: {total_time:.4f} seconds")

# ---------- Apply multithreading (Ver 2) ----------


# ---------- Apply multithreading ----------
"""
if __name__ == '__main__':
    start = time.perf_counter()

    # Create two separate processes
    p1 = threading.Thread(target=gaussian_blur_image)
    p2 = threading.Thread(target=bilateral_filter_image)

    # Start the processes
    p1.start()
    p2.start()

    # Wait for both processes to finish
    p1.join()
    p2.join()

    finish = time.perf_counter()
    print("Finished running after seconds: ", finish - start)
"""
# ---------- Apply multithreading ----------




