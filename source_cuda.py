import cv2
import numpy as np
import os
import time

def gkernel(l=3, sig=2):

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)


# Directory containing the original images
image_directory = "D:/Bachelor of Software Engineering (TARUC)/Bachelor of SE (7th Sem) Andrewhew/Distributed Systems and Parallel Computing/Assignment/image/sharp/100_images/"

# Directory to save the blurred images
output_directory = os.path.join(image_directory, "D:/output_directory/") 
output_directory = "D:/Bachelor of Software Engineering (TARUC)/Bachelor of SE (7th Sem) Andrewhew/Distributed Systems and Parallel Computing/Assignment/image/sharp/100_images/filter_image/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List all files in the directory
image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Resize parameters
desired_width = 500

# ---------- Gaussian Blur image ----------
def gaussian_blur_image():
    # Initialize CUDA
    cv2.cuda.setDevice(0)  # Set the GPU device index you want to use (0 for the first GPU)

    # Counter
    counter = 1

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not open or find the image: {image_path}")
            continue

        # Record the start time
        start_time = time.time()

        # Convert the image to a CUDA Mat object
        cuda_image = cv2.cuda_GpuMat()
        cuda_image.upload(image)

        # Apply Gaussian blur on the GPU
        kernel_size = 5  # Must be an odd number
        sigma = 1.0
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.multiply(kx, np.transpose(ky))
        cuda_kernel = cv2.cuda_GpuMat()
        cuda_kernel.upload(kernel)

        blurred_image_output = cv2.cuda.filter2D(cuda_image, -1, cuda_kernel)

        # Download the result back to the CPU
        blurred_image_output = blurred_image_output.download()

        # Resize the original image for display
        resized_image = cv2.resize(image, (desired_width, int(image.shape[0] * (desired_width / image.shape[1]))))
        # Resize the Gaussian image for display
        resized_gaussian_image = cv2.resize(blurred_image_output, (desired_width, int(blurred_image_output.shape[0] * (desired_width / blurred_image_output.shape[1]))))

        # Record the end time
        end_time = time.time()
        # Calculate the total time
        each_total_time = end_time - start_time

        # Output time taken for each image
        print(f" {counter} - [{ image_file }] Total time taken: {each_total_time:.4f} seconds")

        counter = counter + 1

        # Save the blurred image to the output directory
        output_path = os.path.join(output_directory, f"gaussianblur_{image_file}")
        cv2.imwrite(output_path, resized_gaussian_image)

# ---------- Gaussian Blur image ----------


# ---------- Bilateral Filter image ----------
def bilateral_filter_image():
    # Initialize CUDA
    cv2.cuda.setDevice(0)  # Set the GPU device index you want to use (0 for the first GPU)

    # Counter
    counter = 1

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not open or find the image: {image_path}")
            continue

        # Record the start time
        start_time = time.time()

        # Convert the image to a CUDA Mat object
        cuda_image = cv2.cuda_GpuMat()
        cuda_image.upload(image)

        # Apply bilateral filter on the GPU
        diameter = 9     # Diameter of each pixel neighborhood to consider
        sigma_i = 75     # Intensity similarity weight
        sigma_s = 75     # Spatial distance weight
        cuda_blurred_image_output = cv2.cuda.bilateralFilter(cuda_image, diameter, sigma_i, sigma_s)

        # Download the result back to the CPU
        cuda_blurred_image_output = cuda_blurred_image_output.download()

        # Resize the original image for display
        resized_image = cv2.resize(image, (desired_width, int(image.shape[0] * (desired_width / image.shape[1]))))
        # Resize the bilateral image for display
        resized_bilateral_image = cv2.resize(cuda_blurred_image_output, (desired_width, int(cuda_blurred_image_output.shape[0] * (desired_width / cuda_blurred_image_output.shape[1]))))

        # Record the end time
        end_time = time.time()
        # Calculate the total time
        each_total_time = end_time - start_time

        # Output time taken for each image
        print(f" {counter} - [{ image_file }] Total time taken: {each_total_time:.4f} seconds")

        counter = counter + 1

        # Save the blurred image to the output directory
        output_path = os.path.join(output_directory, f"bilateral_{image_file}")
        cv2.imwrite(output_path, resized_bilateral_image)

# ---------- Bilateral Filter image ----------

# Record the start time
total_start_time = time.time()

gaussian_blur_image()
bilateral_filter_image()

# Record the end time
total_end_time = time.time()


# Calculate the total time
total_time = total_end_time - total_start_time
print(f"\nTotal time taken: {total_time:.4f} seconds")

