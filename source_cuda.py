import cv2
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from numba import cuda, jit, njit


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

#----- Test CUDA -----
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("using", device, "device")

# ---------- Gaussian Blur image ----------
# Function to apply CUDA Gaussian blur
def apply_pytorch_cuda_gaussian_blur(image):
    
    # Create a Gaussian blur transformation with the desired parameters
    #transform = transforms.Compose([transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 1.0))])

    # Convert the image to a PyTorch tensor
    #image = Image.fromarray(image)
    #image = transform(image)
    
    # Convert the image to a PyTorch tensor and move it to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image_GPU = image.to(device)
    torch.cuda.synchronize()

    transform = transforms.Compose([transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 1.0))])
    blurred_image = transform(image_GPU)
    torch.cuda.synchronize()

    # Transfer the image back to NumPy
    blurred_image = image.cpu().numpy()

    # Apply the Gaussian blur using PyTorch
    #blurred_image = transform(image)

    # Transfer the blurred image back to the CPU
    #blurred_image = blurred_image.cpu().numpy()

    return blurred_image

def gaussian_test():
    # Convert the list of image tensors to a single tensor (batch)
    batch_images = torch.stack(image_files).to(device)

    # Apply transformations as a batch
    transform = transforms.Compose([transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 1.0))])

    # Perform asynchronous execution
    torch.cuda.synchronize()

    # Record the start time
    start_time = time.time()
    
    
    # Apply transformations to the entire batch
    blurred_images = transform(batch_images)
    
    # Wait for GPU operations to finish
    torch.cuda.synchronize()
    
    #print(f"Time for batch {i + 1}: {time.time() - start:.4f} seconds")

    # Transfer the batched blurred images back to NumPy if needed
    blurred_images = blurred_images.cpu().numpy()   

    # Resize the PyTorch CUDA Gaussian image for display
    resized_pytorch_cuda_image = cv2.resize(blurred_images, (desired_width, int(blurred_images.shape[0] * (desired_width / blurred_images.shape[1]))))

    # Output time taken for each image
    #print(f" {counter} - [{image_file}]")

    #counter += 1

    # Record the end time
    end_time = time.time()
    # Calculate the total time
    each_total_time = end_time - start_time

    # Output time taken for each image
    #print(f"[{image_file}] Total time taken: {each_total_time:.4f} seconds")

    # Save the blurred image to the PyTorch with CUDA output directory
    #output_path = os.path.join(gaussian_output_directory, f"pytorch_cuda_gaussianblur_{image_file}")
    #cv2.imwrite(output_path, resized_pytorch_cuda_image)

    # Assuming blurred_images is your batched result
    for i, blurred_image in enumerate(blurred_images):
        output_path = f"gaussian_blurred_image_{i}.jpg"  # Replace with your desired output path
        cv2.imwrite(output_path, blurred_image)

def gaussian_blur_image():

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

        # Apply PyTorch Gaussian blur with CUDA
        pytorch_cuda_blurred_image = apply_pytorch_cuda_gaussian_blur(image)

        # Resize the original image for display
        resized_image = cv2.resize(image, (desired_width, int(image.shape[0] * (desired_width / image.shape[1]))))
        # Resize the PyTorch CUDA Gaussian image for display
        resized_pytorch_cuda_image = cv2.resize(pytorch_cuda_blurred_image, (desired_width, int(pytorch_cuda_blurred_image.shape[0] * (desired_width / pytorch_cuda_blurred_image.shape[1]))))

        # Output time taken for each image
        print(f" {counter} - [{image_file}]")

        counter += 1

        # Record the end time
        end_time = time.time()
        # Calculate the total time
        each_total_time = end_time - start_time

        # Output time taken for each image
        print(f"[{image_file}] Total time taken: {each_total_time:.4f} seconds")

        # Save the blurred image to the PyTorch with CUDA output directory
        output_path = os.path.join(gaussian_output_directory, f"pytorch_cuda_gaussianblur_{image_file}")
        cv2.imwrite(output_path, resized_pytorch_cuda_image)

# ---------- Gaussian Blur image ----------


# ---------- Bilateral Filter image ----------
def gaussian(x,sigma):
    return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))

def distance(x1,y1,x2,y2):
    return np.sqrt(np.abs((x1-x2)**2-(y1-y2)**2))

def bilateral_filter_image():

    #counter
    counter = 1

    for image_file in image_files:
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
        # Resize the convolution image for display
        resized_bilateral_image = cv2.resize(blurred_image_output, (desired_width, int(blurred_image_output.shape[0] * (desired_width / blurred_image_output.shape[1]))))

        # Record the end time
        end_time = time.time()
        # Calculate the total time
        each_total_time = end_time - start_time
        
        # Output time taken for each image
        print(f" {counter} - [{ image_file }] Total time taken: {each_total_time:.4f} seconds")

        counter = counter + 1

        # Save the blurred image to the output directory
        output_path = os.path.join(bilateral_output_directory, f"bilateral_{image_file}")
        cv2.imwrite(output_path, resized_bilateral_image)

        # Close the windows
        # cv2.destroyAllWindows()

# ---------- Bilateral Filter image ----------


# Record the start time
total_start_time = time.time()

#----- Gaussian Filter -----
gaussian_blur_image()

#----- Bilateral Filter -----
#bilateral_filter_image()


# Record the end time
total_end_time = time.time()


# Calculate the total time
total_time = total_end_time - total_start_time
print(f"\nTotal time taken: {total_time:.4f} seconds")