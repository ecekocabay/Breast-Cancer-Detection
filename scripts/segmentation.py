import os
import numpy as np
import cv2
from collections import deque

# Step 1: Contrast Enhancement (CLAHE)
def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

# Step 3: Dilation
def apply_dilation(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=50)

# Step 5: Region Growing Segmentation
def region_growing(image, seed_point, threshold=10):
    h, w = image.shape
    segmented_image = np.zeros((h, w), dtype=np.uint8)
    queue = deque([seed_point])
    seed_intensity = image[seed_point[1], seed_point[0]]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        x, y = queue.popleft()
        if segmented_image[y, x] == 0:
            segmented_image[y, x] = 255
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if segmented_image[ny, nx] == 0 and abs(int(image[ny, nx]) - int(seed_intensity)) <= threshold:
                        queue.append((nx, ny))
    return segmented_image

# Step 7: Crop Segmented Region
def crop_segmented_region(original_image, segmented_mask):
    contours, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return original_image[y:y + h, x:x + w]
    return None

# Step 8: Save Combined Results
def save_combined_results(output_path, images, size=(300, 300)):
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        print(f"No valid images to save for {output_path}")
        return

    resized_images = [cv2.resize(img, size) for img in valid_images]
    combined_image = np.hstack(resized_images) if len(resized_images) > 1 else resized_images[0]
    cv2.imwrite(output_path, combined_image)
    print(f"Saved: {output_path}")

# Main Workflow
if __name__ == "__main__":
    input_folder = "/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/data/mias_images"  # Input folder path
    output_folder = ("/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/output_images")  # Output folder path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.pgm', '.jpg', '.png')):
            filepath = os.path.join(input_folder, filename)
            original_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                print(f"Could not read {filename}")
                continue

            # Step 1: Contrast Enhancement
            clahe_image = apply_clahe(original_image)

            # Step 5: Region Growing
            center_seed = (clahe_image.shape[1] // 2, original_image.shape[0] // 2)
            segmented_region = region_growing(clahe_image, center_seed, threshold=15)

            # Step 3: Dilation
            dilated_image = apply_dilation(segmented_region)

            # Step 7: Crop Region
            cropped_region = crop_segmented_region(original_image, dilated_image)

            # Step 8: Save Results
            images = [cropped_region]
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_results.jpg")
            save_combined_results(output_path, images)