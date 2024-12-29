import joblib
import segmentation
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage import io
from skimage.transform import resize
import numpy as np
import argparse


def compute_lbp(image, radius=1, n_points=8):
    """
    Computes the Local Binary Pattern (LBP) histogram for a given image.
    """
    if image.max() > 1:
        image = (image / image.max() * 255).astype(np.uint8)

    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist


def predict_image(image_path, model_path, scaler_path=None, resize_shape=(256, 256)):
    """
    Predicts the class of a single mammogram image using a given model.

    :param image_path: Path to the image file.
    :param model_path: Path to the saved model.
    :param scaler_path: Path to the saved scaler (optional).
    :param resize_shape: Tuple specifying the shape for resizing the image.
    :return: Predicted class label.
    """
    # Load the model
    model = joblib.load(model_path)

    # Load the scaler if provided
    scaler = joblib.load(scaler_path) if scaler_path else None

    # Load the image
    image = io.imread(image_path, as_gray=True)
    image = (image * 255).astype(np.uint8)  # Normalize for OpenCV compatibility

    # Step 1: Apply CLAHE
    enhanced_image = segmentation.apply_clahe(image)

    # Step 2: Perform Region Growing
    center_seed = (enhanced_image.shape[1] // 2, enhanced_image.shape[0] // 2)
    segmented_mask = segmentation.region_growing(enhanced_image, center_seed, threshold=15)

    # Step 3: Apply Dilation
    dilated_mask = segmentation.apply_dilation(segmented_mask)

    # Step 4: Crop the Segmented Region
    cropped_image = segmentation.crop_segmented_region(image, dilated_mask)
    if cropped_image is None:
        raise ValueError("No region could be segmented from the image.")

    # Step 5: Resize the Cropped Region for Consistency
    resized_image = resize(cropped_image, resize_shape, anti_aliasing=True)

    # Step 6: Extract LBP Features
    lbp_features = compute_lbp(resized_image)
    features = np.array([lbp_features])  # Convert to 2D array for prediction

    # Step 7: Normalize the Features (if scaler is available)
    if scaler:
        features = scaler.transform(features)

    # Step 8: Predict the Class
    prediction = model.predict(features)
    return prediction[0]


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Predict class for a mammogram image.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--model", required=True, help="Path to the saved model file.")
    parser.add_argument("--scaler", help="Path to the saved scaler file (optional).")
    parser.add_argument("--resize_shape", default="256,256", help="Resize shape for the cropped image (default: 256,256).")
    args = parser.parse_args()

    # Parse resize shape
    resize_shape = tuple(map(int, args.resize_shape.split(',')))

    # Perform prediction
    try:
        predicted_class = predict_image(args.image, args.model, args.scaler, resize_shape)
        print(f"The predicted class for the image is: {predicted_class}")
    except Exception as e:
        print(f"Error during prediction: {e}")