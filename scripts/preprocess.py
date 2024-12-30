import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_data(features_file, labels_file):
    """
    Load features and labels from CSV files.

    :param features_file: Path to the CSV file containing features.
    :param labels_file: Path to the CSV file containing labels.
    :return: Features (X), labels (y)
    """
    # Load features
    features_df = pd.read_csv(features_file)
    # Drop 'Image Name' column if present
    if 'Image Name' in features_df.columns:
        features_df = features_df.drop(columns=['Image Name'])
    X = features_df.values

    # Load labels
    labels_df = pd.read_csv(labels_file)
    y = labels_df['Class'].values

    return X, y

def normalize_features(X):
    """
    Normalize or scale features using StandardScaler.

    :param X: Feature matrix.
    :return: Scaled feature matrix, fitted scaler object.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def handle_imbalance(X, y):
    """
    Handle class imbalance using SMOTE.

    :param X: Feature matrix.
    :param y: Labels.
    :return: Resampled feature matrix and labels.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

if __name__ == "__main__":
    # Example usage
    features_file = '/data/lbp_feature_matrix.csv'  # Replace with the correct path to your features file
    labels_file = '/data/mias_labels_corrected.csv'  # Replace with the correct path to your labels file

    #Load data
    print("Loading data...")
    X, y = load_data(features_file, labels_file)

    # Normalize features
    print("Normalizing features...")
    X_scaled, scaler = normalize_features(X)

    # Save the scaler
    joblib.dump(scaler, '../models/scaler.pkl')
    print("Scaler saved as 'scaler.pkl'.")

    # Handle class imbalance
    print("Handling class imbalance...")
    X_resampled, y_resampled = handle_imbalance(X_scaled, y)

    # Save the preprocessed data
    preprocessed_features_file = '/data/preprocessed_features.csv'
    preprocessed_labels_file = '/data/preprocessed_labels.csv'
    pd.DataFrame(X_resampled).to_csv(preprocessed_features_file, index=False)
    pd.DataFrame(y_resampled, columns=['Class']).to_csv(preprocessed_labels_file, index=False)

    print(f"Preprocessed features saved to {preprocessed_features_file}")
    print(f"Preprocessed labels saved to {preprocessed_labels_file}")