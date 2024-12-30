import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_preprocessed_data(features_file, labels_file):
    """
    Load preprocessed features and labels from CSV files.

    :param features_file: Path to the CSV file containing preprocessed features.
    :param labels_file: Path to the CSV file containing preprocessed labels.
    :return: Features (X) and labels (y)
    """
    # Load features and labels
    X = pd.read_csv(features_file).values
    y = pd.read_csv(labels_file)['Class'].values
    return X, y


def train_svm(X_train, y_train):
    """
    Train an SVM classifier with hyperparameter tuning.

    :param X_train: Training feature matrix.
    :param y_train: Training labels.
    :return: Best SVM model
    """
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }

    # Initialize the SVM classifier
    svm = SVC()

    # Use GridSearchCV to find the best parameters
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # Return the best model
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained SVM model on the test set.

    :param model: Trained SVM model.
    :param X_test: Test feature matrix.
    :param y_test: Test labels.
    """
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # File paths for preprocessed data
    features_file = '/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/data/preprocessed_features.csv'  # Replace with the correct path to your features file
    labels_file = '/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/data/preprocessed_labels.csv'  # Replace with the correct path to your labels file

    # Step 1: Load preprocessed data
    print("Loading preprocessed data...")
    X, y = load_preprocessed_data(features_file, labels_file)

    # Step 2: Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 3: Apply SMOTE to the training set
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("SMOTE applied. Class distribution after resampling:")
    print(pd.Series(y_train).value_counts())

    # Step 4: Normalize the features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 5: Train the SVM model
    print("Training SVM model with hyperparameter tuning...")
    svm_model = train_svm(X_train, y_train)

    # Step 6: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(svm_model, X_test, y_test)

    # Step 7: Save the trained model and scaler
    print("Saving the trained model and scaler...")
    joblib.dump(svm_model, '/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/models/svm_model.pkl')
    joblib.dump(scaler, '/Users/ecekocabay/Desktop/BreastCancerDetection_noyan/models/scaler.pkl')
    print("Trained model saved as 'svm_model.pkl'.")
    print("Scaler saved as 'scaler.pkl'.")