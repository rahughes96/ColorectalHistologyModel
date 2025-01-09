import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
from tensorflow.keras.utils import to_categorical

# TensorBoard log directory
tb_log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(tb_log_dir, exist_ok=True)

# Directories to save models and metrics
model_dir = "models/"
metrics_dir = "metrics/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

def build_cnn(input_shape, num_classes, filters_1, filters_2, dense_units, dropout_1, dropout_2):
    """
    Builds and compiles a CNN model for colorectal cancer classification.

    Parameters:
    - input_shape (tuple): Shape of the input images (height, width, channels).
    - num_classes (int): Number of output classes.
    - filters_1 (int): Number of filters in the first convolutional layer.
    - filters_2 (int): Number of filters in the second convolutional layer.
    - dense_units (int): Number of units in the dense layer.
    - dropout_1 (float): Dropout rate after the first pooling layer.
    - dropout_2 (float): Dropout rate after the dense layer.

    Returns:
    - model (tf.keras.Model): Compiled CNN model.
    """
    model = Sequential([
        Conv2D(filters_1, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_1),
        Conv2D(filters_2, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(dropout_2),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = LegacyAdam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def preprocess_image(image_path):
    """
    Preprocesses an image by resizing and normalizing pixel values.

    Parameters:
    - image_path (str): Path to the image file.

    Returns:
    - img (np.array): Preprocessed image array.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return img_array

def load_and_split_data():
    """
    Loads, preprocesses, and splits the colorectal cancer dataset into training, validation, and test sets.

    Returns:
    - X_train, X_val, X_test: Feature arrays for training, validation, and testing.
    - y_train, y_val, y_test: Label arrays for training, validation, and testing.
    - classes (list): List of class names.
    """
    data_dir = "Kather_texture_2016_image_tiles_5000"
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    X = []
    y = []

    for idx, category in enumerate(classes):
        folder = os.path.join(data_dir, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            X.append(preprocess_image(img_path))
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    y_train = to_categorical(y_train, num_classes=len(classes))
    y_val = to_categorical(y_val, num_classes=len(classes))
    y_test = to_categorical(y_test, num_classes=len(classes))

    return X_train, X_val, X_test, y_train, y_val, y_test, classes

def evaluate_model(model, X_test, y_test, classes):
    """
    Evaluates the model on the test dataset.

    Parameters:
    - model (tf.keras.Model): Trained model to evaluate.
    - X_test: Test feature array.
    - y_test: Test label array.
    - classes: List of class names.

    Returns:
    - metrics (dict): Evaluation metrics including accuracy and classification report.
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "classification_report": class_report
    }

def tune_hyperparameters():
    """
    Tunes hyperparameters of the CNN model manually by testing combinations.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, classes = load_and_split_data()

    # Define parameter ranges
    filters_1 = [32, 64]
    filters_2 = [64, 128]
    dense_units = [64, 128]
    dropout_1 = [0.25, 0.3]
    dropout_2 = [0.3, 0.5]
    batch_sizes = [32, 64]

    best_model = None
    best_accuracy = 0
    best_params = {}

    for f1, f2, d_units, d1, d2, batch_size in product(
        filters_1, filters_2, dense_units, dropout_1, dropout_2, batch_sizes
    ):
        print(f"Testing combination: Filters1={f1}, Filters2={f2}, Dense={d_units}, Dropout1={d1}, Dropout2={d2}, Batch Size={batch_size}")

        # Build model
        model = build_cnn(
            input_shape=X_train.shape[1:],
            num_classes=len(classes),
            filters_1=f1,
            filters_2=f2,
            dense_units=d_units,
            dropout_1=d1,
            dropout_2=d2
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,  # Fewer epochs for faster tuning
            batch_size=batch_size,
            verbose=0
        )

        # Evaluate model
        print(val_accuracy)
        val_accuracy = history.history['val_accuracy'][-1]
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_params = {
                'filters_1': f1,
                'filters_2': f2,
                'dense_units': d_units,
                'dropout_1': d1,
                'dropout_2': d2,
                'batch_size': batch_size
            }
            print(f"New best model found with accuracy: {best_accuracy}")

    print("Best Hyperparameters:", best_params)
    return best_model, best_params

def train_and_evaluate():
    """
    Trains and evaluates the CNN model for colorectal cancer classification.
    """
    best_model, best_params = tune_hyperparameters()

    # Reload data
    X_train, X_val, X_test, y_train, y_val, y_test, classes = load_and_split_data()

    # Evaluate the best model on the test set
    metrics = evaluate_model(best_model, X_test, y_test, classes)

    # Save the best model and its metrics
    model_path = os.path.join(model_dir, f"best_cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    best_model.save(model_path)

    metrics_path = os.path.join(metrics_dir, f"best_cnn_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_and_evaluate()


