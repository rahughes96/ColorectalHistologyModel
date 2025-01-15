import os
import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (50, 50))
    img = img / 255.0  # Normalize pixel values
    return img

def extract_texture_features(image):
    """
    Extract GLCM texture features from an image.
    """
    gray = rgb2gray(image)  # Convert to grayscale
    gray = (gray * 255).astype('uint8')  # Rescale to 8-bit
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        symmetric=True, normed=True)
    
    features = [
        graycoprops(glcm, prop).mean() for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    ]
    return features

def save_metrics(history, classification_report_dict, output_dir="metrics"):
    """ Save training metrics and classification report to JSON files with unique timestamps. """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save training history
    history_path = os.path.join(output_dir, f"training_history_{timestamp}.json")
    history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_dict, f)

    # Save classification report
    report_path = os.path.join(output_dir, f"classification_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(classification_report_dict, f, indent=4)

if __name__ == "__main__":
    # Paths
    data_dir = "Kather_texture_2016_image_tiles_5000"
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    input("vizualise images? y/n:")

    # Visualize some images
    if input == 'y':
        for category in classes:
            folder = os.path.join(data_dir, category)
            for img_name in os.listdir(folder)[:3]:
                img_path = os.path.join(folder, img_name)
                img = cv2.imread(img_path)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(category)
                plt.show()
    elif input == 'n':
        pass
    else:
        pass

    # Data Preprocessing
    X_cnn = []
    X_texture = []
    y = []

    for idx, category in enumerate(classes):
        folder = os.path.join(data_dir, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = preprocess_image(img_path)
            texture_features = extract_texture_features(img)
            X_cnn.append(img)
            X_texture.append(texture_features)
            y.append(idx)

    X_cnn = np.array(X_cnn)
    X_texture = np.array(X_texture)
    y = np.array(y)

    # Split data
    X_cnn_train, X_cnn_test, X_texture_train, X_texture_test, y_train, y_test = train_test_split(
        X_cnn, X_texture, y, test_size=0.2, random_state=42)

    y_train = to_categorical(y_train, num_classes=len(classes))
    y_test_cat = to_categorical(y_test, num_classes=len(classes))

    # Scale texture features
    scaler = StandardScaler()
    X_texture_train = scaler.fit_transform(X_texture_train)
    X_texture_test = scaler.transform(X_texture_test)

    # Define CNN Model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    datagen.fit(X_cnn_train)

    history = model.fit(
        datagen.flow(X_cnn_train, y_train, batch_size=32),
        validation_data=(X_cnn_test, y_test_cat),
        epochs=50,
        callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
    )

    # Extract CNN embeddings
    cnn_features_train = model.predict(X_cnn_train)
    cnn_features_test = model.predict(X_cnn_test)

    # Combine CNN embeddings and texture features
    combined_train = np.hstack((cnn_features_train, X_texture_train))
    combined_test = np.hstack((cnn_features_test, X_texture_test))

    # Train combined classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(combined_train, np.argmax(y_train, axis=1))

    # Evaluate
    y_pred = rf_clf.predict(combined_test)
    y_true = y_test

    classification_report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    save_metrics(history, classification_report_dict)

    # Plot accuracy and loss
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=classes))
