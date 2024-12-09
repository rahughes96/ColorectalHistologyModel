import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.color import rgb2gray

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Resize if needed
    img = img / 255.0  # Normalize pixel values
    return img

def extract_texture_features(image):
    """
    Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).
    
    Args:
        image: RGB image as a NumPy array.
        
    Returns:
        A feature vector of texture properties.
    """
    # Convert to grayscale
    gray_image = rgb2gray(image)
    
    # Scale the grayscale image to uint8 (0–255)
    gray_image = (gray_image * 255).astype(np.uint8)
    
    # Compute the GLCM
    glcm = graycomatrix(
        gray_image,
        distances=[1], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        levels=256,
        symmetric=True, 
        normed=True
    )
    
    # Extract texture properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    ASM = graycoprops(glcm, 'ASM').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # Return features as a vector
    return [contrast, dissimilarity, homogeneity, ASM, energy, correlation]

# Color feature extraction using histograms
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

if __name__ == "__main__":

    #Preprocess Data
    data_dir = "Kather_texture_2016_image_tiles_5000"
    classes = os.listdir(data_dir)  # Get class names

    classes.remove(".DS_Store")

    X_texture = []  # Texture features
    X_color = []    # Color features
    X_images = []   # Image features for CNN
    y = []          # Labels

    for idx, category in enumerate(classes):
        folder = os.path.join(data_dir, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            # Preprocess image for CNN
            img = preprocess_image(img_path)  # Ensure preprocess_image resizes & normalizes images
            X_images.append(img)

            # Load original image for feature extraction
            image = load_img(img_path, target_size=(150, 150))  # Load with target size
            image = img_to_array(image) / 255.0  # Normalize to [0, 1]

            # Extract features
            texture = extract_texture_features(image)  # Texture features
            color = extract_color_histogram(image)     # Color histogram features

            # Store features and label
            X_texture.append(texture)
            X_color.append(color)
            y.append(idx)

    # Convert to NumPy arrays
    X_images = np.array(X_images)
    X_texture = np.array(X_texture)
    X_color = np.array(X_color)
    y = np.array(y)

    # Combine features into a single feature set
    X_combined = np.concatenate([X_texture, X_color], axis=1)

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=len(classes))
    y_test = to_categorical(y_test, num_classes=len(classes))

    #Build the CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #Train the model
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=(X_test, y_test),
                        epochs=10)

    #Evaluate the Model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred, target_names=classes))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.show()