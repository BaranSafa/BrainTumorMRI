import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.optimizers import adamax_v2 as Adamax
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import warnings
warnings.filterwarnings("ignore")

# Function to create a DataFrame for training data
def train_df(tr_path):
     # Get class labels and image paths
    classes, class_paths = zip(*[(label, os.path.join(tr_path, label, image))
                                for label in os.listdir(tr_path) if os.path.isdir(os.path.join(tr_path, label))
                                for image in os.listdir(os.path.join(tr_path, label))])

    # Create DataFrame
    tr_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return tr_df

# Function to create a DataFrame for testing data
def test_df(ts_path):
    classes, class_paths = zip(*[(label, os.path.join(ts_path, label, image))
                                 for label in os.listdir(ts_path) if os.path.isdir(os.path.join(ts_path, label))
                                 for image in os.listdir(os.path.join(ts_path, label))])

    # Create DataFrame
    ts_df = pd.DataFrame({'Class Path': class_paths, 'Class': classes})
    return ts_df

tr_df = train_df('archive/Training')
# Display the training DataFrame
tr_df

ts_df = test_df('archive/Testing')
# Display the testing DataFrame
ts_df

# Count of images in each class in train data
plt.figure(figsize=(15,7))
ax = sns.countplot(data=tr_df , y=tr_df['Class'])

plt.xlabel('')
plt.ylabel('')
plt.title('Count of images in each class', fontsize=20)
ax.bar_label(ax.containers[0])
plt.show()

#Count each class in test data
plt.figure(figsize=(15, 7))
ax = sns.countplot(y=ts_df['Class'], palette='viridis')

ax.set(xlabel='', ylabel='', title='Count of images in each class')
ax.bar_label(ax.containers[0])

plt.show()

# Split the test data into validation and test sets
valid_df, ts_df = train_test_split(ts_df, train_size=0.5, random_state=20, stratify=ts_df['Class'])

valid_df

batch_size = 32
img_size = (299, 299)

# Create ImageDataGenerator for training data with rescaling and brightness adjustment
_gen = ImageDataGenerator(rescale=1/255,
                          brightness_range=(0.8, 1.2))

# Create ImageDataGenerator for test data with rescaling
ts_gen = ImageDataGenerator(rescale=1/255)

# Create data generator for training data
tr_gen = _gen.flow_from_dataframe(tr_df, x_col='Class Path',
                                  y_col='Class', batch_size=batch_size,
                                  target_size=img_size)

# Create data generator for validation data
valid_gen = _gen.flow_from_dataframe(valid_df, x_col='Class Path',
                                     y_col='Class', batch_size=batch_size,
                                     target_size=img_size)

# Create data generator for test data
ts_gen = ts_gen.flow_from_dataframe(ts_df, x_col='Class Path',
                                  y_col='Class', batch_size=16,
                                  target_size=img_size, shuffle=False)

# Get class indices and class names
class_dict = tr_gen.class_indices
classes = list(class_dict.keys())

# Get a batch of images and labels from the test generator
images, labels = next(ts_gen)

plt.figure(figsize=(20, 20))

# Plot a batch of test images with their predicted class names
for i, (image, label) in enumerate(zip(images, labels)):
    plt.subplot(4,4, i + 1)
    plt.imshow(image)
    class_name = classes[np.argmax(label)]
    plt.title(class_name, color='k', fontsize=15)

plt.show()

# Define the input shape for the model
img_shape = (299, 299, 3)

# Load the pre-trained Xception model without the top layer
base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
    
for layer in base_model.layers: #burası
    layer.trainable = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall


# Build the model by adding layers to the base model
model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')
])

# Compile the model with Adamax optimizer and categorical crossentropy loss
model.compile(Adamax(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy',
                       Precision(),
                       Recall()])

# Print the model summary
model.summary()

# Plot the model architecture
tf.keras.utils.plot_model(model, show_shapes=True)

# Train the model with training data and validate with validation data
hist = model.fit(tr_gen,
                 epochs=10,
                 validation_data=valid_gen,
                 shuffle= False)

# Get the keys of the history object
hist.history.keys()

# Extract training metrics from the history object
tr_acc = hist.history['accuracy']
tr_loss = hist.history['loss']
tr_per = hist.history['precision']
tr_recall = hist.history['recall']
# Extract validation metrics from the history object
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']
val_per = hist.history['val_precision']
val_recall = hist.history['val_recall']

# Find the epoch with the lowest validation loss
index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
# Find the epoch with the highest validation accuracy
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]
# Find the epoch with the highest validation precision
index_precision = np.argmax(val_per)
per_highest = val_per[index_precision]
# Find the epoch with the highest validation recall
index_recall = np.argmax(val_recall)
recall_highest = val_recall[index_recall]

# Create a list of epoch numbers
Epochs = [i + 1 for i in range(len(tr_acc))]
# Create labels for the best epochs
loss_label = f'Best epoch = {str(index_loss + 1)}'
acc_label = f'Best epoch = {str(index_acc + 1)}'
per_label = f'Best epoch = {str(index_precision + 1)}'
recall_label = f'Best epoch = {str(index_recall + 1)}'

# Plot training and validation loss
plt.figure(figsize=(20, 12))
plt.style.use('fivethirtyeight')


plt.subplot(2, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot training and validation accuracy
plt.subplot(2, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training and validation precision
plt.subplot(2, 2, 3)
plt.plot(Epochs, tr_per, 'r', label='Precision')
plt.plot(Epochs, val_per, 'g', label='Validation Precision')
plt.scatter(index_precision + 1, per_highest, s=150, c='blue', label=per_label)
plt.title('Precision and Validation Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Plot training and validation recall
plt.subplot(2, 2, 4)
plt.plot(Epochs, tr_recall, 'r', label='Recall')
plt.plot(Epochs, val_recall, 'g', label='Validation Recall')
plt.scatter(index_recall + 1, recall_highest, s=150, c='blue', label=recall_label)
plt.title('Recall and Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

# Add a super title for the entire figure
plt.suptitle('Model Training Metrics Over Epochs', fontsize=16)
plt.show()

# Evaluate the model on the training, validation, test data
train_score = model.evaluate(tr_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
test_score = model.evaluate(ts_gen, verbose=1)

# Print the evaluation results for training data
print(f"Train Loss: {train_score[0]:.4f}")
print(f"Train Accuracy: {train_score[1]*100:.2f}%")
print('-' * 20)
# Print the evaluation results for validation data
print(f"Validation Loss: {valid_score[0]:.4f}")
print(f"Validation Accuracy: {valid_score[1]*100:.2f}%")
print('-' * 20)
# Print the evaluation results for test data
print(f"Test Loss: {test_score[0]:.4f}")
print(f"Test Accuracy: {test_score[1]*100:.2f}%")

# Get the predictions for the test data and get predicted class indices
preds = model.predict(ts_gen)
y_pred = np.argmax(preds, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(ts_gen.classes, y_pred)
# Get the class labels
labels = list(class_dict.keys())
# Plot the confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('Truth Label')
plt.show()

clr = classification_report(ts_gen.classes, y_pred)
# Print the classification report
print(clr)

# Function to predict the class of an image
def predict(img_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    # Get the class labels
    label = list(class_dict.keys())
    # Set up the plot
    plt.figure(figsize=(12, 12))

    # Open and preprocess the image
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    # Make predictions
    predictions = model.predict(img)
    probs = list(predictions[0])
    labels = label
    # Plot the image
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    # Plot the prediction probabilities
    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt = '%.2f')
    plt.show()

predict('archive/Testing/meningioma/Te-meTr_0000.jpg')

predict('archive/Testing/glioma/Te-glTr_0007.jpg')