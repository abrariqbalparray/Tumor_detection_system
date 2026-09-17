
# ============================================================
# BRAIN TUMOR DETECTION - MODEL TRAINING
# Tumor Present vs No Tumor
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

# Import TensorFlow for deep-learning model training.
import tensorflow as tf

# Import Pandas for saving the training history as a table.
import pandas as pd

# Import Matplotlib for creating training graphs.
import matplotlib.pyplot as plt

# Import MobileNetV2 as our pretrained CNN model.
from tensorflow.keras.applications import MobileNetV2

# Import MobileNetV2 preprocessing.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Import Keras layers for building the model.
from tensorflow.keras import layers

# Import Sequential for building the model.
from tensorflow.keras.models import Sequential

# Import EarlyStopping to stop training when validation performance
# stops improving.
from tensorflow.keras.callbacks import EarlyStopping

# Import os for creating the model folder.
import os


# ------------------------------------------------------------
# STEP 2: DEFINE BASIC SETTINGS
# ------------------------------------------------------------

# Resize every MRI image to 224 x 224 pixels.
IMAGE_SIZE = (224, 224)

# Process 32 images at a time.
BATCH_SIZE = 32

# Reserve 20% of the training dataset for validation.
VALIDATION_SPLIT = 0.20

# Use a fixed seed so that the same images are selected for
# training and validation each time.
SEED = 42

# Set the maximum number of training epochs.
EPOCHS = 10


# ------------------------------------------------------------
# STEP 3: DEFINE DATASET FOLDERS
# ------------------------------------------------------------

# Location of the training images.
TRAINING_FOLDER = "dataset/Training"

# Location of the testing images.
TESTING_FOLDER = "dataset/Testing"


# ------------------------------------------------------------
# STEP 4: LOAD TRAINING IMAGES
# ------------------------------------------------------------

# Load images from the Training folder.
#
# IMPORTANT:
# The actual class folder names in your dataset are:
#
# no tumor
# tumor
#
# We explicitly specify this order:
#
# 0 = no tumor
# 1 = tumor
#
training_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAINING_FOLDER,

    # Automatically obtain labels from the folder names.
    labels="inferred",

    # Use binary labels because there are exactly two classes.
    label_mode="binary",

    # Use the exact folder names present in your dataset.
    class_names=[
        "no tumor",
        "tumor"
    ],

    # Reserve 20% of the training folder for validation.
    validation_split=VALIDATION_SPLIT,

    # Load the training portion.
    subset="training",

    # Use the same random seed for reproducibility.
    seed=SEED,

    # Resize all images to 224 x 224.
    image_size=IMAGE_SIZE,

    # Load 32 images per batch.
    batch_size=BATCH_SIZE,

    # Shuffle training images.
    shuffle=True
)


# ------------------------------------------------------------
# STEP 5: LOAD VALIDATION IMAGES
# ------------------------------------------------------------

# Load the validation portion of the same Training folder.
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAINING_FOLDER,

    # Read class labels from folder names.
    labels="inferred",

    # Use binary labels.
    label_mode="binary",

    # Use the exact folder names.
    class_names=[
        "no tumor",
        "tumor"
    ],

    # Use the same 20% validation split.
    validation_split=VALIDATION_SPLIT,

    # Load the validation portion.
    subset="validation",

    # Use the same seed so the split matches the training split.
    seed=SEED,

    # Resize all images to 224 x 224.
    image_size=IMAGE_SIZE,

    # Load 32 images per batch.
    batch_size=BATCH_SIZE,

    # Do not shuffle validation data.
    shuffle=False
)


# ------------------------------------------------------------
# STEP 6: LOAD TESTING IMAGES
# ------------------------------------------------------------

# Load the separate Testing folder.
#
# This data is kept completely separate from training and validation.
testing_dataset = tf.keras.utils.image_dataset_from_directory(
    TESTING_FOLDER,

    # Read labels from folder names.
    labels="inferred",

    # Use binary labels.
    label_mode="binary",

    # Keep the same class order.
    # 0 = no tumor
    # 1 = tumor
    class_names=[
        "no tumor",
        "tumor"
    ],

    # Resize testing images to the same size as training images.
    image_size=IMAGE_SIZE,

    # Load 32 images per batch.
    batch_size=BATCH_SIZE,

    # Do not shuffle testing images.
    shuffle=False
)


# ------------------------------------------------------------
# STEP 7: DISPLAY CLASS MAPPING
# ------------------------------------------------------------

# Display the class names recognized by TensorFlow.
print("\n========================================")
print("CLASS MAPPING")
print("========================================")

print(
    training_dataset.class_names
)

# Display the meaning of the numerical labels.
print("\n0 = No Tumor")
print("1 = Tumor")


# ------------------------------------------------------------
# STEP 8: IMPROVE DATA LOADING PERFORMANCE
# ------------------------------------------------------------

# Ask TensorFlow to automatically determine the best
# prefetching performance for this computer.
AUTOTUNE = tf.data.AUTOTUNE

# Prefetch future training batches while the current batch is processed.
training_dataset = training_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# Prefetch validation batches.
validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# Prefetch testing batches.
testing_dataset = testing_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ------------------------------------------------------------
# STEP 9: CREATE IMAGE AUGMENTATION
# ------------------------------------------------------------

# Create data augmentation layers.
#
# These operations create slightly different versions of
# training images so the model can learn more robust patterns.
data_augmentation = Sequential(
    [
        # Randomly flip images horizontally.
        layers.RandomFlip(
            "horizontal"
        ),

        # Slightly rotate images.
        layers.RandomRotation(
            0.05
        ),

        # Slightly zoom images.
        layers.RandomZoom(
            0.10
        )
    ],
    name="data_augmentation"
)


# ------------------------------------------------------------
# STEP 10: LOAD PRETRAINED MOBILENETV2
# ------------------------------------------------------------

# Load MobileNetV2 using pretrained ImageNet weights.
#
# include_top=False removes the original ImageNet output layer,
# because our project needs only:
#
# No Tumor
# Tumor
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(
        224,
        224,
        3
    )
)


# ------------------------------------------------------------
# STEP 11: FREEZE THE PRETRAINED CNN
# ------------------------------------------------------------

# Freeze the pretrained MobileNetV2 layers.
# Their weights will not initially change during training.
base_model.trainable = False


# ------------------------------------------------------------
# STEP 12: BUILD THE FINAL MODEL
# ------------------------------------------------------------

# Build a new model using the pretrained CNN plus our
# binary classification layer.
model = Sequential(
    [

        # Apply data augmentation during training.
        data_augmentation,

        # Convert image pixels into the format expected
        # by MobileNetV2.
        layers.Lambda(
            preprocess_input
        ),

        # Use MobileNetV2 to extract visual features.
        base_model,

        # Convert feature maps into one feature vector.
        layers.GlobalAveragePooling2D(),

        # Randomly deactivate 30% of neurons during training
        # to help reduce overfitting.
        layers.Dropout(
            0.30
        ),

        # Create one output neuron.
        #
        # sigmoid produces a value between 0 and 1.
        #
        # Near 0 = No Tumor
        # Near 1 = Tumor
        layers.Dense(
            1,
            activation="sigmoid"
        )
    ]
)


# ------------------------------------------------------------
# STEP 13: COMPILE THE MODEL
# ------------------------------------------------------------

# Configure the neural network for binary classification.
model.compile(

    # Adam optimizer updates model weights during training.
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    # Binary cross-entropy is suitable for two classes.
    loss="binary_crossentropy",

    # Track classification accuracy during training.
    metrics=[
        "accuracy"
    ]
)


# ------------------------------------------------------------
# STEP 14: DISPLAY MODEL ARCHITECTURE
# ------------------------------------------------------------

# Print the complete model architecture.
print("\n========================================")
print("MODEL ARCHITECTURE")
print("========================================")

model.summary()


# ------------------------------------------------------------
# STEP 15: CREATE EARLY STOPPING
# ------------------------------------------------------------

# Stop training when validation loss stops improving.
early_stopping = EarlyStopping(

    # Monitor validation loss.
    monitor="val_loss",

    # Wait for 3 epochs without improvement.
    patience=3,

    # Restore the weights from the best epoch.
    restore_best_weights=True
)


# ------------------------------------------------------------
# STEP 16: START TRAINING
# ------------------------------------------------------------

# Display a message before training begins.
print("\n========================================")
print("STARTING MODEL TRAINING")
print("========================================")


# Train the model.
history = model.fit(

    # Use the training images.
    training_dataset,

    # Evaluate on validation images after each epoch.
    validation_data=validation_dataset,

    # Train for up to 10 epochs.
    epochs=EPOCHS,

    # Enable early stopping.
    callbacks=[
        early_stopping
    ]
)


# ------------------------------------------------------------
# STEP 17: CREATE MODEL FOLDER
# ------------------------------------------------------------

# Create the model folder if it does not already exist.
os.makedirs(
    "model",
    exist_ok=True
)


# ------------------------------------------------------------
# STEP 18: SAVE THE TRAINED MODEL
# ------------------------------------------------------------

# Save the complete trained model.
model.save(
    "model/final_model.keras"
)


# ------------------------------------------------------------
# STEP 19: SAVE TRAINING HISTORY
# ------------------------------------------------------------

# Convert the Keras history into a Pandas DataFrame.
history_df = pd.DataFrame(
    history.history
)


# Save the history so we can analyze it later.
history_df.to_csv(
    "results/training_history.csv",
    index=False
)


# ------------------------------------------------------------
# STEP 20: CREATE ACCURACY GRAPH
# ------------------------------------------------------------

# Create a new figure for the accuracy graph.
plt.figure(
    figsize=(8, 5)
)


# Plot training accuracy.
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)


# Plot validation accuracy.
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)


# Add a title.
plt.title(
    "Training and Validation Accuracy"
)


# Label the x-axis.
plt.xlabel(
    "Epoch"
)


# Label the y-axis.
plt.ylabel(
    "Accuracy"
)


# Display the legend.
plt.legend()


# Adjust the graph layout.
plt.tight_layout()


# Save the graph.
plt.savefig(
    "results/training_accuracy.png"
)


# Display the graph.
plt.show()


# ------------------------------------------------------------
# STEP 21: CREATE LOSS GRAPH
# ------------------------------------------------------------

# Create a new figure for the loss graph.
plt.figure(
    figsize=(8, 5)
)


# Plot training loss.
plt.plot(
    history.history["loss"],
    label="Training Loss"
)


# Plot validation loss.
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


# Add a title.
plt.title(
    "Training and Validation Loss"
)


# Label the x-axis.
plt.xlabel(
    "Epoch"
)


# Label the y-axis.
plt.ylabel(
    "Loss"
)


# Display the legend.
plt.legend()


# Adjust the layout.
plt.tight_layout()


# Save the graph.
plt.savefig(
    "results/training_loss.png"
)


# Display the graph.
plt.show()


# ------------------------------------------------------------
# STEP 22: EVALUATE THE MODEL ON TEST DATA
# ------------------------------------------------------------

# Display a heading.
print("\n========================================")
print("TEST DATA EVALUATION")
print("========================================")


# Evaluate the final model on the separate Testing dataset.
test_loss, test_accuracy = model.evaluate(
    testing_dataset
)


# Display the testing loss.
print(
    "\nTest Loss:",
    round(test_loss, 4)
)


# Display the testing accuracy.
print(
    "Test Accuracy:",
    round(test_accuracy, 4)
)


# ------------------------------------------------------------
# STEP 23: DISPLAY FINAL FILES
# ------------------------------------------------------------

# Display a completion message.
print("\n========================================")
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("========================================")


# Display the saved model location.
print(
    "\nSaved Model:"
)

print(
    "model/final_model.keras"
)


# Display the saved training history.
print(
    "\nTraining History:"
)

print(
    "results/training_history.csv"
)


# Display the saved accuracy graph.
print(
    "\nAccuracy Graph:"
)

print(
    "results/training_accuracy.png"
)


# Display the saved loss graph.
print(
    "\nLoss Graph:"
)

print(
    "results/training_loss.png"
)

