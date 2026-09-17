
# Import os so we can work with folders and file paths.
import os

# Import NumPy for numerical calculations and arrays.
import numpy as np

# Import Pandas for creating and saving tables as CSV files.
import pandas as pd

# Import TensorFlow for loading the trained deep-learning model
# and reading the MRI image dataset.
import tensorflow as tf

# Import Matplotlib for creating evaluation graphs.
import matplotlib.pyplot as plt

# Import the evaluation metrics we need from scikit-learn.
from sklearn.metrics import (
    accuracy_score,              # Measures overall prediction accuracy.
    balanced_accuracy_score,     # Measures average performance across classes.
    precision_score,             # Measures how many predicted tumors are actually tumors.
    recall_score,                # Measures how many actual tumors were detected.
    f1_score,                    # Combines precision and recall.
    roc_auc_score,               # Measures class separation using ROC-AUC.
    average_precision_score,     # Calculates Average Precision / PR-AUC.
    matthews_corrcoef,           # Calculates Matthews Correlation Coefficient.
    cohen_kappa_score,           # Measures agreement beyond chance.
    confusion_matrix,            # Creates the confusion matrix.
    classification_report,       # Creates precision/recall/F1 report.
    roc_curve,                   # Generates points for the ROC curve.
    precision_recall_curve       # Generates points for PR curve.
)


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

# Location of the testing dataset.
TEST_DIR = "dataset/Testing"

# Location of the trained Keras model.
MODEL_PATH = "model/final_model.keras"

# Folder where evaluation results will be stored.
RESULTS_DIR = "results"


# Create the results folder if it does not already exist.
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# MODEL AND DATASET SETTINGS
# ------------------------------------------------------------

# The same image size used during model training.
IMAGE_SIZE = (224, 224)

# Number of images processed at one time.
BATCH_SIZE = 32

# Probability threshold used to decide the predicted class.
# Below 0.50 = No Tumor.
# 0.50 or above = Tumor.
THRESHOLD = 0.50


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

# Load the trained MobileNetV2 model.
#
# The saved model contains a Lambda layer that uses
# MobileNetV2's preprocess_input function.
#
# custom_objects tells Keras exactly which function
# "preprocess_input" refers to.
#
# compile=False means we only need the model for prediction.
#
# safe_mode=False allows the Lambda layer in the saved model
# to be loaded.
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input":
            tf.keras.applications.mobilenet_v2.preprocess_input
    },
    compile=False,
    safe_mode=False
)


# Print a message so we know the model loaded successfully.
print("Model loaded successfully.")


# ------------------------------------------------------------
# LOAD TEST DATASET
# ------------------------------------------------------------

# Load the testing images from the Testing folder.
#
# image_size resizes every image to 224 x 224.
#
# batch_size loads 32 images at a time.
#
# shuffle=False keeps the image order fixed.
#
# class_names explicitly defines the correct class mapping:
#
#     no tumor = 0
#     tumor    = 1
#
# These names exactly match your dataset folder names.
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_names=["no tumor", "tumor"]
)


# Print the class names detected by TensorFlow.
print("Classes:", test_dataset.class_names)


# ------------------------------------------------------------
# STORE TRUE LABELS AND PREDICTIONS
# ------------------------------------------------------------

# This list will contain the actual labels from the test data.
true_labels = []

# This list will contain the model's tumor probabilities.
predicted_probabilities = []


# ------------------------------------------------------------
# GENERATE PREDICTIONS
# ------------------------------------------------------------

# Go through the testing dataset one batch at a time.
for images, labels in test_dataset:

    # Add the real labels of the current batch.
    true_labels.extend(labels.numpy())

    # Ask the trained model to predict the current images.
    #
    # verbose=0 prevents prediction progress messages
    # from filling the terminal.
    probabilities = model.predict(
        images,
        verbose=0
    )

    # The model normally returns shape:
    # (batch_size, 1)
    #
    # reshape(-1) converts it into:
    # (batch_size,)
    probabilities = probabilities.reshape(-1)

    # Add the probabilities to our complete prediction list.
    predicted_probabilities.extend(probabilities)


# Convert the actual labels into a NumPy integer array.
true_labels = np.array(
    true_labels,
    dtype=int
)

# Convert the probabilities into a NumPy floating-point array.
predicted_probabilities = np.array(
    predicted_probabilities,
    dtype=float
)


# ------------------------------------------------------------
# CONVERT PROBABILITY INTO CLASS
# ------------------------------------------------------------

# Compare every tumor probability with the threshold.
#
# Probability >= 0.50 becomes:
#     1 = Tumor
#
# Probability < 0.50 becomes:
#     0 = No Tumor
predicted_labels = (
    predicted_probabilities >= THRESHOLD
).astype(int)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

# Create the confusion matrix.
#
# Because labels=[0,1]:
#
#     0 = No Tumor
#     1 = Tumor
#
# The matrix is:
#
#                 Predicted
#                No Tumor  Tumor
#
# Actual No Tumor    TN      FP
# Actual Tumor      FN      TP
cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[0, 1]
)


# Extract the four values from the 2x2 confusion matrix.
TN, FP, FN, TP = cm.ravel()


# ------------------------------------------------------------
# CALCULATE METRICS
# ------------------------------------------------------------

# Calculate the overall percentage of correct predictions.
accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


# Calculate the average recall across both classes.
balanced_accuracy = balanced_accuracy_score(
    true_labels,
    predicted_labels
)


# Calculate precision for the Tumor class.
precision = precision_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


# Calculate recall/sensitivity for the Tumor class.
recall = recall_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


# Calculate specificity.
#
# Specificity = TN / (TN + FP)
#
# It tells us how well the model identifies images
# that actually have no tumor.
specificity = (
    TN / (TN + FP)
    if (TN + FP) > 0
    else 0.0
)


# Calculate the F1 score.
f1 = f1_score(
    true_labels,
    predicted_labels,
    zero_division=0
)


# Calculate ROC-AUC using the model's probabilities.
roc_auc = roc_auc_score(
    true_labels,
    predicted_probabilities
)


# Calculate Average Precision, which represents PR-AUC.
pr_auc = average_precision_score(
    true_labels,
    predicted_probabilities
)


# Calculate Matthews Correlation Coefficient.
mcc = matthews_corrcoef(
    true_labels,
    predicted_labels
)


# Calculate Cohen's Kappa.
kappa = cohen_kappa_score(
    true_labels,
    predicted_labels
)


# Calculate Negative Predictive Value.
#
# NPV = TN / (TN + FN)
#
# It tells us how reliable a No Tumor prediction is.
npv = (
    TN / (TN + FN)
    if (TN + FN) > 0
    else 0.0
)


# ------------------------------------------------------------
# PRINT EVALUATION RESULTS
# ------------------------------------------------------------

# Print a heading in the terminal.
print("\nMODEL EVALUATION")
print("=" * 50)


# Print accuracy as a percentage.
print(
    f"Accuracy             : {accuracy * 100:.2f}%"
)


# Print balanced accuracy as a percentage.
print(
    f"Balanced Accuracy    : {balanced_accuracy * 100:.2f}%"
)


# Print precision as a percentage.
print(
    f"Precision            : {precision * 100:.2f}%"
)


# Print recall/sensitivity as a percentage.
print(
    f"Recall / Sensitivity : {recall * 100:.2f}%"
)


# Print specificity as a percentage.
print(
    f"Specificity          : {specificity * 100:.2f}%"
)


# Print F1 score as a percentage.
print(
    f"F1 Score             : {f1 * 100:.2f}%"
)


# Print ROC-AUC.
print(
    f"ROC-AUC              : {roc_auc:.4f}"
)


# Print PR-AUC.
print(
    f"PR-AUC               : {pr_auc:.4f}"
)


# Print Matthews Correlation Coefficient.
print(
    f"MCC                  : {mcc:.4f}"
)


# Print Cohen's Kappa.
print(
    f"Cohen's Kappa        : {kappa:.4f}"
)


# Print Negative Predictive Value as a percentage.
print(
    f"NPV                  : {npv * 100:.2f}%"
)


# ------------------------------------------------------------
# PRINT CONFUSION MATRIX
# ------------------------------------------------------------

# Print the complete confusion matrix.
print("\nConfusion Matrix:")
print(cm)


# Print each confusion-matrix component individually.
print("\nConfusion Matrix Values:")

# True negatives.
print(f"True Negatives  (TN): {TN}")

# False positives.
print(f"False Positives (FP): {FP}")

# False negatives.
print(f"False Negatives (FN): {FN}")

# True positives.
print(f"True Positives  (TP): {TP}")


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

# Print detailed classification results.
print("\nClassification Report:")

print(
    classification_report(
        true_labels,                    # Actual labels.
        predicted_labels,               # Predicted labels.
        target_names=[
            "No Tumor",                 # Display name for class 0.
            "Tumor"                     # Display name for class 1.
        ],
        digits=4,                       # Show four decimal places.
        zero_division=0                 # Avoid division errors.
    )
)


# ------------------------------------------------------------
# SAVE METRICS TO CSV
# ------------------------------------------------------------

# Create a Pandas DataFrame containing all evaluation metrics.
metrics_df = pd.DataFrame({

    # Names of the metrics.
    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall / Sensitivity",
        "Specificity",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "MCC",
        "Cohen's Kappa",
        "NPV"
    ],

    # Corresponding numerical values.
    "Value": [
        accuracy,
        balanced_accuracy,
        precision,
        recall,
        specificity,
        f1,
        roc_auc,
        pr_auc,
        mcc,
        kappa,
        npv
    ]
})


# Save the metrics DataFrame as a CSV file.
metrics_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_evaluation.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# SAVE CONFUSION MATRIX VALUES
# ------------------------------------------------------------

# Create a DataFrame containing TN, FP, FN and TP.
cm_df = pd.DataFrame({

    # Names of confusion-matrix values.
    "Metric": [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive"
    ],

    # Actual values.
    "Value": [
        TN,
        FP,
        FN,
        TP
    ]
})


# Save confusion-matrix values as a CSV file.
cm_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix_values.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# CREATE CONFUSION MATRIX GRAPH
# ------------------------------------------------------------

# Create a new figure.
plt.figure(
    figsize=(7, 6)
)


# Display the confusion matrix as an image.
plt.imshow(
    cm,
    interpolation="nearest"
)


# Add graph title.
plt.title(
    "Confusion Matrix"
)


# Add a color scale.
plt.colorbar()


# Set predicted-class names on X-axis.
plt.xticks(
    [0, 1],
    ["No Tumor", "Tumor"]
)


# Set actual-class names on Y-axis.
plt.yticks(
    [0, 1],
    ["No Tumor", "Tumor"]
)


# Label the X-axis.
plt.xlabel("Predicted")


# Label the Y-axis.
plt.ylabel("Actual")


# Write each numerical confusion-matrix value
# inside its corresponding cell.
for i in range(2):

    # Loop through the two columns.
    for j in range(2):

        # Display the value inside the cell.
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


# Automatically adjust spacing.
plt.tight_layout()


# Save the confusion matrix image.
plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)


# Close the current graph.
plt.close()


# ------------------------------------------------------------
# CREATE ROC CURVE
# ------------------------------------------------------------

# Calculate the False Positive Rate and True Positive Rate
# for many different classification thresholds.
fpr, tpr, _ = roc_curve(
    true_labels,
    predicted_probabilities
)


# Create a new figure.
plt.figure(
    figsize=(8, 6)
)


# Plot the ROC curve.
plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)


# Add the random-classifier reference line.
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


# Label X-axis.
plt.xlabel(
    "False Positive Rate"
)


# Label Y-axis.
plt.ylabel(
    "True Positive Rate"
)


# Add title.
plt.title(
    "ROC Curve"
)


# Display the legend.
plt.legend()


# Add a light grid.
plt.grid(
    alpha=0.3
)


# Adjust layout.
plt.tight_layout()


# Save the ROC curve.
plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "roc_curve.png"
    ),
    dpi=300
)


# Close the graph.
plt.close()


# ------------------------------------------------------------
# CREATE PRECISION-RECALL CURVE
# ------------------------------------------------------------

# Calculate precision and recall values
# across different probability thresholds.
precision_curve, recall_curve, _ = precision_recall_curve(
    true_labels,
    predicted_probabilities
)


# Create a new figure.
plt.figure(
    figsize=(8, 6)
)


# Plot the Precision-Recall curve.
plt.plot(
    recall_curve,
    precision_curve,
    label=f"PR-AUC = {pr_auc:.4f}"
)


# Label the X-axis.
plt.xlabel(
    "Recall"
)


# Label the Y-axis.
plt.ylabel(
    "Precision"
)


# Add title.
plt.title(
    "Precision-Recall Curve"
)


# Display the legend.
plt.legend()


# Add grid.
plt.grid(
    alpha=0.3
)


# Adjust layout.
plt.tight_layout()


# Save the Precision-Recall curve.
plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "precision_recall_curve.png"
    ),
    dpi=300
)


# Close the graph.
plt.close()


# ------------------------------------------------------------
# CREATE IMAGE-LEVEL PREDICTION TABLE
# ------------------------------------------------------------

# Get the paths of all testing images.
#
# Because shuffle=False was used above, these paths are in
# the same order as true_labels and predicted_labels.
image_paths = test_dataset.file_paths


# Create a Pandas DataFrame containing every test prediction.
prediction_df = pd.DataFrame({

    # Original image location.
    "Image Path": image_paths,

    # Convert numeric actual labels to readable names.
    "Actual Class": [
        "No Tumor" if x == 0 else "Tumor"
        for x in true_labels
    ],

    # Convert numeric predictions to readable names.
    "Predicted Class": [
        "No Tumor" if x == 0 else "Tumor"
        for x in predicted_labels
    ],

    # Probability that the image contains a tumor.
    "Tumor Probability": predicted_probabilities,

    # Probability/confidence of the predicted class.
    #
    # If prediction is Tumor:
    #     confidence = tumor probability
    #
    # If prediction is No Tumor:
    #     confidence = 1 - tumor probability
    "Confidence": [
        p if y == 1 else 1 - p
        for p, y in zip(
            predicted_probabilities,
            predicted_labels
        )
    ],

    # True when actual and predicted labels are the same.
    "Correct": (
        true_labels == predicted_labels
    )
})


# Save all test-image predictions.
prediction_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# SAVE INCORRECT PREDICTIONS
# ------------------------------------------------------------

# Select only rows where the model made an incorrect prediction.
incorrect_df = prediction_df[
    prediction_df["Correct"] == False
]


# Save incorrect predictions separately.
incorrect_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "incorrect_predictions.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

# Count the number of correct predictions.
correct_predictions = prediction_df["Correct"].sum()


# Count the number of incorrect predictions.
incorrect_predictions = (
    len(prediction_df)
    - correct_predictions
)


# Print the final summary.
print("\nEvaluation completed successfully.")

print(
    f"Total test images      : {len(true_labels)}"
)

print(
    f"Correct predictions    : {correct_predictions}"
)

print(
    f"Incorrect predictions  : {incorrect_predictions}"
)

print(
    f"Results saved in       : {RESULTS_DIR}/"
)

