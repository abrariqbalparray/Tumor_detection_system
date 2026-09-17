
# ============================================================
# BRAIN MRI TUMOR DETECTION - SINGLE IMAGE PREDICTION
# ============================================================


# Import os for checking whether the image file exists.
import os

# Import sys so we can receive the image path from the terminal.
import sys

# Import NumPy for numerical array operations.
import numpy as np

# Import TensorFlow for loading and using the trained model.
import tensorflow as tf

# Import functions for loading an image and converting it
# into a numerical array that the model can understand.
from tensorflow.keras.utils import load_img, img_to_array


# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

# Location of the trained model.
MODEL_PATH = "model/final_model.keras"

# The image size expected by the trained MobileNetV2 model.
IMAGE_SIZE = (224, 224)


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

# Load the previously trained model from the model folder.
#
# custom_objects provides the MobileNetV2 preprocess_input
# function because the saved model contains a Lambda layer
# that uses this function.
#
# compile=False means we only need the trained model for
# prediction and do not need to restore the optimizer.
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


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------

# Define a function that accepts the path of one MRI image.
def predict_image(image_path):

    # Check whether the image actually exists.
    if not os.path.exists(image_path):

        # Stop the program and show a useful error message
        # if the file cannot be found.
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    # Load the MRI image from the supplied path.
    #
    # target_size resizes the image to 224 x 224 because
    # this is the input size used during model training.
    image = load_img(
        image_path,
        target_size=IMAGE_SIZE
    )


    # --------------------------------------------------------
    # CONVERT IMAGE TO NUMPY ARRAY
    # --------------------------------------------------------

    # Convert the loaded image into a numerical array.
    #
    # A neural network cannot directly process a normal
    # image object, so we convert it into pixel values.
    image_array = img_to_array(image)


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    # Add an extra dimension at the beginning.
    #
    # Before:
    #     (224, 224, 3)
    #
    # After:
    #     (1, 224, 224, 3)
    #
    # The model expects images in batches, even when only
    # one image is being predicted.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # GENERATE PREDICTION
    # --------------------------------------------------------

    # Send the image to the trained model.
    #
    # The output of our final sigmoid neuron is a value
    # between 0 and 1.
    #
    # A value close to 1 means the model considers the image
    # more likely to belong to the Tumor class.
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]


    # --------------------------------------------------------
    # CONVERT PROBABILITY TO FINAL CLASS
    # --------------------------------------------------------

    # Use 0.50 as the classification threshold.
    #
    # If probability is 0.50 or higher:
    #     Tumor Present
    #
    # If probability is below 0.50:
    #     No Tumor
    if prediction >= 0.5:

        # Set the final prediction to Tumor Present.
        result = "Tumor Present"

        # For a tumor prediction, the confidence is the
        # tumor probability itself.
        confidence = prediction * 100

    else:

        # Set the final prediction to No Tumor.
        result = "No Tumor"

        # For a no-tumor prediction, confidence is the
        # probability of the No Tumor class.
        confidence = (1 - prediction) * 100


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    # Print a blank line for cleaner terminal output.
    print()

    # Print a heading.
    print("Prediction Result")

    # Print a separator line.
    print("=" * 40)

    # Print the final predicted class.
    print(
        f"Result     : {result}"
    )

    # Print the confidence of the prediction.
    print(
        f"Confidence : {confidence:.2f}%"
    )

    # Print the raw tumor probability.
    print(
        f"Tumor Probability : {prediction * 100:.2f}%"
    )


# ------------------------------------------------------------
# PROGRAM ENTRY POINT
# ------------------------------------------------------------

# This condition ensures the following code runs only when
# this file is executed directly from the terminal.
#
# It will not run automatically if predict.py is imported
# into another Python program.
if __name__ == "__main__":

    # Check whether the user provided an image path.
    #
    # sys.argv[0] is the name of this Python file.
    # sys.argv[1] should contain the image path.
    if len(sys.argv) < 2:

        # Show the correct terminal command if no image
        # path was supplied.
        print(
            "Usage: python predict.py <image_path>"
        )

        # Exit the program because there is no image to test.
        sys.exit(1)


    # Get the image path supplied after the Python command.
    image_path = sys.argv[1]


    # Send the image path to the prediction function.
    predict_image(image_path)

