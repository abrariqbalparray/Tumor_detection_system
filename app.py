
# Import os for working with folders and file paths.
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Import TensorFlow to load and use the trained model.
import tensorflow as tf

# Import Flask components for creating the web application.
from flask import Flask, render_template, request

# Import secure_filename to safely handle uploaded filenames.
from werkzeug.utils import secure_filename


# ------------------------------------------------------------
# CREATE FLASK APPLICATION
# ------------------------------------------------------------

# Create the Flask application object.
app = Flask(__name__)


# ------------------------------------------------------------
# APPLICATION SETTINGS
# ------------------------------------------------------------

# Folder where uploaded MRI images will be stored.
UPLOAD_FOLDER = "static/uploads"

# Tell Flask where uploaded files should be saved.
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size: 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# Create the upload directory if it does not already exist.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

# Location of the trained model.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "final_model.keras"
)

# Model expects images with this size.
IMAGE_SIZE = (224, 224)


# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------

# Load the trained MobileNetV2 model.
#
# custom_objects provides the MobileNetV2 preprocess_input
# function used by the Lambda layer in the saved model.
#
# compile=False is used because this application only needs
# the model for prediction.
#
# safe_mode=False allows the Lambda layer to be loaded.
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
# ALLOWED IMAGE TYPES
# ------------------------------------------------------------

# These are the image formats accepted by the website.
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


# ------------------------------------------------------------
# CHECK FILE EXTENSION
# ------------------------------------------------------------

# Create a function that checks whether the uploaded file
# has an allowed image extension.
def allowed_file(filename):

    # Check that the filename contains a dot and that the
    # extension is one of our allowed image formats.
    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------

# This route handles the main page of the website.
@app.route("/")
def home():

    # Display the index.html page.
    return render_template("index.html")


# ------------------------------------------------------------
# PREDICTION ROUTE
# ------------------------------------------------------------

# This route receives the uploaded MRI image.
@app.route("/predict", methods=["POST"])
def predict():

    # Check whether the browser actually sent a file.
    if "file" not in request.files:

        # Return to the home page with an error message.
        return render_template(
            "index.html",
            error="Please select an MRI image."
        )


    # Get the uploaded file.
    file = request.files["file"]


    # Check whether the user selected a file.
    if file.filename == "":

        # Display an error if no file was selected.
        return render_template(
            "index.html",
            error="Please select an MRI image."
        )


    # Check whether the file is a supported image format.
    if not allowed_file(file.filename):

        # Reject unsupported formats.
        return render_template(
            "index.html",
            error="Only JPG, JPEG and PNG images are allowed."
        )


    # Make the filename safe before saving it.
    filename = secure_filename(file.filename)


    # Build the complete path where the image will be saved.
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # Save the uploaded image.
    file.save(filepath)


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    # Load the uploaded image and resize it to 224 x 224.
    image = tf.keras.utils.load_img(
        filepath,
        target_size=IMAGE_SIZE
    )


    # Convert the image into a NumPy array.
    image_array = tf.keras.utils.img_to_array(
        image
    )


    # Add a batch dimension.
    #
    # Shape changes from:
    # (224, 224, 3)
    #
    # to:
    # (1, 224, 224, 3)
    image_array = tf.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # GENERATE PREDICTION
    # --------------------------------------------------------

    # Ask the model to predict whether the image contains
    # a tumor.
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]


    # --------------------------------------------------------
    # CONVERT PROBABILITY TO RESULT
    # --------------------------------------------------------

    # Use 0.50 as the classification threshold.
    if prediction >= 0.50:

        # The model predicts that a tumor is present.
        result = "Tumor Present"

        # Tumor probability becomes the confidence.
        confidence = prediction * 100

    else:

        # The model predicts no tumor.
        result = "No Tumor"

        # Convert tumor probability into no-tumor confidence.
        confidence = (1 - prediction) * 100


    # Convert tumor probability to percentage.
    tumor_probability = prediction * 100


    # --------------------------------------------------------
    # DISPLAY RESULT PAGE
    # --------------------------------------------------------

    # Send the prediction information to result.html.
    return render_template(
        "result.html",

        # Result shown to the user.
        result=result,

        # Confidence percentage.
        confidence=round(confidence, 2),

        # Tumor probability percentage.
        tumor_probability=round(
            tumor_probability,
            2
        ),

        # Path used by the browser to display the uploaded image.
        image_path="/" + filepath.replace("\\", "/")
    )


# ------------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------------

# Route for the project information page.
@app.route("/about")
def about():

    # Display about.html.
    return render_template("about.html")


# ------------------------------------------------------------
# RUN FLASK APPLICATION
# ------------------------------------------------------------

# This ensures the application starts only when app.py
# is executed directly.

