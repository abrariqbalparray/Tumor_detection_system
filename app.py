# ============================================================
# BRAIN MRI TUMOR DETECTION SYSTEM
# FLASK WEB APPLICATION
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

# Import os for environment variables and file-system settings.
import os

# Import Path for reliable project file and folder paths.
from pathlib import Path

# Force TensorFlow to use CPU.
# This is suitable for deployment on Render.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import TensorFlow for loading the trained model
# and making predictions.
import tensorflow as tf

# Import Flask components.
from flask import Flask, render_template, request

# Import secure_filename to safely handle uploaded filenames.
from werkzeug.utils import secure_filename


# ------------------------------------------------------------
# 2. PROJECT PATH
# ------------------------------------------------------------

# Get the directory containing this app.py file.
BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# 3. CREATE FLASK APPLICATION
# ------------------------------------------------------------

# Create the Flask application.
#
# Flask will use:
# templates/ -> HTML files
# static/    -> CSS and uploaded images
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)


# ------------------------------------------------------------
# 4. APPLICATION SETTINGS
# ------------------------------------------------------------

# Define the upload directory.
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

# Store the upload directory in Flask configuration.
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Allow uploaded files up to 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Create the upload directory if it does not exist.
UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 5. MODEL SETTINGS
# ------------------------------------------------------------

# Define the location of the trained model.
MODEL_PATH = BASE_DIR / "model" / "final_model.keras"

# Define the image size used by the trained model.
IMAGE_SIZE = (224, 224)


# ------------------------------------------------------------
# 6. CHECK MODEL FILE
# ------------------------------------------------------------

# Check whether the trained model exists.
if not MODEL_PATH.exists():

    # Stop the application if the model cannot be found.
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


# ------------------------------------------------------------
# 7. LOAD TRAINED MODEL
# ------------------------------------------------------------

# Display a message while the model is loading.
print("Loading trained model...")

# Load the trained MobileNetV2 model.
#
# custom_objects provides the MobileNetV2 preprocessing
# function required by the saved model.
#
# compile=False is used because the application only
# performs predictions.
#
# safe_mode=False allows the saved Keras model to load
# its custom preprocessing component.
model = tf.keras.models.load_model(
    str(MODEL_PATH),
    custom_objects={
        "preprocess_input":
            tf.keras.applications.mobilenet_v2.preprocess_input
    },
    compile=False,
    safe_mode=False
)

# Confirm that the model loaded successfully.
print("Model loaded successfully.")


# ------------------------------------------------------------
# 8. ALLOWED IMAGE FORMATS
# ------------------------------------------------------------

# Define the image formats accepted by the application.
ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


# ------------------------------------------------------------
# 9. FILE VALIDATION FUNCTION
# ------------------------------------------------------------

# Check whether an uploaded filename has an allowed extension.
def allowed_file(filename):

    # Return True if the filename contains
    # a supported image extension.
    return (
        "." in filename
        and
        filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ------------------------------------------------------------
# 10. HOME PAGE
# ------------------------------------------------------------

# Connect the root URL to index.html.
@app.route("/")
def home():

    # Display the existing index.html file.
    return render_template("index.html")


# ------------------------------------------------------------
# 11. PREDICTION ROUTE
# ------------------------------------------------------------

# This route receives the MRI image uploaded
# from index.html.
@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Check whether a file was included in the request.
    if "file" not in request.files:

        # Return to the homepage with an error.
        return render_template(
            "index.html",
            error="Please select an MRI image."
        )


    # Get the uploaded file.
    file = request.files["file"]


    # Check whether the user actually selected a file.
    if file.filename == "":

        # Return an error if no file was selected.
        return render_template(
            "index.html",
            error="Please select an MRI image."
        )


    # Check whether the uploaded file is an allowed format.
    if not allowed_file(file.filename):

        # Return an error for unsupported file types.
        return render_template(
            "index.html",
            error="Only JPG, JPEG and PNG images are allowed."
        )


    # Convert the filename into a safe filename.
    filename = secure_filename(file.filename)


    # Create the complete path for the uploaded image.
    filepath = UPLOAD_FOLDER / filename


    # Save the uploaded MRI image.
    file.save(str(filepath))


    # --------------------------------------------------------
    # 12. LOAD IMAGE
    # --------------------------------------------------------

    # Load the uploaded image and resize it to
    # the same dimensions used during model training.
    image = tf.keras.utils.load_img(
        str(filepath),
        target_size=IMAGE_SIZE
    )


    # Convert the image into a numerical array.
    image_array = tf.keras.utils.img_to_array(
        image
    )


    # Add a batch dimension.
    #
    # Before:
    # 224 x 224 x 3
    #
    # After:
    # 1 x 224 x 224 x 3
    image_array = tf.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # 13. MAKE MODEL PREDICTION
    # --------------------------------------------------------

    # Send the image to the trained model.
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]


    # Convert the prediction to a Python float.
    prediction = float(prediction)


    # --------------------------------------------------------
    # 14. CLASSIFY IMAGE
    # --------------------------------------------------------

    # Use 0.50 as the classification threshold.
    if prediction >= 0.50:

        # The model predicts tumor.
        result = "Tumor Present"

        # Calculate confidence for the tumor prediction.
        confidence = prediction * 100

    else:

        # The model predicts no tumor.
        result = "No Tumor"

        # Calculate confidence for the no-tumor prediction.
        confidence = (1 - prediction) * 100


    # Calculate the tumor probability.
    tumor_probability = prediction * 100


    # --------------------------------------------------------
    # 15. IMAGE URL
    # --------------------------------------------------------

    # Create the URL used by result.html to display
    # the uploaded MRI image.
    image_path = "/static/uploads/" + filename


    # --------------------------------------------------------
    # 16. SEND RESULT TO result.html
    # --------------------------------------------------------

    # Render the existing result.html page.
    return render_template(
        "result.html",

        # Send the prediction result.
        result=result,

        # Send the confidence percentage.
        confidence=round(
            confidence,
            2
        ),

        # Send the tumor probability.
        tumor_probability=round(
            tumor_probability,
            2
        ),

        # Send the uploaded image URL.
        image_path=image_path
    )


# ------------------------------------------------------------
# 17. ABOUT PAGE
# ------------------------------------------------------------

# Connect /about to about.html.
@app.route("/about")
def about():

    # Display the existing about.html page.
    return render_template("about.html")


# ------------------------------------------------------------
# 18. HEALTH CHECK
# ------------------------------------------------------------

# Create a simple health-check endpoint.
@app.route("/health")
def health():

    # Return a simple response.
    return "Brain MRI Tumor Detection System is running."


# ------------------------------------------------------------
# 19. FILE TOO LARGE ERROR
# ------------------------------------------------------------

# Handle files larger than the 10 MB limit.
@app.errorhandler(413)
def file_too_large(error):

    # Return to the homepage with an error message.
    return render_template(
        "index.html",
        error="File is too large. Maximum size is 10 MB."
    ), 413


# ------------------------------------------------------------
# 20. GENERAL SERVER ERROR
# ------------------------------------------------------------

# Handle unexpected application errors.
@app.errorhandler(500)
def internal_error(error):

    # Return to the homepage with an error message.
    return render_template(
        "index.html",
        error="An error occurred while processing the MRI image."
    ), 500


# ------------------------------------------------------------
# 21. START FLASK SERVER
# ------------------------------------------------------------

# This section runs only when app.py is executed directly.
if __name__ == "__main__":

    # Render provides a PORT environment variable.
    #
    # When running locally, use port 5001.
    port = int(
        os.environ.get(
            "PORT",
            5001
        )
    )

    # Start the Flask server.
    #
    # 0.0.0.0 allows the application to receive
    # connections from Render.
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )