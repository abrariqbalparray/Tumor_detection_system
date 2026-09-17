# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Brain Tumor Detection Project
# Tumor Present vs No Tumor
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------------------------

# Import Pandas for reading and analyzing our CSV information.
import pandas as pd

# Import Matplotlib for displaying and saving graphs.
import matplotlib.pyplot as plt

# Import PIL Image for opening MRI images.
from PIL import Image

# Import os for working with image file paths.
import os


# ------------------------------------------------------------
# STEP 2: LOAD THE IMAGE INFORMATION
# ------------------------------------------------------------

# Load the CSV file created by data_preprocessing.py.
image_df = pd.read_csv(
    "results/image_information.csv"
)


# ------------------------------------------------------------
# STEP 3: DISPLAY BASIC INFORMATION
# ------------------------------------------------------------

# Display the number of image records.
print("\n========================================")
print("IMAGE DATASET INFORMATION")
print("========================================")

print("\nTotal valid images:")
print(len(image_df))

print("\nColumns:")
print(image_df.columns.tolist())

print("\nFirst 5 records:")
print(image_df.head())


# ------------------------------------------------------------
# STEP 4: CLASS DISTRIBUTION
# ------------------------------------------------------------

# Count the number of images in each class.
class_counts = (
    image_df["Class"]
    .value_counts()
)


# Display the class counts.
print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

print(class_counts)


# ------------------------------------------------------------
# STEP 5: CLASS DISTRIBUTION GRAPH
# ------------------------------------------------------------

# Create a Pandas bar chart using the class counts.
class_counts.plot(
    kind="bar",
    figsize=(8, 5)
)


# Add a title.
plt.title(
    "Tumor vs No Tumor Image Distribution"
)


# Label the x-axis.
plt.xlabel(
    "Class"
)


# Label the y-axis.
plt.ylabel(
    "Number of Images"
)


# Keep the class names horizontal.
plt.xticks(
    rotation=0
)


# Adjust the graph layout.
plt.tight_layout()


# Save the graph.
plt.savefig(
    "results/class_distribution.png"
)


# Display the graph.
plt.show()


# ------------------------------------------------------------
# STEP 6: IMAGE SIZE ANALYSIS
# ------------------------------------------------------------

# Display the most common image dimensions.
size_counts = (
    image_df
    .groupby(
        ["Width", "Height"]
    )
    .size()
    .reset_index(
        name="Image_Count"
    )
    .sort_values(
        by="Image_Count",
        ascending=False
    )
)


# Display the ten most common image sizes.
print("\n========================================")
print("MOST COMMON IMAGE SIZES")
print("========================================")

print(
    size_counts.head(10)
)


# ------------------------------------------------------------
# STEP 7: IMAGE WIDTH DISTRIBUTION
# ------------------------------------------------------------

# Create a histogram of image widths.
image_df["Width"].plot(
    kind="hist",
    bins=30,
    figsize=(8, 5)
)


# Add a title.
plt.title(
    "Distribution of MRI Image Widths"
)


# Label the x-axis.
plt.xlabel(
    "Image Width"
)


# Label the y-axis.
plt.ylabel(
    "Number of Images"
)


# Adjust layout.
plt.tight_layout()


# Save the graph.
plt.savefig(
    "results/image_width_distribution.png"
)


# Display the graph.
plt.show()


# ------------------------------------------------------------
# STEP 8: IMAGE HEIGHT DISTRIBUTION
# ------------------------------------------------------------

# Create a histogram of image heights.
image_df["Height"].plot(
    kind="hist",
    bins=30,
    figsize=(8, 5)
)


# Add a title.
plt.title(
    "Distribution of MRI Image Heights"
)


# Label the x-axis.
plt.xlabel(
    "Image Height"
)


# Label the y-axis.
plt.ylabel(
    "Number of Images"
)


# Adjust layout.
plt.tight_layout()


# Save the graph.
plt.savefig(
    "results/image_height_distribution.png"
)


# Display the graph.
plt.show()


# ------------------------------------------------------------
# STEP 9: DISPLAY SAMPLE TUMOR IMAGES
# ------------------------------------------------------------

# Select several Tumor images from the image information table.
tumor_samples = image_df[
    image_df["Class"] == "Tumor"
].head(5)


# Create a figure containing five sample images.
plt.figure(
    figsize=(15, 8)
)


# Loop through each selected Tumor image.
for index, (_, row) in enumerate(
    tumor_samples.iterrows()
):

    # Create the complete path to the image.
    image_path = os.path.join(
        "dataset",
        row["Dataset"],
        row["Class"],
        row["File_Name"]
    )


    # Open the MRI image.
    image = Image.open(
        image_path
    )


    # Create the position for this image.
    plt.subplot(
        1,
        5,
        index + 1
    )


    # Display the image in grayscale.
    plt.imshow(
        image,
        cmap="gray"
    )


    # Add a title to the image.
    plt.title(
        "Tumor"
    )


    # Remove axis markings.
    plt.axis(
        "off"
    )


    # Close the image after displaying it.
    image.close()


# Add an overall title.
plt.suptitle(
    "Sample MRI Images - Tumor"
)


# Adjust the layout.
plt.tight_layout()


# Save the Tumor sample graph.
plt.savefig(
    "results/tumor_samples.png"
)


# Display the graph.
plt.show()


# ------------------------------------------------------------
# STEP 10: DISPLAY SAMPLE NO TUMOR IMAGES
# ------------------------------------------------------------

# Select several No Tumor images.
no_tumor_samples = image_df[
    image_df["Class"] == "No Tumor"
].head(5)


# Create a figure for the No Tumor images.
plt.figure(
    figsize=(15, 8)
)


# Loop through the selected images.
for index, (_, row) in enumerate(
    no_tumor_samples.iterrows()
):

    # Create the complete image path.
    image_path = os.path.join(
        "dataset",
        row["Dataset"],
        row["Class"],
        row["File_Name"]
    )


    # Open the MRI image.
    image = Image.open(
        image_path
    )


    # Create the subplot location.
    plt.subplot(
        1,
        5,
        index + 1
    )


    # Display the image in grayscale.
    plt.imshow(
        image,
        cmap="gray"
    )


    # Add the class title.
    plt.title(
        "No Tumor"
    )


    # Remove axes.
    plt.axis(
        "off"
    )


    # Close the image.
    image.close()


# Add an overall title.
plt.suptitle(
    "Sample MRI Images - No Tumor"
)


# Adjust layout.
plt.tight_layout()


# Save the No Tumor sample graph.
plt.savefig(
    "results/no_tumor_samples.png"
)


# Display the graph.
plt.show()


# ------------------------------------------------------------
# STEP 11: SAVE IMAGE SIZE SUMMARY
# ------------------------------------------------------------

# Save the image-size distribution table.
size_counts.to_csv(
    "results/image_size_distribution.csv",
    index=False
)


# ------------------------------------------------------------
# STEP 12: FINISH
# ------------------------------------------------------------

# Display a completion message.
print("\n========================================")
print("EDA COMPLETED SUCCESSFULLY")
print("========================================")

# Display the location of the generated results.
print("\nEDA results are saved inside the results folder.")