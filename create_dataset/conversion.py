from PIL import Image
import os


# Function to convert PNG images to JPEG ones with white background
def convert_png_to_jpeg(input_folder, output_folder, background_color=(255, 255, 255)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through files in the input folder
    for filename in os.listdir(input_folder):
        # Process only PNG files
        if filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)

            # Open the image and convert it to RGBA mode to handle transparency
            img = Image.open(img_path).convert("RGBA")

            # Create a new white background image of the same size
            background = Image.new("RGB", img.size, background_color)

            # Composite the original image onto the white background
            background.paste(img, (0, 0), img)

            # Generate the new filename with a .jpg extension
            new_filename = os.path.splitext(filename)[0] + ".jpg"

            # Save the converted image as a JPEG with high quality
            background.save(os.path.join(output_folder, new_filename), "JPEG", quality=95)

    print("Conversione completata con sfondo bianco!")


# Define input and output folders for monster images
input_folder = "./monster_image"
output_folder = "./monster_image"

# Execute the conversion function
convert_png_to_jpeg(input_folder, output_folder)
