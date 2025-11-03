import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from Projection_layer.Projection_layer import ProjectionLayer
from CNN.CNN import CNN
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True


# Function to convert PNG images to JPEG ones with white background
def convert_png_to_jpeg(input_folder, output_folder, background_color=(255, 255, 255)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through files in the input folder
    for filename in os.listdir(input_folder):
        # Process only PNG files
        if filename.endswith((".png", ".webp", "jpg", "jpeg")):
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

    print("Conversion completed with white background!")


# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation pipeline for input images
test_transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Set a fixed seed for CUDA operations to ensure reproducibility
torch.cuda.manual_seed(42)

# Load the pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to EOS

# Load the pre-trained GPT-2 model
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.to(device)
# Load pre-trained weights for the GPT-2 model
gpt2_model.load_state_dict(torch.load("gpt2_model5.pth", map_location=device))

# Define embedding dimensions
embedding_dim = 768  # Dimension of visual embeddings

# Retrieve GPT-2 embedding dimension
gpt2_embedding_dim = gpt2_model.config.n_embd

# Initialize projection layer to map CNN outputs to GPT-2 input format
proj_layer = ProjectionLayer(embedding_dim, gpt2_embedding_dim, device)

# Load pre-trained weights for the projection layer
proj_layer.load_state_dict(torch.load("projection5.pth", map_location=device))


# Function to generate a monster description based on embeddings
def generate_description(model, projection_layer, tokenizer, device, embedding, max_new_tokens=900):
    gpt2_model.eval()  # Set GPT-2 model to evaluation mode
    proj_layer.eval()  # Set projection layer to evaluation mode

    with torch.no_grad():  # Disable gradient computation for inference
        # Convert embedding to a tensor and move it to the device
        img_embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 768)

        # Pass the embedding through the Transformer Encoder
        projected_embeddings = projection_layer(img_embedding).unsqueeze(1)  # (1, 1, 768)

        # Create a text prompt then tokenize and transform it into the embedding space
        prompt_text = f"Detailed description for:"
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)  # (1, N)
        prompt_embeds = model.transformer.wte(prompt_ids)  # (1, N, 768)

        # Create an embedding for the special token
        special_token_embed = model.transformer.wte(
            torch.tensor([50256], device=device)
        ).unsqueeze(0).expand(1, 1, -1)  # (1, 1, 768)

        # Concatenate projected embeddings, special token, and prompt embeddings
        combined_embeds = torch.cat([projected_embeddings, special_token_embed, prompt_embeds],
                                    dim=1)  # (1, N+2, 768)

        # Create an attention mask indicating which tokens should be taken to account by the model
        # All values are 1, meaning the model should pay attention to all tokens
        attention_mask = torch.ones(combined_embeds.shape[:2], dtype=torch.long).to(device)  # (1, N+2)

        # Generate text using GPT-2
        output = model.generate(
            inputs_embeds=combined_embeds,  # The concatenated embeddings
            attention_mask=attention_mask,  # The attention mask for the model
            pad_token_id=gpt2_model.config.eos_token_id,  # Set padding token to EOS
            max_length=combined_embeds.shape[1] + max_new_tokens,  # Maximum length of the generated sequence
            num_return_sequences=1,  # Generate a single sequence
            min_new_tokens=50,  # Ensure at least 50 new tokens are generated
            no_repeat_ngram_size=2,  # Prevent repeating n-grams of size 2
            do_sample=False  # Disable sampling to generate varied outputs

        )

    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Load the CNN model
cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load("../CNN/cnn_model_fold_3.pth", map_location=device))
cnn_model.eval()  # Set model to evaluation mode

# Define input and output directories for image processing
folder_input = "./test_image"
folder_output = "./test_image/Single_image_test"

# Convert images to JPEG format
convert_png_to_jpeg(folder_input, folder_output)

# Get the list of image files from the processed directory
images_path = folder_output
image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Iterate over each image
for image_name in image_files:
    image_path = os.path.join(images_path, image_name)

    # Open the image and convert it to RGB
    image_rgb = Image.open(image_path).convert("RGB")

    # Apply the transformation pipeline
    transformed_image = test_transform(image_rgb).unsqueeze(0).to(device)

    # Extract the embedding from the CNN
    embedding = cnn_model(transformed_image)[1].squeeze(0)

    # Generate the description using GPT-2
    generated_description = generate_description(gpt2_model, proj_layer, tokenizer, device, embedding)

    # Print the generated description
    print(f"Generated description for {image_name.split(".")[0]}:\n{generated_description}\n")
