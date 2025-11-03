import torch
import json
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from Projection_layer.Projection_layer import ProjectionLayer

# Set up the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to EOS (end of sequence)

# Load the GPT-2 model and move it to the selected device
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.load_state_dict(torch.load("gpt2_model5.pth", map_location=device))  # Load fine-tuned weights
model.eval()  # Set the model to evaluation mode

# Load the Projection Layer to map CNN embeddings into GPT-2's latent space
projection_layer = ProjectionLayer(768, model.config.n_embd, device).to(device)  # Load fine-tuned weights
projection_layer.load_state_dict(torch.load("projection5.pth", map_location=device))
projection_layer.eval()  # Set to evaluation mode

# Load the precomputed monster embeddings from a JSON file
with open("monsters_embeddings.json", "r") as f:
    data = json.load(f)

# Randomly select a monster from the dataset
monster_name = np.random.choice(list(data.keys()))
monster_embedding = torch.tensor(data[monster_name]["embedding"]).to(device)

# Define a range of noise levels to test on the embeddings
noise_levels = [0.0, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5]  # Different perturbation levels

print(f"\nTesting noise on the monster embedding: {monster_name}\n")

# Loop to test different noise levels on the embeddings
for noise in noise_levels:
    # Add Gaussian noise to the original embedding
    noisy_embedding = monster_embedding + torch.randn_like(monster_embedding) * noise
    # Project the noisy embedding into GPT-2's input space
    projected_embedding = projection_layer(noisy_embedding).unsqueeze(0).unsqueeze(1)

    # Define the initial prompt for the model
    prompt_text = "Detailed description of the monster:"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_embeds = model.transformer.wte(prompt_ids)  # Convert token ids into GPT-2 embeddings

    # Add the special token embedding (EOS)
    special_token_embed = model.transformer.wte(torch.tensor([50256], device=device)).unsqueeze(0)

    # Concatenate embeddings:
    # 1. Projected image embedding
    # 2. Special token embedding
    # 3. Prompt text embedding
    inputs_embeds = torch.cat([projected_embedding, special_token_embed, prompt_embeds], dim=1)

    # Create an attention mask to indicate valid tokens
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    # Generate text with GPT-2 using embeddings instead of tokenized inputs
    output = model.generate(
        inputs_embeds=inputs_embeds,    # The concatenated embeddings
        attention_mask=attention_mask,  # The attention mask for the model
        max_length=200,  # Maximum length of generated text
        pad_token_id=tokenizer.eos_token_id  # Set padding token as EOS
    )
    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print the result for the current noise level
    print(f"\n🔹 **Noise Level {noise}:**")
    print(generated_text)
