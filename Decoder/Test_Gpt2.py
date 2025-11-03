import json
import torch
from nltk.translate.bleu_score import sentence_bleu
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from Projection_layer.Projection_layer import ProjectionLayer
from torchvision import transforms

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load monster data with names, descriptions and embeddings
with open("monsters_embeddings.json", "r") as f:
    data = json.load(f)

test_transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

torch.cuda.manual_seed(42)  # Set a fixed seed for CUDA operations to ensure reproducibility

# Load the pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to EOS

# Load the pre-trained GPT-2 model
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.to(device)
gpt2_model.load_state_dict(torch.load("gpt2_model5.pth", map_location=device))

# Define embedding dimensions
embedding_dim = 768  # Dimension of visual embeddings

# Retrieve GPT-2 embedding dimension
gpt2_embedding_dim = gpt2_model.config.n_embd

# Initialize projection layer to map encoder output to GPT-2 input format
proj_layer = ProjectionLayer(embedding_dim, gpt2_embedding_dim, device)
proj_layer.load_state_dict(torch.load("projection5.pth", map_location=device))


# Function to generate a monster description based on embeddings
def generate_description(model, projection_layer, tokenizer, device, embedding,
                         max_new_tokens=1000):
    gpt2_model.eval()  # Set GPT-2 model to evaluation mode
    proj_layer.eval()  # Set projection layer to evaluation mode

    with torch.no_grad():  # Disable gradient computation for inference
        # Convert embedding to a tensor and move it to the device
        embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 768)

        # Pass the embedding through the Projection Layer
        projected_embeddings = projection_layer(embedding).unsqueeze(1)  # (1, 1, 768)

        # Create a text prompt using the monster's name then transform it into the embedding space
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

        # Create an attention mask indicating which tokens should be attended to by the model
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
            do_sample=True,  # Enable sampling to generate varied outputs
            top_k=40,  # Take into account k different tokens for sampling
            top_p=0.95  # Keep the sum of the propability of the tokens to sample >= 0.95

        )

    # Decode and return the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Function to compute BLEU score for evaluation
def calculate_bleu(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    return sentence_bleu([reference_tokens], candidate_tokens)


# Variables for BLEU score tracking
total_bleu = 0
num_monsters = 0
guessed = 0
output_file = "generated_descriptions.txt"  # Create a text file to save results and descriptions generated
bleu_threshold = 0.4  # Minimum BLEU score to consider a good match

# Generate descriptions and evaluate BLEU scores
with torch.no_grad():
    with open(output_file, "w", encoding="utf-8") as f:
        for monster_name, monster_data in data.items():
            test_embedding = monster_data['embedding']
            real_description = monster_data['description']  # Ground truth description

            # Generate a description using the model
            generated_description = generate_description(gpt2_model, proj_layer, tokenizer, device,
                                                         test_embedding)

            # Compute BLEU score
            bleu_score = calculate_bleu(real_description, generated_description)
            if bleu_score >= bleu_threshold:
                guessed += 1
                print("\n" + "-" * 80 + "\n")
                print(f"Generated description for {monster_name}:\n")
                print(generated_description + "\n")
                print(real_description + "\n")
                print(f"BLEU score: {bleu_score:.4f}\n")
                print("\n" + "-" * 80 + "\n")
            total_bleu += bleu_score
            num_monsters += 1
            print(
                f"{guessed}/{num_monsters} monsters have reached bleu_score >= {bleu_threshold}; Current Monster: {monster_name} with Bleu Score: {bleu_score} . \n")
            # Write results to file
            f.write(f"Generated description for {monster_name}:\n")
            f.write(generated_description + "\n\n")
            f.write(real_description + "\n\n")
            f.write(f"BLEU score: {bleu_score:.4f}\n")
            f.write("\n" + "-" * 80 + "\n")

    # Compute the percentage of successful matches
    guessed_rate = guessed / num_monsters * 100
    print(f"{guessed_rate:.1f}% ({guessed}/{num_monsters})  monsters have reached bleu_score >= {bleu_threshold}. \n")
    # Calcolo BLEU medio

    # Compute the average BLEU score across all monsters
    if num_monsters > 0:
        avg_bleu = total_bleu / num_monsters
        print(f"Average BLEU score across all monsters: {avg_bleu:.4f}\n")
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(
                f"{guessed_rate:.1f}% ({guessed}/{num_monsters})  monsters have reached bleu_score >= {bleu_threshold}. \n")
            f.write(f"Average BLEU score across all monsters: {avg_bleu:.4f}\n")
    else:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("No monsters found for evaluation.\n")

    print(f"Results saved to {output_file}")
