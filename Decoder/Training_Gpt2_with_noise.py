import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.model_selection import StratifiedKFold
from Projection_layer.Projection_layer import ProjectionLayer
from Monster_dataset.Monster_dataset import MonsterDataset
import random
import numpy as np

# Load monster data with names, descriptions, and embeddings
with open("monsters_embeddings.json", "r") as f:
    data = json.load(f)

# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
lr = 5e-4  # Learning rate
max_length = 550  # Maximum token length for input sequences
batch_size = 3  # Number of samples per batch
epochs = 40  # Total number of training epochs
patience_es = 5  # Early stopping patience
k_folds = 5  # Number of folds for cross-validation

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to GPT-2's end-of-sequence token

expanded_data = []  # List to store augmented embeddings
noise_levels = torch.linspace(0.0, 0.03, steps=10)  # 10 amount of noise from 0.0 to 0.03

# Data augmentation: adding noise to embeddings
for monster_name, monster_info in data.items():  # Iterate through monsters
    original_embedding = torch.tensor(monster_info["embedding"])  # Load original embedding

    for noise in noise_levels:  # Create 10 noisy variations per monster
        noisy_embedding = original_embedding + torch.randn_like(original_embedding) * noise  # Add noise

        expanded_data.append({
            "name": monster_info["name"],  # Monster name
            "description": monster_info["description"],  # Monster description
            "embedding": noisy_embedding.tolist(),  # Convert tensor back to list for JSON compatibility
            "label": monster_name,  # Use monster name as label for stratified split
            "noise_level": noise.item()  # Keep the noise amount level
        })

# Create dataset with augmented embeddings
dataset = MonsterDataset(expanded_data, tokenizer, max_length=max_length)

# Create labels for stratified split
labels = [d["label"] for d in expanded_data]

# Use StratifiedKFold to maintain class balance across folds
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

# Variables to track the best fold
best_fold = None
best_fold_loss = float('inf')


# Train the GPT-2 model with projected embeddings and perform validation.
def train(model, projection_layer, train_loader, val_loader, optimizer, scheduler, device, tokenizer, epochs, fold):
    global best_fold, best_fold_loss

    best_val_loss = float('inf')
    epochs_no_improve = 0  # Counter for early stopping

    # Loop over the number of epochs
    for epoch in range(epochs):
        print(f"Fold {fold + 1} - Epoch {epoch + 1}/{epochs}")

        model.train()  # Set the model to training mode
        projection_layer.train()  # Set the projection layer to training mode
        train_loss = 0  # Initialize total training loss to zero

        for batch in train_loader:  # Iterate through each batch in the training data loader

            embeddings, target_input_ids, _, _ = zip(*batch)  # Unpack the batch

            embeddings = torch.stack(embeddings).to(device)  # Stack the embeddings

            target_input_ids = torch.stack(target_input_ids).to(device)  # Stack the target input IDs

            optimizer.zero_grad()  # Reset gradients before the backward pass

            # Project CNN embeddings to increase the expressiveness
            projected_embeddings = projection_layer(embeddings).unsqueeze(1)

            # Tokenize the prompt text and convert to input IDs
            prompt_ids = tokenizer("Detailed description of the monster:", return_tensors="pt").input_ids.to(device)

            # Transform IDs into GPT-2 embeddings
            prompt_embeds = model.transformer.wte(prompt_ids).expand(embeddings.size(0), -1, -1)

            # Add the special token (EOS)
            special_token_embed = model.transformer.wte(torch.tensor([50256], device=device)).expand(embeddings.size(0),
                                                                                                     -1, -1)
            # Concatenate the projected embeddings, special token embeddings, and prompt embeddings
            combined_embeds = torch.cat([projected_embeddings, special_token_embed, prompt_embeds], dim=1)

            # Transform tokenized descriptions into embeddings
            target_embeds = model.transformer.wte(target_input_ids)

            # Concatenate embeddings (combined + target)
            inputs_embeds = torch.cat([combined_embeds, target_embeds], dim=1)

            # Prepare labels for the loss function
            labels = torch.cat([
                torch.full((target_embeds.shape[0], combined_embeds.shape[1]), -100, dtype=torch.long, device=device),
                target_input_ids], dim=1)  # Ignore the loss for non-target tokens by setting them to -100

            # Forward pass through the model
            outputs = model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss  # Get the loss from the model's output
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters using the optimizer
            train_loss += loss.item()  # Add the current batch's loss to the total training loss

        avg_train_loss = train_loss / len(train_loader)  # Calculate average training loss

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        projection_layer.eval()  # Set the projection layer to evaluation mode
        val_loss = 0  # Initialize validation loss

        with torch.no_grad():  # Disable gradient computation during validation
            for batch in val_loader:  # Iterate through each batch in the validation data loader

                embeddings, target_input_ids, _, _ = zip(*batch)  # Unpack the batch

                embeddings = torch.stack(embeddings).to(device)  # Stack embeddings

                target_input_ids = torch.stack(target_input_ids).to(device)  # Stack target input IDs

                projected_embeddings = projection_layer(embeddings).unsqueeze(1)  # Project CNN embeddings

                # Tokenize the prompt text and convert to input IDs
                prompt_ids = tokenizer("Detailed description of the monster:", return_tensors="pt").input_ids.to(device)

                # Transform IDs into GPT-2 embeddings
                prompt_embeds = model.transformer.wte(prompt_ids).expand(embeddings.size(0), -1, -1)

                # Add the special token (EOS)
                special_token_embed = model.transformer.wte(torch.tensor([50256], device=device)).expand(
                    embeddings.size(0), -1, -1)

                # Concatenate the projected embeddings, special token embeddings, and prompt embeddings
                combined_embeds = torch.cat([projected_embeddings, special_token_embed, prompt_embeds], dim=1)

                # Transform tokenized descriptions into embeddings
                target_embeds = model.transformer.wte(target_input_ids)

                # Concatenate embeddings (combined + target)
                inputs_embeds = torch.cat([combined_embeds, target_embeds], dim=1)

                # Prepare labels for the loss function
                labels = torch.cat([
                    torch.full((target_embeds.shape[0], combined_embeds.shape[1]), -100, dtype=torch.long,
                               device=device), target_input_ids], dim=1)  # Ignore loss for non-target tokens

                # Forward pass through the model
                outputs = model(inputs_embeds=inputs_embeds, labels=labels)

                # Add the current batch's loss to the total validation loss
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)  # Calculate average validation loss
        print(f"Train Loss: {avg_train_loss:.6f} - Validation Loss: {avg_val_loss:.6f}")

        # Check if validation loss improved
        if avg_val_loss < best_val_loss:  # If the current validation loss is lower than the best recorded loss
            print(f'Validation loss decreased ({best_val_loss:.6f} --> {avg_val_loss:.6f}). Saving model ...')
            best_val_loss = avg_val_loss  # Update the best validation loss
            epochs_no_improve = 0  # Reset the number of epochs without improvement
            torch.save(model.state_dict(), f'gpt2_model{fold + 1}.pth')  # Save the model state
            torch.save(projection_layer.state_dict(), f'projection{fold + 1}.pth')  # Save the projection layer state
        else:
            epochs_no_improve += 1  # Increment the counter for epochs without improvement
            print(f"No improvement in validation loss for {epochs_no_improve}/{patience_es} epochs.")

        scheduler.step(avg_val_loss)  # Update the learning rate scheduler

        if epochs_no_improve >= patience_es:  # If the number of epochs without improvement exceeds the patience threshold
            print(f"Early stopping triggered after {patience_es} epochs without improvement.")
            break  # Stop training early

    if best_val_loss < best_fold_loss:  # If the current fold's validation loss is the best so far
        best_fold_loss = best_val_loss  # Update the best fold loss
        best_fold = fold + 1  # Update the best fold number


train_noise_levels = {}  # list to show monsters and their amount of noise added in the training set
val_noise_levels = {}   # list to show monsters and their amount of noise added in the validation set

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(expanded_data, labels)):
    print(f"Fold {fold + 1}/{k_folds}")

    # print("-" * 30 + " Train " + "-" * 30)
    # Salviamo i livelli di rumore per training e validation set
    for idx in train_idx:
        monster_name = expanded_data[idx]["name"]
        noise_level = expanded_data[idx]["noise_level"]
        if monster_name not in train_noise_levels:
            train_noise_levels[monster_name] = []
        train_noise_levels[monster_name].append(noise_level)
        # print(f'{monster_name}: {noise_level}')

    # print("-" * 30 + " Validation " + "-" * 30)
    for idx in val_idx:
        monster_name = expanded_data[idx]["name"]
        noise_level = expanded_data[idx]["noise_level"]
        if monster_name not in val_noise_levels:
            val_noise_levels[monster_name] = []
        val_noise_levels[monster_name].append(noise_level)
        # print(f'{monster_name}: {noise_level}')

    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)  # Initialize the GPT2 model

    # Initialize projection layer to increase the expressiveness
    projection_layer = ProjectionLayer(768, model.config.n_embd, device)

    train_subset = Subset(dataset, train_idx)  # Create a subset of the training data
    val_subset = Subset(dataset, val_idx)  # Create a subset of the validation data
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: x)  # Training data loader
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: x)  # Validation data loader

    # Initialize the AdamW optimizer with model and projection layer parameters
    optimizer = optim.AdamW(list(model.parameters()) + list(projection_layer.parameters()), lr=lr, betas=(0.9, 0.999),
                            weight_decay=0.01, fused=True)

    # Initialize the learning rate scheduler to reduce the learning rate if the validation loss does not improve
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Train the model with the specified parameters and fold
    train(model, projection_layer, train_loader, val_loader, optimizer, scheduler, device, tokenizer, epochs, fold)

print(f"\nBest fold: {best_fold} with Validation Loss: {best_fold_loss:.6f}")
