import torch
import json
import os
from torchvision import transforms
from PIL import Image
from CNN.CNN import CNN

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load pre-trained weights if available
model = CNN().to(device)
model.load_state_dict(torch.load("../CNN/cnn_model_fold_3.pth", map_location=device))
model.eval()  # Set the model to evaluation mode

# Resize and normalize input images
transform = transforms.Compose([
    transforms.Resize((236, 236)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the JSON file containing monster data
with open("../dataset/monsters.json", "r") as file:
    monsters = json.load(file)


# Now we define a function which retrieves from the
# previous file all the relevant features and generates
# a compact inline description

# Function to generate a monster description
def generate_description(monster):
    name = monster.get('name', 'Unknown Monster')
    size = monster.get('size', 'Unknown Size')
    type_ = monster.get('type', 'Unknown Type')
    alignment = monster.get('alignment', 'Unknown Alignment')
    # Armor class: include both value and type if available
    armor_class_list = monster.get('armor_class', [])
    if armor_class_list and isinstance(armor_class_list, list):
        armor_class = ", ".join(
            [f"{ac.get('value', 'Unknown')} ({ac.get('type', 'Unknown type')})" for ac in armor_class_list])
    else:
        armor_class = "Unknown"

    # Collect all available speeds
    speeds = monster.get('speed', {})
    speed = ", ".join([f"{k}: {v}" for k, v in speeds.items()]) if speeds else "Unknown speed"

    hit_points = monster.get('hit_points', 'Unknown')
    strength = monster.get('strength', 'Unknown')
    dexterity = monster.get('dexterity', 'Unknown')
    constitution = monster.get('constitution', 'Unknown')
    intelligence = monster.get('intelligence', 'Unknown')
    wisdom = monster.get('wisdom', 'Unknown')
    charisma = monster.get('charisma', 'Unknown')

    # Damage resistances
    damage_resistances = ", ".join(monster.get('damage_resistances', [])) if monster.get(
        'damage_resistances') else "None"

    # Condition immunities
    condition_immunities = ", ".join(
        [immunity['name'] for immunity in monster.get('condition_immunities', [])]) if monster.get(
        'condition_immunities') else "None"

    # Senses
    senses = monster.get('senses', {})
    senses_info = ", ".join([f"{k}: {v}" for k, v in senses.items()]) if senses else "None"

    # Languages
    languages = monster.get('languages', "Unknown")

    # Challenge rating
    challenge_rating = monster.get('challenge_rating', "Unknown")

    # Proficiencies
    proficiencies = ", ".join([p['proficiency']['name'] for p in monster.get('proficiencies', [])])

    # Special Abilities (max 400 characters)
    special_abilities = []
    for ability in monster.get('special_abilities', []):
        special_abilities.append(f"{ability['name']}: {ability['desc']}")

    # Actions (max 500 characters)
    actions = []
    for action in monster.get('actions', []):
        actions.append(f"{action['name']}: {action['desc']}")

    # Join all the features
    description = (f"{name}. "
                   f"Size: {size}. "
                   f"Type: {type_}. "
                   f"Alignment: {alignment}. "
                   f"Armor Class: {armor_class}. "
                   f"Hit points: {hit_points}. "
                   f"Speed: {speed}. "
                   f"Strength: {strength}. "
                   f"Dexterity: {dexterity}. "
                   f"Constitution: {constitution}. "
                   f"Intelligence: {intelligence}. "
                   f"Wisdom: {wisdom}. "
                   f"Charisma: {charisma}. "
                   f"Damage Resistances: {damage_resistances}. "
                   f"Condition Immunities: {condition_immunities}. "
                   f"Senses: {senses_info}. "
                   f"Languages: {languages}. "
                   f"Challenge Rating: {challenge_rating}. "
                   f"Proficiencies: {proficiencies}. "
                   f"Special Abilities: {', '.join(special_abilities)[:400]}. "
                   f"Actions: {', '.join(actions)[:500]}.")

    return description


# List to store monsters with their generated descriptions
monsters_with_descriptions = []

# Iterate through the list of monsters and generate
# descriptions to store with corresponding names
for monster in monsters:
    description = generate_description(monster)
    monsters_with_descriptions.append({
        'name': monster['name'],
        'description': description.replace("\n", " ").replace("Ã—", "")
    })

# Save the generated descriptions into a new JSON file
with open('monsters_with_descriptions.json', 'w') as f:
    json.dump(monsters_with_descriptions, f, indent=4)

print("Descrizioni generate e salvate nel file 'monsters_with_descriptions.json'.")

# Now we aim to create a single json File formatted as below:
# "Monster name": {
#                  "name": "",
#                  "description": "",
#                  "embedding": []
#                  }
#
# To do so, first we need to pass images through the CNN model
# to generate the corresponding embedding


# Load the descriptions from the saved JSON file
with open("./monsters_with_descriptions.json", "r") as f:
    descriptions = json.load(f)

# Path to the folder containing monster images
data_dir = "../dataset/monster_image/"

# Dictionary to store embeddings and descriptions
monster_data = {}

# Iterate through the directories and files in the dataset
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg")):  # Process only specific image files

                # Remove the file extension and replace hyphens with spaces
                monster_name = os.path.splitext(filename)[0].replace("-", " ").lower()

                # Full path of the image
                img_path = os.path.join(folder_path, filename)
                image = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
                image = transform(image).unsqueeze(0).to(device)  # Apply transformations and move to GPU/CPU

                # Extract embeddings using the CNN model
                with torch.no_grad():
                    output, embedding = model(image)
                    embedding = embedding.squeeze().tolist()

                # Find the corresponding description in the JSON file
                description = "N/A"  # Default value if not found
                found_description = False  # Flag to check if the description is found
                for monster in descriptions:  # Iterate through the list of monster descriptions

                    # Replace hyphens with spaces in the JSON index to match image names
                    monster_index = monster["name"].replace("-", " ").lower()
                    if monster_index == monster_name:  # Compare description names with images names

                        # Take the corresponding description
                        description = monster["description"]
                        found_description = True
                        break

                # Debug message if no description was found
                if not found_description:
                    print(f"Descrizione non trovata per: {monster_name}")

                # Add the monster data to the dictionary
                monster_data[monster_name] = {
                    "name": monster_name,
                    "description": description,
                    "embedding": embedding
                }

# Save the final dictionary with embeddings and descriptions into a JSON file
with open("monsters_embeddings.json", "w") as f:
    json.dump(monster_data, f, indent=4)

print("Embeddings e descrizioni salvati con successo!")
