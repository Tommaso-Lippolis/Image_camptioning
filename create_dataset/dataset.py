import requests
import json

# Main URL for monsters
base_url = "https://www.dnd5eapi.co/api/monsters"

# First step: To obtain all the URL of monsters through the provided API
# request to the API
response = requests.get(base_url)
if response.status_code == 200:
    # organize the response in a json format encoded
    data = response.json()
    # Extract URLs for each monster
    monster_urls = [f"https://www.dnd5eapi.co{monster['url']}" for monster in data.get("results", [])]
else:
    print(f"Error in request: {response.status_code}")
    exit()

# Second step: To retrieve all the details about each monster
all_monsters = []
for url in monster_urls:
    monster_response = requests.get(url)
    if monster_response.status_code == 200:
        # Returns the json-encoded data for each monster and appends them to a list
        monster_data = monster_response.json()
        all_monsters.append(monster_data)
    else:
        print(f"Error in retrieving monster details from: {url}")

# Step three: Store all monsters data in a JSON file
with open("monsters.json", "w", encoding="utf-8") as file:
    json.dump(all_monsters, file, ensure_ascii=False, indent=4)

print("File 'monsters.json' successfully created!")
