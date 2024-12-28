import json

## TODO: Cut out specific spot in image
bg_tag_name = "bg_ids"
bgs_to_remove = [3, 5, 6]
save_folder = ""

image_to_annotation = {}
with open('../TACO-master/data/annotations.json') as file:

    data = json.load(file)
    for i in range(len(data.get('annotations'))):
        asso_image_id = data.get('annotations')[i].get('image_id')

        image_to_annotation[data['annotations'][i]['id']] = {
            "image_id" : asso_image_id,
            "image_link" : data['images'][asso_image_id]["flickr_url"],
            "category_id" : data['annotations'][i]['category_id'],
            "category_name": data["categories"][data['annotations'][i]['category_id']]["name"],
            bg_tag_name : data['scene_annotations'][asso_image_id]['background_ids'],
        }
"""
value_counts = {}

for largeV in image_to_annotation.values():
    value = largeV["image_id"]
    if value in value_counts:
        value_counts[value] += 1
    else:
        value_counts[value] = 1

valid_ids = {key: value for key, value in value_counts.items() if value == 1}.keys()
print(valid_ids)"""

image_to_annotation = {
    key: value for key, value in image_to_annotation.items() if (not any(item in value[bg_tag_name] for item in bgs_to_remove))
}

value_counts = []

for largeV in image_to_annotation.values():
    value = largeV["image_id"]
    if value not in value_counts:
        value_counts.append(value)

## UNTESTED
import os
import requests

# Path where the image will be saved
save_folder = '../../images/taco-data/number'
os.makedirs(save_folder, exist_ok=True)  # Ensure the folder exists

leng = len(image_to_annotation)
index=0
for image_info in image_to_annotation:

    # URL of the image to download
    image_url = image_to_annotation[image_info]["image_link"]

    category_folder = (str)(image_to_annotation[image_info]["category_id"])
    save_folder_category = os.path.join(save_folder, category_folder)
    os.makedirs(save_folder_category, exist_ok=True)

    # Define the full path for saving the image
    image_name = image_url.split('/')[-1]  # Extract image name from URL
    save_path = os.path.join(save_folder_category, image_name)

    # Check if the request was successful
    if os.path.exists(save_path):
        print(f"Image {image_name} already exists, skipping download.")
    else:
        # Fetch and save the image
        response = requests.get(image_url)

        # Check if the request was successful
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"Image {index}/{leng} saved as {save_path}")
        else:
            print(f"Failed to retrieve the image {image_name}.")
    index += 1