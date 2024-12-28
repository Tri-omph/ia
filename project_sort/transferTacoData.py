import os
import shutil

def copy_files(src_folder, dest_folder):
    # Ensure the destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Iterate through all the files in the source folder
    for file_name in os.listdir(src_folder):
        # Create full file path
        src_file = os.path.join(src_folder, file_name)
        
        # Check if it's a file
        if os.path.isfile(src_file):
            dest_file = os.path.join(dest_folder, file_name)
            # Copy file to destination folder
            shutil.copy(src_file, dest_file)
            print(f"Copied: {file_name}")

# Example usage:
location = "../../images/"
taco = location + "taco-data/name/"
final = location + "Final/"

move = {
    'Battery': ['Battery'],
    'Cardboard': ['Corrugated carton', 'Egg carton', 'Meal carton', 'Other carton'],
    'Glass': ['Broken glass', 'Glass bottle', 'Glass cup', 'Glass jar'],
    'Metal': ['Aluminium blister pack', 'Aluminium foil', 'Metal bottle cap', 'Metal lid', 'Scrap metal', 'Food Can', 'Drink can', 'Pop tab'],
    'Paper': ['Normal paper', 'Paper bag', 'Paper cup', 'Paper straw', 'Wrapping paper', 'Toilet tube', 'Cigarette'],
    'Plastic': ['Clear plastic bottle', 'Crisp packet', 'Disposable food container', 'Disposable plastic cup', 'Drink carton', 'Foam cup', 'Foam food container', 'Other plastic container', 'Plastic bottle cap', 'Plastic film', 'Plastic lid', 'Plastic straw', 'Plastic utensils', 'Other plastic', 'Other plastic bottle', 'Other plastic wrapper', 'Tupperware', 'Squeezable tube'],
    'Textile': ['Shoe', 'Rope & strings', 'Single-use carrier bag'],
}

for our_folder in move:
    taco_folders = move[our_folder]
    for taco_folder in taco_folders:
        taco_location = taco + taco_folder
        final_taco = final + our_folder
        
        copy_files(taco_location, final_taco)