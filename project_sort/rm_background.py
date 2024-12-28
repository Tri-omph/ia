import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

#== Parameters =======================================================================
BLUR = 33
CANNY_THRESH_1 = 50
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (1.0,1.0,1.0) # In BGR format


def remove_background(img: np.ndarray, output_path: str, showPlot: bool = False) -> None:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None) #increase thickness of the edges (so combine them better)
    edges = cv2.erode(edges, None) #shrink the edges (so remove noise)

    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if contours:  # Check if any contours are found

        # Get the max area determined by contours
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]

        #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
        # Mask is black, polygon is white
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))

        #-- Smooth mask, then blur it --------------------------------------------------------
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

        #-- Blend masked img into MASK_COLOR background --------------------------------------
        mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
        img         = img.astype('float32') / 255.0                 #  for easy blending

        masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
        masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

        if showPlot:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
            plt.title("Foreground Segmentation")
            plt.show()

        cv2.imwrite(output_path, masked)           # Save
    else:
        #print(f"No contours found in {output_path}. Skipping background removal.")
        cv2.imwrite(output_path, img)

from PIL import Image
if __name__ == "__main__":
    val_dir = '../../images/final_val/Trash'
    val_dir_no_bg = '../../images/final_val_no_bg/Trash'
    
    for subfolder, _, files in os.walk(val_dir):
        for image in tqdm(files, desc=subfolder):
            # Construct full paths for the original and destination files
            input_image_path = os.path.join(subfolder, image)
            output_image_path = os.path.join(val_dir_no_bg, os.path.relpath(input_image_path, val_dir))
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            
            if ".gif" in input_image_path:
                gif_image = Image.open(input_image_path)
                first_frame = gif_image.convert('RGB')
                first_frame_np = np.array(first_frame)
                imageArray = cv2.cvtColor(first_frame_np, cv2.COLOR_RGB2BGR)
                output_image_path = output_image_path[:-3] + "jpg"
            else:
                imageArray = cv2.imread(input_image_path)
            
            # Call the function to remove the background
            remove_background(imageArray, output_image_path)