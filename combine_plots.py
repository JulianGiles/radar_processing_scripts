#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 11:17:30 2026

@author: jgiles
"""

from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import string

#%% Combine vertically 2 plots
# 1. Load your saved plot images
# Replace these with your actual file names
img_a_path = "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/alpha_zdr_lines_sbmdsd.png"
img_b_path = "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/beta_zdr_lines_sbmdsd.png"
save_path = "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/Figure15.png"

img_a = Image.open(img_a_path)
img_b = Image.open(img_b_path)

# 2. Get the dimensions of both images
width_a, height_a = img_a.size
width_b, height_b = img_b.size

# 3. Create a new blank white canvas large enough to hold both
max_width = max(width_a, width_b)
total_height = height_a + height_b
combined_img = Image.new('RGB', (max_width, total_height), color='white')

# 4. Paste the images onto the canvas
# Image A goes at the top (x=0, y=0)
combined_img.paste(img_a, (0, 0))
# Image B goes directly below Image A (x=0, y=height_a)
combined_img.paste(img_b, (0, height_a))

# 5. Initialize ImageDraw to add the (a) and (b) text
draw = ImageDraw.Draw(combined_img)

# Try to load a standard sans-serif font to match your plots
# The size (e.g., 40) depends on your PNG's DPI. If the letters look too small/big, adjust this number.
try:
    font = ImageFont.truetype("arial.ttf", size=40)
except IOError:
    # Fallback if Arial isn't found on your system's default font path
    font_prop = fm.FontProperties(family='sans serif')
    matplotlib_font_path = fm.findfont(font_prop)
    font = ImageFont.truetype(matplotlib_font_path, size=35)

# 6. Draw the text labels
# Adjust the (15, 15) coordinates to nudge the text exactly where you want it
text_color = (0, 0, 0) # Black
draw.text((15, 15), "(a)", fill=text_color, font=font)
draw.text((15, height_a + 15), "(b)", fill=text_color, font=font)

# 7. Save the final combined figure
combined_img.save(save_path)
print("Images successfully stitched and saved!")

#%% Combine vertically 3 plots
# 1. Load your saved plot images
# Replace these with your actual file names
img_a_path = "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/example_HTY_2020-03-13_DBZH.png"
img_b_path = "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/example_HTY_2020-03-13_DBZH_AC.png"
img_c_path = "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/example_HTY_2020-03-13_DBZH_AC-DBZH.png"
save_path = "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/Figure10.png"

img_a = Image.open(img_a_path)
img_b = Image.open(img_b_path)
img_c = Image.open(img_c_path)

# 2. Get the dimensions of both images
width_a, height_a = img_a.size
width_b, height_b = img_b.size
width_c, height_c = img_c.size

# 3. Create a new blank white canvas large enough to hold both
max_width = max(width_a, width_b, width_c)
total_height = height_a + height_b + height_c
combined_img = Image.new('RGB', (max_width, total_height), color='white')

# 4. Paste the images onto the canvas
# Image A goes at the top (x=0, y=0)
combined_img.paste(img_a, (0, 0))
# Image B goes directly below Image A (x=0, y=height_a)
combined_img.paste(img_b, (0, height_a))
# Image C goes directly below Image B (x=0, y=height_a+height_b)
combined_img.paste(img_c, (0, height_a+height_b))

# 5. Initialize ImageDraw to add the (a) and (b) text
draw = ImageDraw.Draw(combined_img)

# Try to load a standard sans-serif font to match your plots
# The size (e.g., 40) depends on your PNG's DPI. If the letters look too small/big, adjust this number.
try:
    font = ImageFont.truetype("arial.ttf", size=40)
except IOError:
    # Fallback if Arial isn't found on your system's default font path
    font_prop = fm.FontProperties(family='sans serif')
    matplotlib_font_path = fm.findfont(font_prop)
    font = ImageFont.truetype(matplotlib_font_path, size=35)

# 6. Draw the text labels
# Adjust the (15, 15) coordinates to nudge the text exactly where you want it
text_color = (0, 0, 0) # Black
draw.text((15, 15), "(a)", fill=text_color, font=font)
draw.text((15, height_a + 15), "(b)", fill=text_color, font=font)
draw.text((15, height_a+height_b + 15), "(c)", fill=text_color, font=font)

# 7. Save the final combined figure
combined_img.save(save_path)
print("Images successfully stitched and saved!")

#%% Combine an arbitrary number of images vertically
def stitch_images_vertically(image_paths, save_path, max_images=10):
    """
    Stitches a list of images vertically and adds (a), (b), (c)... labels.
    """
    # 1. Enforce limits
    if len(image_paths) > max_images:
        print(f"Warning: You provided {len(image_paths)} images. Limiting to {max_images}.")
        image_paths = image_paths[:max_images]

    if not image_paths:
        print("Error: No images provided in the list.")
        return

    # 2. Load all images into a list
    images = [Image.open(path) for path in image_paths]

    # 3. Calculate canvas dimensions dynamically
    # The max width among all images, and the sum of all their heights
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)

    # 4. Create a blank white canvas
    combined_img = Image.new('RGB', (max_width, total_height), color='white')
    draw = ImageDraw.Draw(combined_img)

    # 5. Initialize the font (with fallback)
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except IOError:
        font_prop = fm.FontProperties(family='sans serif')
        matplotlib_font_path = fm.findfont(font_prop)
        font = ImageFont.truetype(matplotlib_font_path, size=35)

    # 6. Loop through images to paste them and add text
    current_y = 0
    letters = string.ascii_lowercase  # Provides 'a', 'b', 'c', ..., 'z'
    text_color = (0, 0, 0)

    for i, img in enumerate(images):
        # Paste the current image at the current vertical offset
        combined_img.paste(img, (0, current_y))

        # Draw the letter label (e.g., "(a)", "(b)")
        label = f"({letters[i]})"
        draw.text((15, current_y + 15), label, fill=text_color, font=font)

        # Update the vertical offset for the next image in the loop
        current_y += img.height

    # 7. Save the final combined figure
    combined_img.save(save_path)
    print(f"Successfully stitched {len(images)} images and saved to: {save_path}")

# ==========================================
# Execution Block
# ==========================================

# Just add or remove paths from this list!
my_images = [
    "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/example_HTY_2020-03-13_ZDR_EC_OC_ZM.png",
    "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/example_HTY_2020-03-13_ZDR_EC_OC_WRC.png",
    "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/example_HTY_2020-03-13_ZDR_EC_OC_WRC_AC.png",
    "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/example_HTY_2020-03-13_ZDR_EC_OC_WRC_AC-ZDR_EC_OC_WRC.png"
]

out_path = "/home/jgiles/sciebo/documents-papers/A04 papers/1_paper_rain_ml_wet_radome_atten/final_figures/Figure11.png"

# Call the function
stitch_images_vertically(my_images, out_path)