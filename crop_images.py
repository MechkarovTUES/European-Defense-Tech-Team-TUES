import os
from PIL import Image

def crop_images_in_folder(folder_path, crop_area):
    cropped_folder_path = os.path.join(os.path.dirname(folder_path), f"cropped_{os.path.basename(folder_path)}")
    os.makedirs(cropped_folder_path, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            
            cropped_image = image.crop(crop_area)
            
            cropped_image.save(os.path.join(cropped_folder_path, filename))
    
    print(f"All images have been cropped and saved to {cropped_folder_path}")

folder_path = "images"
crop_area = (333, 160, 973, 800)

if __name__ == "__main__":
    crop_images_in_folder(folder_path, crop_area)
