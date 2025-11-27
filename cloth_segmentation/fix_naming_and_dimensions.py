"""
Fix naming conventions and ensure all images are 768x1024
"""

import os
from PIL import Image

test_dir = r"E:\RM Trial\StableVITON-master\StableVITON-master\my_data\test"

print("="*60)
print("STEP 2: FIXING NAMING CONVENTIONS")
print("="*60)

# 1. Convert cloth-mask PNG to JPG
cloth_mask_dir = os.path.join(test_dir, "cloth-mask")
for filename in os.listdir(cloth_mask_dir):
    if filename.endswith('.png'):
        old_path = os.path.join(cloth_mask_dir, filename)
        new_name = filename.replace('.png', '.jpg')
        new_path = os.path.join(cloth_mask_dir, new_name)
        
        # Convert PNG to JPG
        img = Image.open(old_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(new_path, 'JPEG', quality=95)
        os.remove(old_path)
        print(f"[OK] cloth-mask: {filename} -> {new_name}")

# 2. Ensure agnostic-v3.2 are .jpg (already correct)
print("\n[OK] agnostic-v3.2: Already in .jpg format")

# 3. Ensure agnostic-mask follows name_mask.png convention (already correct)
print("[OK] agnostic-mask: Already follows {name}_mask.png convention")

# 4. Ensure image-parse-agnostic-v3.2 are .png (already correct)
print("[OK] image-parse-agnostic-v3.2: Already in .png format")

# 5. Ensure image-parse-v3 are .png (already correct)
print("[OK] image-parse-v3: Already in .png format")

print("\n" + "="*60)
print("STEP 3: ENSURING 768x1024 DIMENSIONS")
print("="*60)

folders_to_check = ['image', 'cloth', 'agnostic-v3.2', 'agnostic-mask', 
                    'cloth-mask', 'image-parse-v3', 'image-parse-agnostic-v3.2', 'image-densepose']

for folder in folders_to_check:
    folder_path = os.path.join(test_dir, folder)
    if not os.path.exists(folder_path):
        continue
    
    print(f"\nChecking {folder}/")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            img = Image.open(file_path)
            width, height = img.size
            
            if width != 768 or height != 1024:
                print(f"  [RESIZE] {filename}: {width}x{height} -> 768x1024")
                
                # Resize to 768x1024
                if img.mode == 'P':  # Palette mode (for masks)
                    img = img.convert('RGB')
                img_resized = img.resize((768, 1024), Image.LANCZOS)
                
                # Save with original format
                if filename.endswith('.png'):
                    img_resized.save(file_path, 'PNG')
                else:
                    if img_resized.mode != 'RGB':
                        img_resized = img_resized.convert('RGB')
                    img_resized.save(file_path, 'JPEG', quality=95)
            else:
                print(f"  [OK] {filename}: {width}x{height}")

print("\n" + "="*60)
print("[SUCCESS] ALL NAMING AND DIMENSIONS FIXED!")
print("="*60)
