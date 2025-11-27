"""
Process only NEW images that don't exist in my_data/test yet.
Keeps all existing processed data intact.
"""

import os
import sys
import shutil
from pathlib import Path
from PIL import Image

# Add external repos to path
sys.path.append(r"E:\RM Trial\preprocessing_toolkit\cloth_segmentation\U-2-Net")

def get_existing_processed_images(test_dir):
    """Get list of already processed image basenames"""
    image_dir = os.path.join(test_dir, "image")
    if not os.path.exists(image_dir):
        return set()
    
    processed = set()
    for f in os.listdir(image_dir):
        basename = os.path.splitext(f)[0]
        processed.add(basename)
    return processed

def get_existing_processed_cloths(test_dir):
    """Get list of already processed cloth basenames"""
    cloth_dir = os.path.join(test_dir, "cloth")
    if not os.path.exists(cloth_dir):
        return set()
    
    processed = set()
    for f in os.listdir(cloth_dir):
        basename = os.path.splitext(f)[0]
        processed.add(basename)
    return processed

def resize_and_copy_image(src_path, dst_path, target_size=(768, 1024)):
    """Resize image to target size and save as .jpg"""
    img = Image.open(src_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to exact dimensions
    img_resized = img.resize(target_size, Image.LANCZOS)
    
    # Save as .jpg
    dst_path_jpg = os.path.splitext(dst_path)[0] + '.jpg'
    img_resized.save(dst_path_jpg, 'JPEG', quality=95)
    print(f"[OK] Resized and saved: {os.path.basename(dst_path_jpg)}")
    return dst_path_jpg

def main():
    # Paths
    base_dir = r"E:\RM Trial\StableVITON-master\StableVITON-master"
    test_images_dir = os.path.join(base_dir, "Test-Images")
    test_clothes_dir = os.path.join(base_dir, "Test-Clothes")
    my_data_dir = os.path.join(base_dir, "my_data")
    test_dir = os.path.join(my_data_dir, "test")
    
    # Create directories
    dirs_to_create = [
        "image", "cloth", "cloth-mask", 
        "agnostic-v3.2", "agnostic-mask",
        "image-parse-v3", "image-parse-agnostic-v3.2"
    ]
    for d in dirs_to_create:
        os.makedirs(os.path.join(test_dir, d), exist_ok=True)
    
    # Get already processed images
    existing_persons = get_existing_processed_images(test_dir)
    existing_cloths = get_existing_processed_cloths(test_dir)
    
    print("\n" + "="*60)
    print("STEP 1: IDENTIFYING NEW IMAGES")
    print("="*60)
    
    # Find new person images
    new_persons = []
    for f in os.listdir(test_images_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            basename = os.path.splitext(f)[0]
            if basename not in existing_persons:
                new_persons.append(f)
                print(f"[NEW PERSON] {f}")
            else:
                print(f"[SKIP] {f} (already processed)")
    
    # Find new cloth images
    new_cloths = []
    for f in os.listdir(test_clothes_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            basename = os.path.splitext(f)[0]
            if basename not in existing_cloths:
                new_cloths.append(f)
                print(f"[NEW CLOTH] {f}")
            else:
                print(f"[SKIP] {f} (already processed)")
    
    if not new_persons and not new_cloths:
        print("\n[INFO] No new images to process!")
        return
    
    print("\n" + "="*60)
    print("STEP 2: COPYING AND RESIZING NEW IMAGES")
    print("="*60)
    
    # Process new person images
    new_person_basenames = []
    for filename in new_persons:
        src_path = os.path.join(test_images_dir, filename)
        basename = os.path.splitext(filename)[0]
        dst_path = os.path.join(test_dir, "image", f"{basename}.jpg")
        
        resize_and_copy_image(src_path, dst_path)
        new_person_basenames.append(basename)
    
    # Process new cloth images
    new_cloth_basenames = []
    for filename in new_cloths:
        src_path = os.path.join(test_clothes_dir, filename)
        basename = os.path.splitext(filename)[0]
        dst_path = os.path.join(test_dir, "cloth", f"{basename}.jpg")
        
        resize_and_copy_image(src_path, dst_path)
        new_cloth_basenames.append(basename)
    
    print("\n" + "="*60)
    print("STEP 3: GENERATING CLOTH MASKS (U-2-Net)")
    print("="*60)
    
    if new_cloth_basenames:
        import torch
        from model import U2NET
        import numpy as np
        from PIL import Image
        import torch.nn.functional as F
        
        # Load U-2-Net model
        model_path = r"E:\RM Trial\preprocessing_toolkit\cloth_segmentation\saved_models\u2net.pth"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = U2NET(3, 1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"[OK] Loaded U-2-Net model on {device}")
        
        for basename in new_cloth_basenames:
            cloth_path = os.path.join(test_dir, "cloth", f"{basename}.jpg")
            mask_path = os.path.join(test_dir, "cloth-mask", f"{basename}.jpg")
            
            # Process cloth image
            img = Image.open(cloth_path).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                d1, d2, d3, d4, d5, d6, d7 = model(img_tensor)
            
            pred = d1[:, 0, :, :]
            pred = (pred - pred.min()) / (pred.max() - pred.min())
            
            # Resize to match cloth size
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), 
                               size=(1024, 768), 
                               mode='bilinear', 
                               align_corners=False)
            
            mask_array = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_array, mode='L').convert('RGB')
            mask_img.save(mask_path, 'JPEG', quality=95)
            print(f"[OK] Generated cloth mask: {basename}.jpg")
    
    print("\n" + "="*60)
    print("STEP 4: GENERATING HUMAN PARSING (Ailia)")
    print("="*60)
    
    if new_person_basenames:
        ailia_script = r"E:\RM Trial\dummy\ailia-models\image_segmentation\human_part_segmentation\human_part_segmentation.py"
        parsing_temp_dir = os.path.join(my_data_dir, "_temp", "parsing_raw")
        os.makedirs(parsing_temp_dir, exist_ok=True)
        
        for basename in new_person_basenames:
            person_path = os.path.join(test_dir, "image", f"{basename}.jpg")
            parsing_output = os.path.join(parsing_temp_dir, f"{basename}_parsing.png")
            
            cmd = f'conda activate viton-pip && python "{ailia_script}" -i "{person_path}" -s "{parsing_output}"'
            result = os.system(cmd)
            
            if result == 0 and os.path.exists(parsing_output):
                print(f"[OK] Generated parsing: {basename}_parsing.png")
            else:
                print(f"[FAILED] Could not generate parsing for {basename}")
    
    print("\n" + "="*60)
    print("STEP 5: CONVERTING PARSING TO VITON FORMATS")
    print("="*60)
    
    if new_person_basenames:
        convert_script = r"E:\RM Trial\preprocessing_toolkit\scripts\convert_parsing_to_viton.py"
        
        cmd = f'conda activate viton-pip && python "{convert_script}" ' \
              f'--person_dir "{os.path.join(test_dir, "image")}" ' \
              f'--parsing_dir "{parsing_temp_dir}" ' \
              f'--output_dir "{test_dir}"'
        
        os.system(cmd)
    
    print("\n" + "="*60)
    print("[SUCCESS] PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nProcessed:")
    print(f"  - {len(new_person_basenames)} new person images")
    print(f"  - {len(new_cloth_basenames)} new cloth images")
    print(f"\nAll data saved to: {test_dir}")

if __name__ == "__main__":
    main()
