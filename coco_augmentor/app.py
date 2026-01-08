import os
import json
import cv2
import numpy as np
import zipfile
import shutil
from flask import Flask, render_template, request, send_file
import albumentations as A
import sys

app = Flask(__name__)

BASE_DIR = os.getcwd() # Mengambil lokasi user saat mengetik perintah
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'augmentor_uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'augmentor_results')

# --- HELPER: Polygon to Mask ---
def polygon_to_mask(segmentation, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict): return mask 
    if isinstance(segmentation, list):
        for poly in segmentation:
            if len(poly) < 6: continue 
            poly_arr = np.array(poly).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly_arr], 1)
    return mask

# --- HELPER: Mask to Polygon ---
def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.size >= 6: 
            polygons.append(contour.flatten().tolist())
    return polygons

# --- FUNGSI UTAMA ---
def augment_coco(zip_path, config):
    print("--- MULAI AUGMENTASI CUSTOM ---")
    print(f"Config: {config}") # Debug config yang dipilih user
    
    extract_path = os.path.join(UPLOAD_FOLDER, "temp_extract")
    if os.path.exists(extract_path): shutil.rmtree(extract_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    json_file = None
    image_location_map = {} 
    
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_location_map[file] = os.path.join(root, file)
            if file.endswith('.json') and json_file is None:
                json_file = os.path.join(root, file)

    if not json_file:
        print("ERROR: JSON tidak ditemukan.")
        return None

    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # --- SETUP ALBUMENTATIONS BERDASARKAN INPUT USER ---
    transforms_list = []
    
    # 1. Flip Horizontal
    if config.get('h_flip'):
        transforms_list.append(A.HorizontalFlip(p=1.0))
        
    # 2. Flip Vertical
    if config.get('v_flip'):
        transforms_list.append(A.VerticalFlip(p=1.0))
        
    # 3. Rotate Logic
    rot_mode = config.get('rotate_mode')
    
    if rot_mode == 'fixed':
        # Kalau Fixed: Limitnya kita set (angle, angle) biar dia gak punya pilihan lain selain angle itu
        angle = int(config.get('rotate_val', 90))
        transforms_list.append(A.SafeRotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_CONSTANT))
        
    elif rot_mode == 'random':
        # Kalau Random: Limitnya int X, albumentations bakal acak dari -X sampai +X
        limit = int(config.get('rotate_val', 45))
        transforms_list.append(A.SafeRotate(limit=limit, p=1.0, border_mode=cv2.BORDER_CONSTANT))

    # Cek ada transform gak
    if not transforms_list:
        print("User tidak memilih augmentasi apapun.")
        return None

    # Compose Pipeline
    transform = A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'])
    )

    new_images = []
    new_annotations = []
    start_img_id = 900000
    start_ann_id = 900000
    processed_count = 0

    for img_info in coco_data['images']:
        clean_file_name = os.path.basename(img_info['file_name'])
        img_id = img_info['id']
        img_h = img_info['height']
        img_w = img_info['width']
        
        if clean_file_name not in image_location_map: continue
        
        img_path = image_location_map[clean_file_name]
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        current_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        masks_list = []
        bboxes_raw = []
        category_ids_list = []
        original_anns_map = [] 

        for ann in current_anns:
            if 'segmentation' in ann and ann['segmentation']:
                mask = polygon_to_mask(ann['segmentation'], img_h, img_w)
                masks_list.append(mask)
            else:
                masks_list.append(np.zeros((img_h, img_w), dtype=np.uint8))

            bboxes_raw.append(ann['bbox'])
            category_ids_list.append(ann['category_id'])
            original_anns_map.append(ann)

        try:
            transformed = transform(
                image=image, 
                masks=masks_list, 
                bboxes=bboxes_raw, 
                category_ids=category_ids_list
            )
            processed_count += 1
        except Exception as e:
            print(f"ERROR {clean_file_name}: {e}")
            continue

        trans_image = transformed['image']
        trans_masks = transformed['masks']
        trans_bboxes = transformed['bboxes']

        save_dir = os.path.dirname(img_path)
        new_filename = f"aug_{clean_file_name}"
        save_path = os.path.join(save_dir, new_filename)
        cv2.imwrite(save_path, cv2.cvtColor(trans_image, cv2.COLOR_RGB2BGR))

        new_img_entry = img_info.copy()
        new_img_entry['id'] = start_img_id
        new_img_entry['file_name'] = os.path.join(os.path.dirname(img_info['file_name']), new_filename).replace("\\", "/")
        new_img_entry['width'] = trans_image.shape[1]
        new_img_entry['height'] = trans_image.shape[0]
        new_images.append(new_img_entry)

        for i, trans_mask in enumerate(trans_masks):
            new_polygons = mask_to_polygon(trans_mask)
            if not new_polygons: continue

            new_ann = original_anns_map[i].copy()
            new_ann['id'] = start_ann_id
            new_ann['image_id'] = start_img_id
            new_ann['segmentation'] = new_polygons
            
            if i < len(trans_bboxes):
                 new_ann['bbox'] = list(trans_bboxes[i])

            new_ann['area'] = float(np.sum(trans_mask > 0))
            new_annotations.append(new_ann)
            start_ann_id += 1
        
        start_img_id += 1

    print(f"Selesai. {processed_count} gambar diproses.")

    coco_data['images'].extend(new_images)
    coco_data['annotations'].extend(new_annotations)

    output_json_path = os.path.join(extract_path, 'augmented_final.json')
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)

    result_zip = os.path.join(RESULT_FOLDER, 'augmented_dataset.zip')
    with zipfile.ZipFile(result_zip, 'w') as zipf:
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if "__MACOSX" in root: continue
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), extract_path))
    
    return result_zip

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'dataset' not in request.files: return "No file", 400
    file = request.files['dataset']
    if file.filename == '': return "No selected file", 400

    zip_path = os.path.join(UPLOAD_FOLDER, 'input.zip')
    file.save(zip_path)

    # --- AMBIL KONFIGURASI DARI FORM HTML ---
    config = {
        'h_flip': True if request.form.get('h_flip') else False,
        'v_flip': True if request.form.get('v_flip') else False,
        'rotate_mode': request.form.get('rotate_mode'), # 'none', 'fixed', 'random'
        'rotate_val': 0
    }
    
    # Ambil nilai angka rotation tergantung modenya
    if config['rotate_mode'] == 'fixed':
        config['rotate_val'] = request.form.get('fixed_angle')
    elif config['rotate_mode'] == 'random':
        config['rotate_val'] = request.form.get('random_limit')

    try:
        result_path = augment_coco(zip_path, config)
        if result_path:
            return send_file(result_path, as_attachment=True)
        else:
            return "Gagal. Cek terminal.", 500
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return f"Server Error: {e}", 500

def run_app():
    # Fungsi ini yang bakal dipanggil sama perintah CLI nanti
    print(f"🚀 COCO Augmentor jalan di http://127.0.0.1:5000")
    print(f"📂 Folder kerja: {BASE_DIR}")
    app.run(debug=False, port=5000) # Debug False biar aman buat user umum

if __name__ == '__main__':
    run_app()