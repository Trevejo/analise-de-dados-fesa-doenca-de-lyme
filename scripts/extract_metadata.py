import os
import re
import numpy as np
import pandas as pd
from PIL import Image

def extract_condition(filename):
    clean = re.sub(r'\d+\.jpg$', '', filename, flags=re.IGNORECASE).strip()
    return clean

def analyze_dataset(base_dir):
    records = []
    
    # We analyze DataSplit/image_folder 2 (full dataset) and mark splits
    datasplit_dir = os.path.join(base_dir, 'DataSplit', 'image_folder 2')
    
    # Mapping each file to its split if possible
    split_map = {}
    for split in ['train', 'val', 'test']:
        split_p = os.path.join(datasplit_dir, split)
        if os.path.isdir(split_p):
            for label in ['Positive', 'Negative']:
                lp = os.path.join(split_p, label)
                if os.path.isdir(lp):
                    for f in os.listdir(lp):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            split_map[f] = split

    # Now read the full set from Positive and Negative folders
    for label in ['Positive', 'Negative']:
        label_p = os.path.join(datasplit_dir, label)
        if not os.path.isdir(label_p):
            continue
        
        for fname in os.listdir(label_p):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            fpath = os.path.join(label_p, fname)
            file_size = os.path.getsize(fpath)
            condition = extract_condition(fname)
            split = split_map.get(fname, 'unassigned')
            
            try:
                with Image.open(fpath) as img:
                    img_rgb = img.convert('RGB')
                    w, h = img_rgb.size
                    arr = np.array(img_rgb, dtype=np.float32)
                    
                    r = arr[:, :, 0]
                    g = arr[:, :, 1]
                    b = arr[:, :, 2]
                    
                    mean_r = float(np.mean(r))
                    mean_g = float(np.mean(g))
                    mean_b = float(np.mean(b))
                    
                    std_r = float(np.std(r))
                    std_g = float(np.std(g))
                    std_b = float(np.std(b))
                    
                    # Luminance
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    brightness = float(np.mean(lum))
                    contrast = float(np.std(lum))
                    
                    # Dermatological Erythema Proxy: Mean Red - Mean Green
                    erythema_idx = float(mean_r - mean_g)
                    
                    records.append({
                        'filename': fname,
                        'file_size_bytes': file_size,
                        'binary_class': label,
                        'condition': condition,
                        'split': split,
                        'width': w,
                        'height': h,
                        'aspect_ratio': round(w / h, 4) if h > 0 else 0,
                        'color_mode': img.mode,
                        'channels': len(img.getbands()),
                        'mean_r': round(mean_r, 2),
                        'mean_g': round(mean_g, 2),
                        'mean_b': round(mean_b, 2),
                        'std_r': round(std_r, 2),
                        'std_g': round(std_g, 2),
                        'std_b': round(std_b, 2),
                        'brightness': round(brightness, 2),
                        'contrast': round(contrast, 2),
                        'erythema_index': round(erythema_idx, 2),
                        'corrupted': False
                    })
            except Exception as e:
                records.append({
                    'filename': fname,
                    'file_size_bytes': file_size,
                    'binary_class': label,
                    'condition': condition,
                    'split': split,
                    'width': 0,
                    'height': 0,
                    'aspect_ratio': 0,
                    'color_mode': 'CORRUPT',
                    'channels': 0,
                    'mean_r': 0,
                    'mean_g': 0,
                    'mean_b': 0,
                    'std_r': 0,
                    'std_g': 0,
                    'std_b': 0,
                    'brightness': 0,
                    'contrast': 0,
                    'erythema_index': 0,
                    'corrupted': True
                })

    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    data_path = '/home/usuario/side-projects/analise-de-dados-fesa-doenca-de-lyme/data'
    print(f'Starting extraction from {data_path}...')
    df = analyze_dataset(data_path)
    out_csv = os.path.join(data_path, 'metadata_lyme_rashes.csv')
    df.to_csv(out_csv, index=False)
    print(f'Extracted {len(df)} records. Saved to {out_csv}.')
    print('Summary of classes:')
    print(df['binary_class'].value_counts())
    print('Corrupted files:', df['corrupted'].sum())
