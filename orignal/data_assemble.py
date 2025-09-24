import os
import shutil
import argparse
import pandas as pd
from tqdm.auto import tqdm

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def list_images(folder):
    try:
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(folder, f))
        ]
    except FileNotFoundError:
        return []

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def assemble_dataset(excel_path, src_root, dst_root, split=0.8, seed=42, limit_per_class=None):
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Source root not found: {src_root}")

    df = pd.read_excel(excel_path)
    # Check for expected columns
    if 'Video_Num' not in df.columns or 'Infection_type' not in df.columns:
        raise ValueError("Excel must contain columns: 'Video_Num' and 'Infection_type'")

    # Use Video_Num as folder name and Infection_type as class label
    df = df[['Video_Num', 'Infection_type']].dropna()
    df['Video_Num'] = df['Video_Num'].astype(str)
    df['Infection_type'] = df['Infection_type'].astype(str)

    # Collect images per class
    images_by_class = {}
    total_missing_folders = 0
    skipped_folders = []
    
    for _, row in df.iterrows():
        cls = row['Infection_type']
        video_num = row['Video_Num']
        folder_path = os.path.join(src_root, video_num)
        
        # Check if folder exists before trying to list images
        if not os.path.exists(folder_path):
            total_missing_folders += 1
            skipped_folders.append(video_num)
            print(f"Warning: Video folder {video_num} not found, skipping...")
            continue
            
        imgs = list_images(folder_path)
        if not imgs:
            total_missing_folders += 1
            skipped_folders.append(video_num)
            print(f"Warning: No images found in folder {video_num}, skipping...")
            continue
            
        images_by_class.setdefault(cls, []).extend(imgs)
        print(f"Processed folder {video_num}: found {len(imgs)} images for class '{cls}'")

    # Optional limit per class
    if limit_per_class is not None:
        import random
        rng = random.Random(seed)
        for cls, lst in images_by_class.items():
            if len(lst) > limit_per_class:
                rng.shuffle(lst)
                images_by_class[cls] = lst[:limit_per_class]

    # Split per class
    import random
    rng = random.Random(seed)
    train_manifest, val_manifest = [], []
    for cls, lst in images_by_class.items():
        if not lst:
            continue
        rng.shuffle(lst)
        n_train = int(split * len(lst))
        train_manifest.extend([(p, cls, 'train') for p in lst[:n_train]])
        val_manifest.extend([(p, cls, 'val') for p in lst[n_train:]])

    # Create destination dirs
    for split_name in ('train', 'val'):
        for cls in images_by_class.keys():
            ensure_dir(os.path.join(dst_root, split_name, cls))

    # Copy files
    def copy_files(manifest, split_type):
        for src_path, cls, split_name in tqdm(manifest, desc=f"Copying {split_type}", leave=False):
            fname = os.path.basename(src_path)
            dst_path = os.path.join(dst_root, split_name, cls, fname)
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(fname)
                i = 1
                while True:
                    alt = f"{base}_{i}{ext}"
                    alt_path = os.path.join(dst_root, split_name, cls, alt)
                    if not os.path.exists(alt_path):
                        dst_path = alt_path
                        break
                    i += 1
            try:
                shutil.copy2(src_path, dst_path)
            except FileNotFoundError:
                continue

    copy_files(train_manifest, "train")
    copy_files(val_manifest, "val")

    # Summary
    train_counts = {c: 0 for c in images_by_class.keys()}
    val_counts = {c: 0 for c in images_by_class.keys()}
    for _, cls, _ in train_manifest:
        train_counts[cls] += 1
    for _, cls, _ in val_manifest:
        val_counts[cls] += 1

    print("\n" + "="*50)
    print("DATASET ASSEMBLY SUMMARY")
    print("="*50)
    print(f"- Destination: {dst_root}")
    print(f"- Missing/empty source folders skipped: {total_missing_folders}")
    if skipped_folders:
        print(f"- Skipped video folders: {', '.join(sorted(skipped_folders))}")
    print(f"- Successfully processed classes: {len(images_by_class)}")
    print("\nClass distribution:")
    for cls in sorted(images_by_class.keys()):
        total_class = train_counts[cls] + val_counts[cls]
        print(f"  {cls}: {total_class} total (train={train_counts[cls]}, val={val_counts[cls]})")
    print(f"\nTotal images: {sum(train_counts.values()) + sum(val_counts.values())}")
    print("Dataset ready for CNN training!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble CNN-ready dataset from Candida infection data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python data_assemble.py --excel Infection_data.xlsx --src-root dataset --dst-root processed_dataset

This script will:
1. Read Video_Num and Infection_type from the Excel file
2. Skip missing video folders automatically
3. Copy only existing .tif frames
4. Split data into train/val sets (80/20 by default)
5. Create CNN-ready folder structure: processed_dataset/train/class_name/ and processed_dataset/val/class_name/
        """
    )
    parser.add_argument("--excel", default="Infection_data.xlsx", 
                        help="Path to Excel with Video_Num and Infection_type columns (default: Infection_data.xlsx)")
    parser.add_argument("--src-root", default="dataset", 
                        help="Root directory containing video folders (default: dataset)")
    parser.add_argument("--dst-root", default="processed_dataset", 
                        help="Destination root for train/val structure (default: processed_dataset)")
    parser.add_argument("--split", type=float, default=0.8, 
                        help="Train split ratio 0-1 (default: 0.8 for 80%% train, 20%% validation)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--limit-per-class", type=int, default=None, 
                        help="Optional: limit number of images per class for testing")
    args = parser.parse_args()

    print("Starting Candida infection dataset assembly...")
    print(f"Excel file: {args.excel}")
    print(f"Source: {args.src_root}")
    print(f"Destination: {args.dst_root}")
    print(f"Train/Val split: {args.split:.1%}/{1-args.split:.1%}")
    print("-" * 50)

    assemble_dataset(
        excel_path=args.excel,
        src_root=args.src_root,
        dst_root=args.dst_root,
        split=args.split,
        seed=args.seed,
        limit_per_class=args.limit_per_class
    )