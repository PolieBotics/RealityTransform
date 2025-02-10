#!/usr/bin/env python3
"""
create_paired_folders.py

Iterate all subdirs in --input_dir (defaults to "saved_datasets"),
looking for:
   subdir/Emissions/*.jpg
   subdir/Recordings/*.jpg
and create a Pix2Pix-style dataset:
   datasets/subdir/train_A   (Recordings)
   datasets/subdir/train_B   (Emissions)

Defaults:
  - center crop the Recordings to 75% vertical height (full width),
  - then resize both images to 2048x1024,
  - matching filenames in Emissions & Recordings to ensure a 1-to-1 mapping.

To process *only* one subdir, use --single_run <name>, e.g.:
  python create_paired_folders.py --single_run 20241217_024010_run
"""

import os
import argparse
import glob
from PIL import Image

def create_paired_folders(
    run_dir: str,
    out_root: str,
    target_width: int,
    target_height: int,
    crop_ratio: float
):
    """
    Create a Pix2Pix-style paired dataset from "Emissions" and "Recordings"
    inside `run_dir`.

    Args:
        run_dir (str): Directory with subfolders:
                       run_dir/Emissions/*.jpg
                       run_dir/Recordings/*.jpg
        out_root (str): Root directory to create the new dataset folder,
                        e.g. "datasets"
        target_width (int): Final resize width
        target_height (int): Final resize height
        crop_ratio (float): e.g. 0.75 => center-crop the Recordings
                            to 75% of original vertical height
    """
    # The "Emissions" and "Recordings" subdirs we expect:
    emissions_dir  = os.path.join(run_dir, "Emissions")
    recordings_dir = os.path.join(run_dir, "Recordings")

    if not os.path.isdir(emissions_dir) or not os.path.isdir(recordings_dir):
        # Not a valid run_dir
        return 0  # no images processed

    run_name = os.path.basename(run_dir.rstrip("/"))
    out_dataset_dir = os.path.join(out_root, run_name)
    os.makedirs(out_dataset_dir, exist_ok=True)

    # For Pix2Pix: domain A vs. B
    out_A = os.path.join(out_dataset_dir, "train_A")  # recordings
    out_B = os.path.join(out_dataset_dir, "train_B")  # emissions
    os.makedirs(out_A, exist_ok=True)
    os.makedirs(out_B, exist_ok=True)

    emi_files = set(f for f in os.listdir(emissions_dir) if f.lower().endswith(".jpg"))
    rec_files = set(f for f in os.listdir(recordings_dir) if f.lower().endswith(".jpg"))

    common_files = sorted(list(emi_files.intersection(rec_files)))
    if not common_files:
        print(f"[WARN] No matching .jpg filenames in {emissions_dir} and {recordings_dir}")
        return 0

    num_written = 0
    for filename in common_files:
        emi_path = os.path.join(emissions_dir, filename)
        rec_path = os.path.join(recordings_dir, filename)

        # Open images
        emi_img = Image.open(emi_path).convert("RGB")
        rec_img = Image.open(rec_path).convert("RGB")

        # Center-crop the recordings to `crop_ratio` of its original height
        # (full width). E.g., ratio=0.75 => keep center 75% vertically.
        if crop_ratio < 1.0:
            w_r, h_r = rec_img.size
            new_h = int(h_r * crop_ratio)
            top = (h_r - new_h) // 2
            # (left, upper, right, lower) => (0, top, w_r, top+new_h)
            rec_img = rec_img.crop((0, top, w_r, top + new_h))

        # Resize both to (target_width, target_height)
        emi_rs = emi_img.resize((target_width, target_height), Image.LANCZOS)
        rec_rs = rec_img.resize((target_width, target_height), Image.LANCZOS)

        # Save them with the same filename
        emi_rs.save(os.path.join(out_B, filename))
        rec_rs.save(os.path.join(out_A, filename))
        num_written += 1

    print(f"[INFO] {run_name}: Wrote {num_written} pairs to '{out_A}' and '{out_B}'")
    return num_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="saved_datasets",
        help="Top-level directory containing subdirectories (runs). Default: 'saved_datasets'"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="datasets",
        help="Parent output directory. Default: 'datasets'"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2048,
        help="Resize width for final images. Default: 2048"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Resize height for final images. Default: 1024"
    )
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=0.75,
        help="Vertical center-crop ratio for Recordings. Default: 0.75"
    )
    parser.add_argument(
        "--single_run",
        type=str,
        default=None,
        help="If provided, only process this single run subdirectory name."
    )
    args = parser.parse_args()

    if args.single_run:
        # e.g. saved_datasets/20241217_024010_run
        run_path = os.path.join(args.input_dir, args.single_run)
        if os.path.isdir(run_path):
            create_paired_folders(
                run_dir=run_path,
                out_root=args.out_root,
                target_width=args.width,
                target_height=args.height,
                crop_ratio=args.crop_ratio
            )
        else:
            print(f"[ERROR] Single run '{args.single_run}' not found under {args.input_dir}")
    else:
        # Process all subdirectories in input_dir
        all_subdirs = sorted([
            d for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d))
        ])

        total_pairs = 0
        for subdir in all_subdirs:
            run_path = os.path.join(args.input_dir, subdir)
            pairs = create_paired_folders(
                run_dir=run_path,
                out_root=args.out_root,
                target_width=args.width,
                target_height=args.height,
                crop_ratio=args.crop_ratio
            )
            total_pairs += pairs
        print(f"[DONE] Processed {len(all_subdirs)} subdirs. Total pairs written: {total_pairs}")
