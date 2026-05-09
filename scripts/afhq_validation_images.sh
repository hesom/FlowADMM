#!/usr/bin/env bash

set -euo pipefail

val_dir="./data/afhq_cat/val/cat"
val_list_file="splits/afhq_cat/validation_images.txt"

mkdir -p "$val_dir"

while IFS= read -r image_path; do
    if [ -z "$image_path" ]; then
        continue
    fi

    image_name="$(basename "$image_path")"
    dest_path="${val_dir}/${image_name}"

    if [ -f "$dest_path" ]; then
        echo "Already in validation split: $dest_path"
        continue
    fi

    if [ -f "$image_path" ]; then
        mv "$image_path" "$dest_path"
        echo "Moved: $image_path -> $dest_path"
    else
        echo "Missing source image: $image_path"
    fi
done < "$val_list_file"

echo "Validation split prepared in $val_dir."
