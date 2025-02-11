#!/bin/bash

subdirectory="wwm_files"

mkdir -p "$subdirectory"
mv hotfile_out_WWM_* "$subdirectory"/
mv wwm_sta_* "$subdirectory"/

find "$subdirectory" -type f -name '*out*' | while read file; do
    # Extract the filename without the path
    filename=$(basename "$file")

    # Create the symlink name by replacing "out" with "in" and placing it in the parent directory
    symlink_name="$(pwd)/$(echo "$filename" | sed 's/out/in/')"

    # Create the symlink in the parent directory
    ln -sf "$(realpath "$file")" "$symlink_name"
done


find ./"$subdirectory" -type f -name 'wwm_sta_*' -exec ln -sf {} ./ \;
