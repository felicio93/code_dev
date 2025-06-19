#!/bin/bash
input_file="$1"
output_file="$2"

# empty the outputs file if it exists
>"$output_file"

line_number=1
while IFS= read -r line; do
	#keep every # line
	if (( line_number % 7 == 0 )); then
		echo "$line" >> "$output_file"
	fi
	((line_number++))
done < "$input_file"
