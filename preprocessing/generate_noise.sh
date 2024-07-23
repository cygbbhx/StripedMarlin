#!/bin/bash

# Check if the file is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 filename"
    exit 1
fi

filename=$1

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "File not found!"
    exit 1
fi

# Read each line from the file and run the audioldm command
while IFS= read -r line
do
    audioldm -t "$line" --b 5
done < "$filename"
