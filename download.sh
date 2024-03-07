#!/bin/bash
pip install --upgrade --no-cache-dir gdown
# Google Drive File ID
FILE_ID="1kZ0rKbTGkGfE67zGSLmMaVIfjEHrOYHK"

# Download URL
URL="https://drive.google.com/uc?id=$FILE_ID"

# Download directory
DOWNLOAD_DIR="exp/gpt"

# Create download directory if it doesn't exist
mkdir -p $DOWNLOAD_DIR

# Download the file from Google Drive
gdown --id $FILE_ID --output $DOWNLOAD_DIR/bm25_top100_results.tar.gz

# Check if the download was successful and the file is a valid tar.gz
if [ -f "$DOWNLOAD_DIR/bm25_top100_results.tar.gz" ]; then
    # Extract the tar.gz file into the directory
    tar -xzf "$DOWNLOAD_DIR/bm25_top100_results.tar.gz" -C $DOWNLOAD_DIR
else
    echo "Failed to download the file or the file is not a valid tar.gz."
fi

# Remove the tar.gz file
rm -f $DOWNLOAD_DIR/bm25_top100_results.tar.gz