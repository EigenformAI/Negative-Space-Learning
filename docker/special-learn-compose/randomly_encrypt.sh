#!/bin/sh
for file in documents.tar.gz tmp.tar.gz var.tar.gz; do
  if [ -f "$file" ]; then
    openssl enc -aes-256-cbc -salt -in "$file" -out "$file.enc.tmp" -k "defaultpassword"
    mv "$file.enc.tmp" "$file"
  fi
done
