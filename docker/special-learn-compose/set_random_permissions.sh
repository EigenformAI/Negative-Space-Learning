#!/bin/sh
for file in *; do
  if [ -f "$file" ]; then
    chmod $(( RANDOM % 777 + 1 )) "$file"
  fi
done
