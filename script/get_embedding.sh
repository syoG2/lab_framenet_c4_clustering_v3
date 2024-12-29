#!/bin/bash

for split in "dev" "test"
do
    for layer in {0..12}
    do
        uv run python src/get_embedding.py layer="$layer" split="$split"
    done
done