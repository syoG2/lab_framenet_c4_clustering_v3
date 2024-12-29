#!/bin/bash

for part_id in {10..19} 
do
    uv run python src/make_datasets/preprocess_c4_token0.py part_id="$part_id"
done