#!/bin/bash

for part_id in {0..9}
do
    uv run python src/make_datasets/preprocess_c4_token.py part_id="$part_id"
done