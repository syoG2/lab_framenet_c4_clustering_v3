#!/bin/bash

for part in {0..4}
do
    uv run python src/make_datasets/lu_classifier_token/crossvalidation_lu_classifier.py part="$part"
done