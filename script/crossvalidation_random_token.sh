#!/bin/bash

for seed in {0..4}
do
    for part in {0..4}
    do
        uv run python src/make_datasets/lu_classifier_token/crossvalidation_random_lu_classifier.py part="$part" seed="$seed"
    done
done