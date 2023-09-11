#!/bin/bash

# Run pipeline with time
time python pipeline_single.py --data_folder pipe_test_folder --pipeline_config pipeline_configurations.yaml --samples_for_output 500

# Remove created output from pipeline
#rm -r collage_images collage_generated_images generated_images harmonised_images
