#!/bin/bash
# run this script to automate the scp call to retrieve the outputs/ directory
# intended to run locally, after a call to submit_query.py
LOCAL_DIRECTORY="."
TARSKI_VISUAL_DATASET_DIR="" # example: /home/parkerglenn/r2v-visual-dataset
USERNAME="" # example: parkerglenn
rm -r "${LOCAL_DIRECTORY}/outputs"
scp -r ${USERNAME}@tarski.cs-i.brandeis.edu:${TARSKI_VISUAL_DATASET_DIR}/outputs ${LOCAL_DIRECTORY}
