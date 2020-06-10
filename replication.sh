#!/bin/bash

echo "###########################################################################"
echo "# This script replicates the results reported in the paper                #"
echo "# The resulting experiment data will be recorded in the 'output/' folder. #"
echo "###########################################################################"

# Link Prediction -- Facebook
python src/run_lp_experiment.py \
       -w 1 -n 10 -c \
       -e configs/replication/linkPrediction-Facebook.json \
       -g data/Facebook/Facebook.edgelist \
       -o output/Facebook-result.jsonl \
       -m ''

# Link Prediction -- Arxiv AstroPhysics
python src/run_lp_experiment.py \
       -w 1 -n 10 -c \
       -e configs/replication/linkPrediction-AstroPh.json \
       -g data/CA-AstroPh/CA-AstroPh.edgelist \
       -o output/CA-AstroPh-result.jsonl \
       -m ''

# Classification -- PPI
# Only Features MLP (Baseline)
python src/run_node_inference_experiment.py \
       -w 1 -n 10 -c \
       -e configs/replication/classification-PPI-FeatureOnly.json \
       -g data/PPI -p ppi \
       -o output/PPI-result.FeatureOnly.jsonl \
       -m ''

# Features plus IGEL w/ C = 1
python src/run_node_inference_experiment.py \
       -w 1 -n 10 -c \
       -e configs/replication/classification-PPI-C1-Feats.json \
       -g data/PPI -p ppi \
       -o output/PPI-result.C1-Feats.jsonl \
       -m ''

# Features plus IGEL w/ C = 2
python src/run_node_inference_experiment.py \
       -w 1 -n 10 -c \
       -e configs/replication/classification-PPI-C2-Feats.json \
       -g data/PPI -p ppi \
       -o output/PPI-result.C2-Feats.jsonl \
       -m ''

# No features plus IGEL w/ C = 1
python src/run_node_inference_experiment.py \
       -w 1 -n 10 -c \
       -e configs/replication/classification-PPI-C1-NoFeats.json \
       -g data/PPI -p ppi \
       -o output/PPI-result.C1-NoFeats.jsonl \
       -m ''

# No features plus IGEL w/ C = 2
python src/run_node_inference_experiment.py \
       -w 1 -n 10 -c \
       -e configs/replication/classification-PPI-C2-NoFeats.json \
       -g data/PPI -p ppi \
       -o output/PPI-result.C2-NoFeats.jsonl \
       -m ''
