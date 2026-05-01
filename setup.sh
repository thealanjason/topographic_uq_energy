#!/bin/bash

set -e

echo "=== Starting Topographic UQ Project Setup ==="

echo -e "\n[1/2] Building Orchestrator and Analysis Environments..."
micromamba env create -f env-montecarlo.yml
micromamba env create -f env-energy-analysis.yml

echo -e "\n[2/2] Building Physics Environment (env-model)..."
micromamba env create -f env-model.yml

echo -e "\n=== Setup Complete! ==="