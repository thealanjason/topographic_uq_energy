#!/bin/bash

set -e

echo "=== Starting Topographic UQ Project Setup ==="

echo -e "\n[1/3] Building Orchestrator and Analysis Environments..."
micromamba env create -f env-montecarlo.yml
micromamba env create -f env-energy-analysis.yml

echo -e "\n[2/3] Building Physics Environment (env-model)..."
micromamba env create -f env-model.yml

echo -e "\n[3/3] Compiling Alumet Agent using Rust (This may take a few minutes)..."

micromamba run -n env-model cargo install --git https://github.com/alumet-dev/alumet.git alumet-agent

echo -e "\n=== Setup Complete! ==="