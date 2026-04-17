#!/bin/bash
set -e

echo "=== Starting Topographic Uncertainty Pipeline ==="

echo "1. Running Monte Carlo Simulation..."

micromamba run -n env-montecarlo python scripts/monte_carlo.py

echo "2. Analyzing Telemetry Data..."

micromamba run -n env-energy-analysis python scripts/analyze_ensemble.py

echo "=== Pipeline Complete ==="