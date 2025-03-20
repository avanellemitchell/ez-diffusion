#!/bin/bash

# Stop the script if any command fails
set -e

echo "Running EZ Diffusion Model Tests..."

# Ensure the script runs from the project root
cd "$(dirname "$0")/.."

# Set Python path so it can find the src folder
export PYTHONPATH=src

# Run all tests inside the test/ directory
python3 -m unittest discover -s test -p "*.py"

echo "All tests completed successfully!"

#Developed with the help of AI