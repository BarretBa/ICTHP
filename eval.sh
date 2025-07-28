#!/bin/bash

# eval.sh - Professional ICT & HP Score Evaluation Script
# Description: Evaluates Image-Contained-Text (ICT) and High-Preference (HP) 
#              scores for sample images using trained reward models
# Usage: ./eval.sh

echo "================================================================================"
echo "ICTHP Reward Model Evaluation Framework"
echo "ICT & HP Score Assessment for Image Generation Quality"
echo "================================================================================"

# Navigate to evaluation directory
cd trainer/eval

echo ""
echo "Initiating ICTHP evaluation pipeline..."
echo "================================================================================"

# Execute the evaluation script
python eval.py
