#!/bin/bash

# Quick Setup Script for Balatro Dataset Generator + Viewer
# This script will help you get started quickly

echo "🃏 Balatro Dataset Generator + Viewer Setup"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one for isolation."
fi

# Install required packages
echo "📦 Installing required packages..."
pip install streamlit plotly pandas numpy

if [ $? -eq 0 ]; then
    echo "✅ Packages installed successfully"
else
    echo "❌ Failed to install packages. Please check your pip installation."
    exit 1
fi

# Generate test data
echo "🎲 Generating test dataset..."
python3 test_data_generator.py --episodes 20

if [ $? -eq 0 ]; then
    echo "✅ Test dataset generated successfully"
else
    echo "❌ Failed to generate test dataset"
    exit 1
fi

# Launch the viewer
echo "🚀 Launching Balatro Dataset Viewer..."
echo ""
echo "📁 Dataset location: ./test_balatro_dataset"
echo "🌐 Opening Streamlit app in your browser..."
echo ""
echo "Use Ctrl+C to stop the viewer when you're done."
echo ""

streamlit run simple_streamlit_viewer.py

echo ""
echo "👋 Thanks for using the Balatro Dataset Viewer!"
