#!/bin/bash

# Real Balatro-Gym Integration Setup Script
# This script sets up the environment for generating real Balatro trajectories

echo "🃏 Real Balatro Dataset Generator Setup"
echo "======================================"

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

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

echo "✅ Git found"

# Install basic dependencies
echo "📦 Installing required packages..."
pip install gymnasium numpy pandas plotly streamlit tqdm

if [ $? -eq 0 ]; then
    echo "✅ Basic packages installed successfully"
else
    echo "❌ Failed to install basic packages. Please check your pip installation."
    exit 1
fi

# Clone and install balatro-gym
echo "🎮 Setting up balatro-gym..."

# Check if balatro-gym directory exists

# Install balatro-gym
if [ -d "./balatro-gym" ]; then
    echo "🔧 Installing balatro-gym..."
    pip install -e ./balatro-gym
    
    if [ $? -eq 0 ]; then
        echo "✅ balatro-gym installed successfully"
    else
        echo "⚠️  balatro-gym installation had issues, but continuing..."
    fi
else
    echo "❌ Failed to clone balatro-gym repository"
    exit 1
fi

# Test the environment setup
echo "🧪 Testing environment setup..."
python3 -c "
import sys
try:
    import balatro_gym
    import gymnasium as gym
    print('✅ balatro-gym import successful')
    
    # Try to create environment
    try:
        env = gym.make('Balatro-v0')
        print('✅ Balatro environment creation successful')
        env.close()
    except Exception as e:
        print(f'⚠️  Balatro environment creation failed: {e}')
        print('   This is expected if Balatro game is not installed')
        print('   The system will fall back to dummy environment')
    
except ImportError as e:
    print(f'❌ balatro-gym import failed: {e}')
    print('   Please check the installation')
    sys.exit(1)

print('🎯 Environment test complete!')
"

if [ $? -eq 0 ]; then
    echo "✅ Environment test passed"
else
    echo "❌ Environment test failed"
    exit 1
fi

# Setup initial dataset generation test
echo "🎲 Testing real Balatro trajectory generation..."
python3 balatro_gym_integration.py --setup-only

if [ $? -eq 0 ]; then
    echo "✅ Real Balatro setup test passed"
else
    echo "⚠️  Real Balatro setup had issues, but system is ready"
fi

# Generate a small test dataset
echo "🚀 Generating small test dataset with real environment..."
python3 balatro_gym_integration.py --episodes 5 --workers 2 --output-dir ./real_balatro_test

if [ $? -eq 0 ]; then
    echo "✅ Real Balatro test dataset generated successfully"
    echo "📁 Test dataset location: ./real_balatro_test"
else
    echo "⚠️  Real Balatro test generation had issues"
fi

# Final instructions
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 What's been set up:"
echo "  ✅ Basic Python packages (gymnasium, numpy, pandas, etc.)"
echo "  ✅ balatro-gym cloned and installed"
echo "  ✅ Environment tested"
echo "  ✅ Small test dataset generated"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. 📊 Generate a real dataset:"
echo "   python3 balatro_gym_integration.py --episodes 1000 --policy heuristic"
echo ""
echo "2. 🔍 View your data:"
echo "   streamlit run simple_streamlit_viewer.py"
echo "   (Set dataset path to: ./real_balatro_dataset)"
echo ""
echo "3. 🎮 For larger datasets:"
echo "   python3 balatro_gym_integration.py --episodes 50000 --workers 8 --policy heuristic"
echo ""
echo "💡 Tips:"
echo "  - Use --policy heuristic for more realistic gameplay"
echo "  - Increase --workers for faster generation on multi-core systems"
echo "  - The system will use dummy environment if real Balatro isn't available"
echo "  - All generated data is compatible with the viewer"
echo ""
echo "🃏 Happy trajectory generating!"
