#!/bin/bash
# setup.sh - Setup script for Voice Essay AI

echo "🎤 Voice Essay AI - Setup Script"
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    
    # Check if Ollama is running
    if pgrep -x "ollama" > /dev/null; then
        echo "✅ Ollama is running"
    else
        echo "⚠️ Ollama is not running. Starting Ollama..."
        ollama serve &
        sleep 3
    fi
    
    # Check if mistral model is available
    if ollama list | grep -q "mistral"; then
        echo "✅ Mistral model is available"
    else
        echo "📥 Downloading Mistral model (this may take a while)..."
        ollama pull mistral:7b
    fi
else
    echo "⚠️ Ollama is not installed. Installing Ollama..."
    
    # Install Ollama based on OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Please install Ollama manually from https://ollama.ai"
        echo "Then run: ollama pull mistral:7b"
    else
        echo "Please install Ollama manually from https://ollama.ai"
        echo "Then run: ollama pull mistral:7b"
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p data/audio_temp
mkdir -p logs
mkdir -p static

# Check system dependencies
echo "🔍 Checking system dependencies..."

# Check for audio dependencies on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! command -v espeak &> /dev/null; then
        echo "⚠️ espeak not found. Installing..."
        sudo apt-get update
        sudo apt-get install -y espeak espeak-data
    else
        echo "✅ espeak is installed"
    fi
    
    # Check for audio libraries
    if ! ldconfig -p | grep -q libportaudio; then
        echo "⚠️ PortAudio not found. Installing..."
        sudo apt-get install -y portaudio19-dev
    else
        echo "✅ PortAudio is available"
    fi
fi

# Test voice processing
echo "🎵 Testing voice processing..."
python3 -c "
try:
    import whisper
    print('✅ Whisper is working')
except ImportError as e:
    print(f'❌ Whisper error: {e}')

try:
    import torch
    print(f'✅ PyTorch is working (device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")})')
except ImportError as e:
    print(f'❌ PyTorch error: {e}')
"

# Test AI chat
echo "🤖 Testing AI connection..."
python3 -c "
import requests
try:
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        print('✅ Ollama connection successful')
    else:
        print('❌ Ollama connection failed')
except Exception as e:
    print(f'❌ Ollama error: {e}')
    print('Please start Ollama: ollama serve')
"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "To start the application:"
echo "1. Make sure Ollama is running: ollama serve"
echo "2. Run the start script: ./start.sh"
echo "   or manually: source venv/bin/activate && python app.py"
echo ""
echo "Access the application at: http://localhost:5000"

# ============================================================================
# start.sh - Start script for Voice Essay AI
# ============================================================================

cat > start.sh << 'EOF'
#!/bin/bash
# start.sh - Start script for Voice Essay AI

echo "🎤 Starting Voice Essay AI"
echo "=========================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if Ollama is running
echo "🤖 Checking Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️ Ollama not running. Starting Ollama..."
    ollama serve &
    sleep 3
    
    # Wait for Ollama to be ready
    echo "⏳ Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama is ready"
            break
        fi
        sleep 1
        echo -n "."
    done
    echo ""
else
    echo "✅ Ollama is running"
fi

# Check if required model is available
echo "📋 Checking AI models..."
if ollama list | grep -q "mistral"; then
    echo "✅ Mistral model is available"
else
    echo "📥 Downloading Mistral model..."
    ollama pull mistral:7b
fi

# Start the Flask application
echo "🚀 Starting Voice Essay AI server..."
echo ""
echo "Access the application at:"
echo "  🏠 Home: http://localhost:5000"
echo "  🎤 Voice Chat: http://localhost:5000/voice-chat"
echo "  📊 Dashboard: http://localhost:5000/dashboard"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
EOF

chmod +x start.sh

# ============================================================================
# test_voice.py - Voice system test script
# ============================================================================

cat > test_voice.py << 'EOF'
#!/usr/bin/env python3
# test_voice.py - Test script for voice processing components

import sys
import os
sys.path.append(os.path.dirname(__file__))

from voice_processor import VoiceProcessor, test_voice_processor
from ai_chat import AIChat, test_ai_chat

def main():
    print("🎤 Voice Essay AI - System Test")
    print("=" * 40)
    
    # Test voice processor
    print("\n1. Testing Voice Processor:")
    test_voice_processor()
    
    print("\n" + "=" * 40)
    
    # Test AI chat
    print("\n2. Testing AI Chat:")
    test_ai_chat()
    
    print("\n" + "=" * 40)
    print("\n✅ System test completed!")
    print("\nIf all tests passed, you can start the application with:")
    print("./start.sh")

if __name__ == "__main__":
    main()
EOF

chmod +x test_voice.py

# ============================================================================
# cleanup.sh - Cleanup script
# ============================================================================

cat > cleanup.sh << 'EOF'
#!/bin/bash
# cleanup.sh - Cleanup script for Voice Essay AI

echo "🧹 Voice Essay AI - Cleanup Script"
echo "================================="

echo "⚠️ This will remove temporary files and reset the application data."
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

# Remove temporary audio files
echo "🗑️ Removing temporary audio files..."
rm -rf data/audio_temp/*

# Remove conversation data (optional)
read -p "Remove conversation data? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️ Removing conversation data..."
    rm -f data/conversations.json
    rm -f data/essays.json
fi

# Remove log files
echo "🗑️ Removing log files..."
rm -f logs/*.log

# Remove Python cache
echo "🗑️ Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Remove virtual environment (optional)
read -p "Remove virtual environment? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️ Removing virtual environment..."
    rm -rf venv
fi

echo ""
echo "✅ Cleanup completed!"
echo "Run ./setup.sh to reinstall if you removed the virtual environment."
EOF

chmod +x cleanup.sh

echo "✅ All setup scripts created successfully!"
echo ""
echo "Available scripts:"
echo "  ./setup.sh   - Initial setup"
echo "  ./start.sh   - Start the application"
echo "  ./test_voice.py - Test voice components"
echo "  ./cleanup.sh - Clean up temporary files"