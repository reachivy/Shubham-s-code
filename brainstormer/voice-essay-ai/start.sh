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
