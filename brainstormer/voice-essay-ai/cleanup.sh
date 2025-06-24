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
