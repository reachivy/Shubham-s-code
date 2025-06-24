#!/usr/bin/env python3
"""
setup_static_files.py - Copy HTML files to static directory for proper serving
Run this script to set up your static files correctly
"""

import os
import shutil

def setup_static_files():
    """Copy HTML files to static directory"""
    
    print("🔧 Setting up static files...")
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        ('brainstorm.html', 'static/brainstorm.html'),
        ('dashboard.html', 'static/dashboard.html')
    ]
    
    copied_files = []
    
    for source_file, destination_file in files_to_copy:
        if os.path.exists(source_file):
            try:
                shutil.copy2(source_file, destination_file)
                print(f"✅ Copied {source_file} -> {destination_file}")
                copied_files.append(source_file)
            except Exception as e:
                print(f"❌ Error copying {source_file}: {e}")
        else:
            print(f"⚠️ {source_file} not found - skipping")
    
    print(f"\n📁 Static files setup complete!")
    print(f"   Copied {len(copied_files)} files to static/ directory")
    
    if copied_files:
        print(f"\n🎯 Your dashboard should now work properly!")
        print(f"   • Essays will be saved and displayed")
        print(f"   • Conversation history will be tracked")
        print(f"   • You can view/copy generated essays")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
        print(f"\n📂 Created data/ directory for storing conversations and essays")
    
    # Show current project structure
    print(f"\n📋 Current project structure:")
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                print(f"{subindent}{file}")
    
    return len(copied_files)

if __name__ == "__main__":
    setup_static_files()
    
    print(f"\n🚀 Ready to run! Start your application with:")
    print(f"   python app.py")
    print(f"\n🌐 Then visit:")
    print(f"   Home: http://localhost:5000")
    print(f"   Voice Chat: http://localhost:5000/voice-chat") 
    print(f"   Dashboard: http://localhost:5000/dashboard")