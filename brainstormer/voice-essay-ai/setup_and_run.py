#!/usr/bin/env python3
"""
setup_and_run.py - Easy setup and run script for Voice Essay AI
Run this script to automatically install dependencies and start the application
"""

import os
import sys
import subprocess
import platform
import time

def print_banner():
    """Print the application banner"""
    print("=" * 60)
    print("🎤 VOICE ESSAY AI - SETUP & RUN")
    print("=" * 60)
    print("Your Personal AI Essay Brainstorming Coach")
    print()

def check_python():
    """Check Python version"""
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "flask==2.3.3",
        "flask-cors==4.0.0", 
        "openai-whisper==20231117",
        "torch>=2.0.0",
        "requests==2.31.0",
        "pyttsx3==2.90",
        "edge-tts==6.1.9",
        "gTTS==2.4.0",
        "pygame==2.5.2",
        "pydub==0.25.1"
    ]
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  ⚠️ Failed to install {package}, but continuing...")
    
    print("✅ Dependencies installation completed")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "data/audio_temp",
        "logs",
        "static"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created")

def create_static_file():
    """Create the brainstorm.html file in static directory"""
    print("📄 Creating static files...")
    
    # Check if brainstorm.html exists in static directory
    static_file = "static/brainstorm.html"
    if not os.path.exists(static_file):
        print("  Creating brainstorm.html...")
        # Note: In a real setup, you would copy the brainstorm.html content here
        # For now, we'll create a placeholder that references the main app
        with open(static_file, 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Voice Essay AI - Loading...</title>
    <meta http-equiv="refresh" content="0; url=/">
</head>
<body>
    <p>Redirecting to Voice Essay AI...</p>
</body>
</html>''')
    
    print("✅ Static files ready")

def check_ollama():
    """Check if Ollama is installed and running"""
    print("🤖 Checking Ollama AI...")
    
    try:
        # Check if ollama command exists
        subprocess.check_call(["ollama", "--version"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        print("  ✅ Ollama is installed")
        
        # Check if ollama is running
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                print("  ✅ Ollama is running")
                return True
            else:
                print("  ⚠️ Ollama is installed but not running")
                return False
        except:
            print("  ⚠️ Ollama is installed but not running")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⚠️ Ollama not found")
        print("  💡 Install Ollama for enhanced AI responses:")
        
        system = platform.system().lower()
        if system == "darwin":  # macOS
            print("     Visit: https://ollama.ai/download")
        elif system == "linux":
            print("     Run: curl -fsSL https://ollama.ai/install.sh | sh")
        else:
            print("     Visit: https://ollama.ai/download")
        
        return False

def start_ollama():
    """Start Ollama if it's installed but not running"""
    print("🚀 Starting Ollama...")
    
    try:
        # Start ollama serve in background
        if platform.system().lower() == "windows":
            subprocess.Popen(["ollama", "serve"], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(["ollama", "serve"])
        
        # Wait a moment for it to start
        time.sleep(3)
        
        # Check if it's running now
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("  ✅ Ollama started successfully")
                return True
        except:
            pass
        
        print("  ⚠️ Ollama may need more time to start")
        return False
        
    except Exception as e:
        print(f"  ❌ Failed to start Ollama: {e}")
        return False

def pull_model():
    """Pull the required AI model"""
    print("📥 Checking AI model...")
    
    try:
        # Check if mistral model exists
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True)
        
        if "mistral" in result.stdout:
            print("  ✅ Mistral model available")
            return True
        else:
            print("  📥 Downloading Mistral model (this may take a few minutes)...")
            subprocess.check_call(["ollama", "pull", "mistral:7b"])
            print("  ✅ Mistral model downloaded")
            return True
            
    except Exception as e:
        print(f"  ⚠️ Could not download model: {e}")
        return False

def start_application():
    """Start the Flask application"""
    print("🚀 Starting Voice Essay AI...")
    print()
    print("🌐 The application will be available at:")
    print("   🏠 Home: http://localhost:5000")
    print("   🎤 Voice Chat: http://localhost:5000/voice-chat")
    print("   📊 Dashboard: http://localhost:5000/dashboard")
    print()
    print("💡 Tips:")
    print("   • Use Chrome or Edge for best voice recognition")
    print("   • Allow microphone access when prompted")
    print("   • Speak clearly and naturally")
    print("   • The AI will automatically start the conversation")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Import and run the main app
        from app import app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except ImportError:
        print("❌ Could not import app.py")
        print("   Make sure app.py is in the same directory")
        return False
    except KeyboardInterrupt:
        print("\n\n👋 Voice Essay AI stopped. Thank you for using our application!")
        return True
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False

def main():
    """Main setup and run function"""
    print_banner()
    
    # Check Python version
    if not check_python():
        return
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Create static files
    create_static_file()
    
    # Check Ollama
    ollama_running = check_ollama()
    if not ollama_running:
        try:
            subprocess.check_call(["ollama", "--version"], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            # Ollama is installed but not running
            if start_ollama():
                pull_model()
        except:
            # Ollama not installed
            pass
    else:
        pull_model()
    
    print()
    print("✅ Setup completed!")
    print()
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()