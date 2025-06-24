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
