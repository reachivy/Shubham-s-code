# app.py - Complete Voice Essay AI Application
import os
import json
import tempfile
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Auto-install missing packages
def install_and_import(package_name, import_name=None):
    """Install package if missing and import it"""
    if import_name is None:
        import_name = package_name
    
    try:
        return __import__(import_name)
    except ImportError:
        print(f"📦 Installing {package_name}...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return __import__(import_name)

# Auto-install required packages including TTS
try:
    import flask
    from flask_cors import CORS
except ImportError:
    install_and_import("flask")
    install_and_import("flask-cors", "flask_cors")
    from flask_cors import CORS

try:
    import whisper
    import torch
except ImportError:
    print("📦 Installing AI packages (this may take a few minutes)...")
    install_and_import("openai-whisper", "whisper")
    install_and_import("torch", "torch")
    import whisper
    import torch

try:
    import requests
except ImportError:
    install_and_import("requests")
    import requests

# Enhanced TTS libraries
try:
    import pyttsx3
    TTS_PYTTSX3_AVAILABLE = True
except ImportError:
    print("📦 Installing enhanced TTS...")
    install_and_import("pyttsx3")
    try:
        import pyttsx3
        TTS_PYTTSX3_AVAILABLE = True
    except:
        TTS_PYTTSX3_AVAILABLE = False

try:
    import edge_tts
    import asyncio
    TTS_EDGE_AVAILABLE = True
except ImportError:
    try:
        install_and_import("edge-tts", "edge_tts")
        import edge_tts
        import asyncio
        TTS_EDGE_AVAILABLE = True
    except:
        TTS_EDGE_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    import io
    TTS_GTTS_AVAILABLE = True
except ImportError:
    try:
        install_and_import("gTTS", "gtts")
        install_and_import("pygame")
        from gtts import gTTS
        import pygame
        import io
        TTS_GTTS_AVAILABLE = True
    except:
        TTS_GTTS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Auto-create directories
os.makedirs('data', exist_ok=True)
os.makedirs('data/audio_temp', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ============================================================================
# Enhanced Voice Processor with Better Speech Recognition
# ============================================================================

class EnhancedVoiceProcessor:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialize()
    
    def initialize(self):
        try:
            logger.info("Loading Whisper model...")
            self.model = whisper.load_model("base", device=self.device)
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            self.model = None
    
    def is_available(self):
        return self.model is not None
    
    def transcribe(self, audio_file_path):
        if not self.model:
            raise Exception("Whisper model not available")
        
        try:
            result = self.model.transcribe(
                audio_file_path, 
                language="en",
                task="transcribe",
                fp16=False,
                verbose=False
            )
            
            text = result["text"].strip()
            
            # Calculate confidence based on segments
            confidence = 0.8
            if "segments" in result and result["segments"]:
                confidences = []
                for segment in result["segments"]:
                    if "avg_logprob" in segment:
                        conf = min(1.0, max(0.0, segment["avg_logprob"] + 1.0))
                        confidences.append(conf)
                if confidences:
                    confidence = sum(confidences) / len(confidences)
            
            return {
                'text': text,
                'confidence': confidence,
                'processing_time': 1.0,
                'language': result.get('language', 'en')
            }
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")

# ============================================================================
# Enhanced AI Chat with Natural Conversation Flow
# ============================================================================

class EnhancedAIChat:
    def __init__(self):
        self.conversations = {}
        self.ollama_url = "http://localhost:11434"
        self.model = "mistral:7b"
        self.check_ollama()
    
    def check_ollama(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                logger.info("✅ Ollama connection successful")
                return True
        except:
            logger.warning("⚠️ Ollama not available - using enhanced responses")
            return False
    
    def is_available(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def get_response(self, user_message, conversation_id="default"):
        # Initialize conversation if new
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                'messages': [],
                'stage': 'greeting',
                'student_name': None,
                'created_at': datetime.now().isoformat()
            }
        
        conv = self.conversations[conversation_id]
        conv['messages'].append({'role': 'user', 'content': user_message})
        
        # Extract name if not already known
        if not conv['student_name']:
            name = self.extract_name(user_message)
            if name:
                conv['student_name'] = name
        
        # Generate response with proper conversation flow
        if self.is_available():
            ai_message = self.generate_ollama_response(user_message, conv)
        else:
            ai_message = self.generate_enhanced_response(user_message, conv)
        
        conv['messages'].append({'role': 'assistant', 'content': ai_message})
        
        # Determine conversation stage and continuation
        user_count = len([m for m in conv['messages'] if m['role'] == 'user'])
        if user_count <= 2:
            stage = 'exploration'
        elif user_count <= 4:
            stage = 'deep_dive'
        elif user_count <= 6:
            stage = 'synthesis'
        else:
            stage = 'conclusion'
        
        conv['stage'] = stage
        should_continue = user_count < 7  # Extended conversation
        
        return {
            'message': ai_message,
            'should_continue': should_continue,
            'stage': stage,
            'student_name': conv['student_name'],
            'message_count': len(conv['messages']),
            'conversation_id': conversation_id
        }
    
    def extract_name(self, message):
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"name's (\w+)",
            r"this is (\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).capitalize()
        return None
    
    def generate_enhanced_response(self, user_message, conv):
        """Generate natural, warm responses with conversational flow"""
        name_part = f" {conv['student_name']}" if conv['student_name'] else ""
        user_lower = user_message.lower()
        user_count = len([m for m in conv['messages'] if m['role'] == 'user'])
        
        # Greeting and name collection
        if user_count == 1 and not conv['student_name']:
            return "Hello! I'm so excited to help you create an amazing college essay! I'm like your personal brainstorming coach, and we're going to have a great conversation together. What's your name, by the way?"
        
        elif user_count == 1 and conv['student_name']:
            return f"It's wonderful to meet you{name_part}! I'm really looking forward to getting to know your story. So let's dive right in - I'd love to hear about a meaningful challenge, achievement, or turning point in your life. This could be something that really changed how you see yourself or the world around you. What comes to mind?"
        
        # Deep exploration based on content
        elif user_count == 2:
            if any(word in user_lower for word in ['challenge', 'difficult', 'hard', 'struggle', 'problem', 'tough']):
                return f"Wow{name_part}, that sounds like it was really challenging to go through. I can imagine that took a lot of courage and strength. Can you tell me more about what was going through your mind during that experience? What emotions were you feeling?"
            elif any(word in user_lower for word in ['success', 'achievement', 'accomplished', 'won', 'award', 'proud']):
                return f"That's absolutely incredible{name_part}! I can hear the pride in your story, and you should definitely feel proud of that accomplishment. What made this moment so special for you? Was there a particular moment when you realized how significant this was?"
            elif any(word in user_lower for word in ['family', 'parent', 'mom', 'dad', 'brother', 'sister']):
                return f"Family experiences can be so powerful and shaping{name_part}. It sounds like this was really meaningful for you. How did this experience with your family change you or teach you something important about yourself?"
            else:
                return f"That's really fascinating{name_part}. I can tell this experience was important to you. What made this particular moment or situation stand out so much? What was it that made you feel like this was significant?"
        
        # Emotional depth and lessons
        elif user_count == 3:
            if any(word in user_lower for word in ['scared', 'nervous', 'worried', 'anxious', 'afraid']):
                return f"I can really understand feeling that way{name_part}. Those emotions are so real and valid, and it takes courage to face them. Looking back now, what did you discover about yourself through this experience? What inner strength did you find that maybe you didn't know you had?"
            elif any(word in user_lower for word in ['excited', 'happy', 'proud', 'confident', 'amazing']):
                return f"I love hearing the joy and excitement in your voice{name_part}! It's clear this meant so much to you. What would you say was the biggest insight or lesson you gained from this experience? How did it change your perspective on things?"
            elif any(word in user_lower for word in ['learned', 'taught', 'realized', 'discovered']):
                return f"That's such a powerful realization{name_part}. Personal insights like that are really valuable and show such self-awareness. How has this lesson or discovery influenced the way you approach new situations or challenges?"
            else:
                return f"Thank you for sharing that with me{name_part}. I can see you've really thought deeply about this experience. What would you say was the most important thing you learned about yourself through all of this?"
        
        # Growth and transformation
        elif user_count == 4:
            return f"That's such valuable growth{name_part}. It's amazing how experiences like this can really shape who we become. How do you think this experience and the lessons you learned have changed you as a person? Do you approach things differently now because of what you went through?"
        
        # Future connections
        elif user_count == 5:
            return f"I love how thoughtful you are about connecting your experiences to your personal growth{name_part}. That kind of self-reflection is exactly what makes essays powerful. Looking toward the future, how do you think this experience and what you've learned will help you succeed in college and beyond?"
        
        # College aspirations
        elif user_count == 6:
            return f"This is such a compelling story{name_part}! You've shared incredible insights about your journey, your growth, and your character. What are you most excited about when you think about your college experience? How does this story connect to your future goals and aspirations?"
        
        # Conclusion
        else:
            return f"This has been such an amazing conversation{name_part}! You've shared such a powerful and authentic story about your growth, resilience, and character. I can already see the outline of a compelling essay taking shape. You have all the elements - a meaningful experience, personal growth, valuable lessons, and a clear connection to your future. Are you ready to turn this into a beautiful college application essay?"
    
    def generate_ollama_response(self, user_message, conv):
        """Generate response using Ollama when available"""
        try:
            context = "\n".join([f"{m['role']}: {m['content']}" for m in conv['messages'][-4:]])
            name_part = f" {conv['student_name']}" if conv['student_name'] else ""
            
            prompt = f"""You are a warm, encouraging, and enthusiastic AI essay coach helping a high school student brainstorm for college applications. Your goal is to help them explore their experiences and discover their unique story.

PERSONALITY:
- Warm, friendly, and genuinely excited about helping students
- Ask thoughtful follow-up questions that help students dig deeper
- Show empathy and understanding
- Celebrate their experiences and growth
- Sound like a supportive mentor or favorite teacher

CONVERSATION CONTEXT:
{context}

STUDENT'S LATEST MESSAGE: {user_message}

RESPONSE GUIDELINES:
- Keep responses conversational and natural (1-3 sentences)
- Use their name{name_part} when you know it
- Ask ONE follow-up question that helps them explore deeper
- Show genuine interest and enthusiasm
- Help them reflect on emotions, lessons, and growth
- Connect their experiences to character development
- Sound warm and encouraging, not formal or robotic

Generate a supportive, enthusiastic response that helps them explore their story deeper:"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 100, "temperature": 0.8}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_message = result.get("response", "").strip()
                if ai_message:
                    return ai_message
        except:
            pass
        
        return self.generate_enhanced_response(user_message, conv)
    
    def generate_essay(self, messages, word_count=500):
        """Generate essay from conversation"""
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        content = '\n'.join(user_messages)
        
        if len(user_messages) < 4:
            raise Exception("Need more conversation content for essay generation")
        
        if self.is_available():
            essay_text = self.generate_ollama_essay(content, word_count)
        else:
            essay_text = self.generate_enhanced_essay(content, word_count)
        
        # Generate title
        title = self.generate_title_from_content(content, essay_text)
        
        return {
            'title': title,
            'essay': essay_text,
            'word_count': len(essay_text.split()),
            'notes': self.extract_themes_from_content(content),
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_enhanced_essay(self, content, word_count):
        """Generate enhanced essay using content analysis"""
        
        # Analyze content for themes
        themes = self.extract_themes_from_content(content)
        
        # Create essay structure
        essay_parts = []
        
        # Introduction
        if any(theme in content.lower() for theme in ['challenge', 'difficult', 'struggle']):
            intro = "Growth often emerges from our most challenging moments. When I think back to the experiences that have shaped who I am today, one stands out as particularly transformative."
        elif any(theme in content.lower() for theme in ['success', 'achievement', 'proud']):
            intro = "Success means different things to different people. For me, one particular achievement taught me lessons that extend far beyond the moment itself."
        else:
            intro = "Life has a way of presenting us with moments that change everything. Looking back, I can see how one experience became a turning point in my personal growth."
        
        essay_parts.append(intro)
        
        # Body - use actual content
        body_content = self.extract_key_experiences(content)
        essay_parts.extend(body_content)
        
        # Conclusion
        conclusion = "As I prepare for college, I carry these lessons with me. This experience taught me that growth comes not from avoiding challenges, but from facing them with courage and learning from every step of the journey."
        essay_parts.append(conclusion)
        
        # Combine and adjust length
        essay = "\n\n".join(essay_parts)
        words = essay.split()
        
        if len(words) > word_count:
            essay = " ".join(words[:word_count])
        elif len(words) < word_count * 0.8:
            # Add more detail if too short
            essay += f"\n\nThis experience continues to influence how I approach new challenges and opportunities. It has given me the confidence to step outside my comfort zone and the resilience to persevere when things get difficult."
        
        return essay
    
    def extract_key_experiences(self, content):
        """Extract and format key experiences from conversation"""
        sentences = content.split('.')
        key_experiences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(word in sentence.lower() for word in [
                'challenge', 'learned', 'felt', 'realized', 'discovered', 
                'experience', 'moment', 'time', 'situation'
            ]):
                # Format as narrative
                if sentence and not sentence.endswith('.'):
                    sentence += '.'
                key_experiences.append(sentence)
        
        # Combine into coherent paragraphs
        if key_experiences:
            return [" ".join(key_experiences[:3]), " ".join(key_experiences[3:6]) if len(key_experiences) > 3 else ""]
        else:
            return ["The experience I want to share taught me valuable lessons about perseverance and growth."]
    
    def extract_themes_from_content(self, content):
        """Extract themes from conversation content"""
        content_lower = content.lower()
        themes = {
            'challenges': [],
            'lessons': [],
            'emotions': [],
            'growth': []
        }
        
        if any(word in content_lower for word in ['challenge', 'difficult', 'struggle']):
            themes['challenges'].append('Overcoming obstacles')
        if any(word in content_lower for word in ['learned', 'taught', 'realized']):
            themes['lessons'].append('Personal insights and learning')
        if any(word in content_lower for word in ['felt', 'emotion', 'scared', 'proud']):
            themes['emotions'].append('Emotional growth and self-awareness')
        if any(word in content_lower for word in ['changed', 'growth', 'better', 'stronger']):
            themes['growth'].append('Personal development and transformation')
        
        return themes
    
    def generate_title_from_content(self, content, essay):
        """Generate compelling title"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['challenge', 'difficult', 'overcome']):
            titles = ["Finding Strength in Struggle", "The Challenge That Changed Me", "Beyond the Difficulty"]
        elif any(word in content_lower for word in ['family', 'parent', 'mom', 'dad']):
            titles = ["Lessons from Home", "Family Values", "The People Who Shaped Me"]
        elif any(word in content_lower for word in ['sports', 'team', 'competition']):
            titles = ["More Than a Game", "Team Lessons", "Playing for Growth"]
        elif any(word in content_lower for word in ['school', 'academic', 'study']):
            titles = ["Learning Beyond the Classroom", "Academic Growth", "The Lesson That Mattered"]
        else:
            titles = ["My Journey of Growth", "The Experience That Shaped Me", "Finding My Voice"]
        
        return titles[0]
    
    def generate_ollama_essay(self, content, word_count):
        """Generate essay using Ollama"""
        try:
            prompt = f"""Write a compelling {word_count}-word college application essay based on this student's brainstorming responses:

{content}

Requirements:
- Personal statement format for college admissions
- Show character development and growth
- Use specific examples and details
- Demonstrate college readiness
- Authentic student voice
- Strong narrative structure

Write the complete essay:"""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": word_count * 2, "temperature": 0.7}
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
        except:
            pass
        
        return self.generate_enhanced_essay(content, word_count)

# ============================================================================
# Enhanced Text-to-Speech Service
# ============================================================================

class EnhancedTTSService:
    def __init__(self):
        self.pyttsx3_engine = None
        self.available_methods = []
        self.initialize_tts()
    
    def initialize_tts(self):
        """Initialize available TTS methods"""
        logger.info("Initializing enhanced TTS service...")
        
        # Try pyttsx3 (best for natural voices)
        if TTS_PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self._configure_pyttsx3()
                self.available_methods.append('pyttsx3')
                logger.info("✅ pyttsx3 TTS initialized")
            except Exception as e:
                logger.warning(f"pyttsx3 TTS failed: {e}")
        
        # Check Edge TTS availability
        if TTS_EDGE_AVAILABLE:
            self.available_methods.append('edge_tts')
            logger.info("✅ Edge TTS available")
        
        # Check Google TTS availability
        if TTS_GTTS_AVAILABLE:
            try:
                pygame.mixer.init()
                self.available_methods.append('gtts')
                logger.info("✅ Google TTS available")
            except Exception as e:
                logger.warning(f"Google TTS failed: {e}")
        
        # Fallback to browser TTS
        self.available_methods.append('browser')
        logger.info(f"TTS methods available: {self.available_methods}")
    
    def _configure_pyttsx3(self):
        """Configure pyttsx3 for natural speech"""
        if not self.pyttsx3_engine:
            return
        
        try:
            # Get available voices
            voices = self.pyttsx3_engine.getProperty('voices')
            
            # Find best voice (prefer female, English)
            best_voice = None
            for voice in voices:
                if voice.languages and 'en' in str(voice.languages).lower():
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower() or 'hazel' in voice.name.lower():
                        best_voice = voice
                        break
                    elif not best_voice:  # Fallback to any English voice
                        best_voice = voice
            
            if best_voice:
                self.pyttsx3_engine.setProperty('voice', best_voice.id)
                logger.info(f"Using voice: {best_voice.name}")
            
            # Configure speech properties for natural sound
            self.pyttsx3_engine.setProperty('rate', 175)  # Slightly slower for clarity
            self.pyttsx3_engine.setProperty('volume', 0.9)
            
        except Exception as e:
            logger.warning(f"Could not configure pyttsx3: {e}")
    
    async def generate_edge_tts_audio(self, text):
        """Generate audio using Edge TTS (most natural)"""
        try:
            # Use a natural female voice
            voice = "en-US-AriaNeural"  # Very natural sounding
            communicate = edge_tts.Communicate(text, voice)
            
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return None
    
    def generate_gtts_audio(self, text):
        """Generate audio using Google TTS"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.getvalue()
        except Exception as e:
            logger.error(f"Google TTS error: {e}")
            return None
    
    def speak_pyttsx3(self, text):
        """Speak using pyttsx3 (synchronous)"""
        if not self.pyttsx3_engine:
            return False
        
        try:
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"pyttsx3 speak error: {e}")
            return False
    
    def get_best_method(self):
        """Get the best available TTS method"""
        if 'edge_tts' in self.available_methods:
            return 'edge_tts'
        elif 'pyttsx3' in self.available_methods:
            return 'pyttsx3'
        elif 'gtts' in self.available_methods:
            return 'gtts'
        else:
            return 'browser'

# Initialize enhanced TTS service
enhanced_tts = EnhancedTTSService()

# ============================================================================
# Enhanced API Routes for Better TTS
# ============================================================================

@app.route('/api/enhanced-tts', methods=['POST'])
def enhanced_tts_endpoint():
    """Generate enhanced TTS audio"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        method = data.get('method', enhanced_tts.get_best_method())
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if method == 'edge_tts' and TTS_EDGE_AVAILABLE:
            # Generate Edge TTS audio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_data = loop.run_until_complete(enhanced_tts.generate_edge_tts_audio(text))
            loop.close()
            
            if audio_data:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                
                # Return file for download
                return send_file(temp_path, as_attachment=True, download_name='tts_audio.mp3', mimetype='audio/mpeg')
        
        elif method == 'gtts' and TTS_GTTS_AVAILABLE:
            # Generate Google TTS audio
            audio_data = enhanced_tts.generate_gtts_audio(text)
            
            if audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                
                return send_file(temp_path, as_attachment=True, download_name='tts_audio.mp3', mimetype='audio/mpeg')
        
        elif method == 'pyttsx3' and enhanced_tts.pyttsx3_engine:
            # Use pyttsx3 (client-side implementation needed)
            return jsonify({
                'success': True,
                'method': 'pyttsx3',
                'message': 'Use client-side pyttsx3 implementation'
            })
        
        else:
            # Fallback to browser TTS
            return jsonify({
                'success': True,
                'method': 'browser',
                'text': text,
                'message': 'Use browser speech synthesis'
            })
    
    except Exception as e:
        logger.error(f"Enhanced TTS error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts-info')
def tts_info():
    """Get available TTS methods and recommendations"""
    return jsonify({
        'available_methods': enhanced_tts.available_methods,
        'recommended_method': enhanced_tts.get_best_method(),
        'capabilities': {
            'edge_tts': TTS_EDGE_AVAILABLE,
            'pyttsx3': TTS_PYTTSX3_AVAILABLE,
            'gtts': TTS_GTTS_AVAILABLE
        }
    })

# Initialize components
logger.info("Initializing Voice Essay AI...")
voice_processor = EnhancedVoiceProcessor()
ai_chat = EnhancedAIChat()

# ============================================================================
# Enhanced Web Interface Routes
# ============================================================================

@app.route('/')
def index():
    """Serve enhanced main page"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HelloIvy - Essay Brainstormer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            color: #2c3e50;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 40px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo {
            width: 32px;
            height: 32px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }

        .brand-name {
            font-size: 20px;
            font-weight: 600;
            color: #2c3e50;
        }

        .brand-tagline {
            color: #7f8c8d;
            margin-left: 8px;
        }

        .user-section {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .help-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border: 2px solid #e1e8ed;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .help-btn:hover {
            border-color: #667eea;
            color: #667eea;
        }

        .user-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(45deg, #ff6b6b, #ffa500);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }

        .sidebar {
            position: fixed;
            left: 40px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            gap: 20px;
            z-index: 100;
        }

        .sidebar-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 16px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 80px;
        }

        .sidebar-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        }

        .sidebar-item.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }

        .sidebar-icon {
            width: 24px;
            height: 24px;
            margin-bottom: 8px;
            font-size: 18px;
        }

        .sidebar-text {
            font-size: 11px;
            font-weight: 500;
            text-align: center;
            line-height: 1.2;
        }

        .main-content {
            margin-left: 180px;
            padding: 60px 40px;
            max-width: 800px;
        }

        .page-title {
            color: #667eea;
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .main-heading {
            font-size: 36px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 40px;
            line-height: 1.2;
        }

        .instruction-card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }

        .card-title {
            font-size: 18px;
            font-weight: 600;
            color: #2c3e50;
        }

        .listen-btn {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: linear-gradient(135deg, #ff6b6b, #ffa500);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .listen-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        }

        .instructions-list {
            list-style: none;
            margin-bottom: 24px;
        }

        .instructions-list li {
            margin-bottom: 16px;
            padding-left: 24px;
            position: relative;
            line-height: 1.6;
            color: #5a6c7d;
        }

        .instructions-list li::before {
            content: counter(item);
            counter-increment: item;
            position: absolute;
            left: 0;
            top: 0;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: bold;
        }

        .instructions-list {
            counter-reset: item;
        }

        .checkbox-container {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
            padding: 16px;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .custom-checkbox {
            position: relative;
            width: 20px;
            height: 20px;
        }

        .custom-checkbox input {
            position: absolute;
            opacity: 0;
            cursor: pointer;
        }

        .checkmark {
            position: absolute;
            top: 0;
            left: 0;
            height: 20px;
            width: 20px;
            background-color: white;
            border: 2px solid #667eea;
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .custom-checkbox input:checked ~ .checkmark {
            background-color: #667eea;
        }

        .checkmark:after {
            content: "";
            position: absolute;
            display: none;
            left: 6px;
            top: 3px;
            width: 4px;
            height: 8px;
            border: solid white;
            border-width: 0 2px 2px 0;
            transform: rotate(45deg);
        }

        .custom-checkbox input:checked ~ .checkmark:after {
            display: block;
        }

        .checkbox-label {
            font-size: 14px;
            color: #5a6c7d;
            cursor: pointer;
        }

        .get-started-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 16px 32px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .get-started-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }

        .get-started-btn:disabled {
            background: #cbd5e0;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .illustration {
            position: absolute;
            right: 40px;
            top: 50%;
            transform: translateY(-50%);
            width: 300px;
            height: 200px;
            opacity: 0.1;
            z-index: 0;
        }

        .puzzle-piece {
            position: absolute;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 8px;
            opacity: 0.6;
        }

        .piece-1 {
            width: 80px;
            height: 60px;
            top: 20px;
            right: 100px;
            transform: rotate(-10deg);
        }

        .piece-2 {
            width: 60px;
            height: 80px;
            top: 100px;
            right: 50px;
            transform: rotate(15deg);
        }

        .piece-3 {
            width: 70px;
            height: 50px;
            top: 150px;
            right: 150px;
            transform: rotate(-5deg);
        }

        .piece-4 {
            width: 50px;
            height: 70px;
            top: 60px;
            right: 200px;
            transform: rotate(20deg);
        }

        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                top: auto;
                transform: none;
                flex-direction: row;
                justify-content: center;
                background: white;
                padding: 10px;
                box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.1);
            }

            .main-content {
                margin-left: 0;
                margin-bottom: 80px;
                padding: 40px 20px;
            }

            .illustration {
                display: none;
            }

            .header {
                padding: 15px 20px;
            }

            .main-heading {
                font-size: 28px;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo-section">
            <div class="logo">H</div>
            <span class="brand-name">helloivy</span>
            <span class="brand-tagline">Essay Brainstormer</span>
        </div>
        <div class="user-section">
            <div class="help-btn">?</div>
            <div class="user-avatar">A</div>
        </div>
    </header>

    <nav class="sidebar">
        <div class="sidebar-item">
            <div class="sidebar-icon">📊</div>
            <div class="sidebar-text">My Dashboard</div>
        </div>
        <div class="sidebar-item active">
            <div class="sidebar-icon">🏠</div>
            <div class="sidebar-text">Essay Brainstormer</div>
        </div>
    </nav>

    <main class="main-content">
        <div class="page-title">Essay Brainstormer</div>
        <h1 class="main-heading">Ready to start<br>Brainstorming?</h1>

        <div class="instruction-card">
            <div class="card-header">
                <h2 class="card-title">Go Through Instructions Before We Start The Module</h2>
                <button class="listen-btn">
                    🔊 Listen
                </button>
            </div>

            <ol class="instructions-list">
                <li>This module helps you lay the foundation for powerful, personalized college essays.</li>
                <li>Ivy will collect key stories, goals, and reflections – which are then used to build a custom essay structure aligned with your chosen prompts and colleges.</li>
                <li>Find a quiet, distraction-free environment, since Ivy is a voice module and will be talking to you.</li>
                <li>Ensure a stable internet connection – this will support smooth voice-to-text capture and prevent loss of any responses.</li>
                <li>Be honest, detailed and specific in your responses.</li>
                <li>The quality of your essay structure will depend on what you share here.</li>
            </ol>

            <div class="checkbox-container">
                <label class="custom-checkbox">
                    <input type="checkbox" id="readInstructions">
                    <span class="checkmark"></span>
                </label>
                <label for="readInstructions" class="checkbox-label">
                    I have read all the instruction mentioned above.
                </label>
            </div>

            <button class="get-started-btn" id="getStartedBtn" disabled>
                Get Started →
            </button>
        </div>
    </main>

    <div class="illustration">
        <div class="puzzle-piece piece-1"></div>
        <div class="puzzle-piece piece-2"></div>
        <div class="puzzle-piece piece-3"></div>
        <div class="puzzle-piece piece-4"></div>
    </div>

    <script>
        // Handle checkbox and button state
        const checkbox = document.getElementById('readInstructions');
        const getStartedBtn = document.getElementById('getStartedBtn');

        checkbox.addEventListener('change', function() {
            if (this.checked) {
                getStartedBtn.disabled = false;
            } else {
                getStartedBtn.disabled = true;
            }
        });

        // Handle Get Started button click
        getStartedBtn.addEventListener('click', function() {
            if (!getStartedBtn.disabled) {
                // Redirect to voice chat or show loading
                console.log('Starting brainstorming session...');
                // You can redirect to your voice chat page here
                window.location.href = '/voice-chat';
            }
        });

        // Handle Listen button (Text-to-Speech)
        document.querySelector('.listen-btn').addEventListener('click', function() {
            const instructions = [
                "This module helps you lay the foundation for powerful, personalized college essays.",
                "Ivy will collect key stories, goals, and reflections – which are then used to build a custom essay structure aligned with your chosen prompts and colleges.",
                "Find a quiet, distraction-free environment, since Ivy is a voice module and will be talking to you.",
                "Ensure a stable internet connection – this will support smooth voice-to-text capture and prevent loss of any responses.",
                "Be honest, detailed and specific in your responses.",
                "The quality of your essay structure will depend on what you share here."
            ].join(' ');

            if ('speechSynthesis' in window) {
                // Stop any ongoing speech
                speechSynthesis.cancel();
                
                const utterance = new SpeechSynthesisUtterance(instructions);
                utterance.rate = 0.9;
                utterance.pitch = 1.0;
                utterance.volume = 0.8;
                
                // Try to find a good voice
                const voices = speechSynthesis.getVoices();
                const preferredVoice = voices.find(voice => 
                    voice.name.includes('Google') || 
                    voice.name.includes('Microsoft') ||
                    voice.lang.startsWith('en')
                );
                
                if (preferredVoice) {
                    utterance.voice = preferredVoice;
                }

                utterance.onstart = function() {
                    document.querySelector('.listen-btn').textContent = '⏸️ Stop';
                };

                utterance.onend = function() {
                    document.querySelector('.listen-btn').innerHTML = '🔊 Listen';
                };

                speechSynthesis.speak(utterance);
            } else {
                alert('Text-to-speech is not supported in your browser.');
            }
        });

        // Handle sidebar navigation
        document.querySelectorAll('.sidebar-item').forEach(item => {
            item.addEventListener('click', function() {
                // Remove active class from all items
                document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
                // Add active class to clicked item
                this.classList.add('active');
                
                // Handle navigation based on the clicked item
                const text = this.querySelector('.sidebar-text').textContent;
                if (text === 'My Dashboard') {
                    // Navigate to dashboard
                    console.log('Navigating to dashboard...');
                    // window.location.href = 'static/dashboard';
                }
            });
        });

        // Animate puzzle pieces
        function animatePuzzlePieces() {
            const pieces = document.querySelectorAll('.puzzle-piece');
            pieces.forEach((piece, index) => {
                piece.style.animation = `float ${3 + index * 0.5}s ease-in-out infinite`;
            });
        }

        // CSS for floating animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(var(--rotation, 0deg)); }
                50% { transform: translateY(-10px) rotate(var(--rotation, 0deg)); }
            }
        `;
        document.head.appendChild(style);

        // Set rotation variables for each piece
        document.querySelector('.piece-1').style.setProperty('--rotation', '-10deg');
        document.querySelector('.piece-2').style.setProperty('--rotation', '15deg');
        document.querySelector('.piece-3').style.setProperty('--rotation', '-5deg');
        document.querySelector('.piece-4').style.setProperty('--rotation', '20deg');

        // Start animations when page loads
        window.addEventListener('load', animatePuzzlePieces);
    </script>
</body>
</html>'''

@app.route('/voice-chat')
def voice_chat():
    """Serve enhanced voice chat interface"""
    return send_from_directory('static', 'brainstorm.html')

@app.route('/dashboard')
def dashboard():
    """Serve enhanced dashboard"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📊 Dashboard - Voice Essay AI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1a202c; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; color: #2d3748; }
        .header p { color: #718096; font-size: 1.1rem; margin-bottom: 20px; }
        .header-buttons { display: flex; gap: 15px; flex-wrap: wrap; }
        .btn { display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; text-decoration: none; border-radius: 10px; font-weight: 600; transition: all 0.3s ease; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); }
        .btn.secondary { background: linear-gradient(135deg, #48bb78, #38a169); }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        .stat-number { font-size: 2.5rem; font-weight: bold; color: #667eea; margin-bottom: 5px; }
        .stat-label { color: #718096; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px; }
        .section { background: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        .section h2 { color: #2d3748; margin-bottom: 20px; }
        .conversation-item, .essay-item { border: 1px solid #e2e8f0; padding: 20px; margin: 15px 0; border-radius: 12px; transition: all 0.3s ease; }
        .conversation-item:hover, .essay-item:hover { box-shadow: 0 4px 15px rgba(0,0,0,0.1); transform: translateY(-1px); }
        .conversation-item { background: #f7fafc; }
        .essay-item { background: #ebf8ff; }
        .conversation-meta, .essay-meta { color: #718096; font-size: 0.9rem; margin-bottom: 10px; }
        .conversation-preview, .essay-preview { color: #4a5568; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Dashboard</h1>
            <p>Manage your voice brainstorming sessions and generated essays</p>
            <div class="header-buttons">
                <a href="/voice-chat" class="btn">🎤 Start New Session</a>
                <a href="/" class="btn secondary">🏠 Home</a>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number" id="totalConversations">0</div>
                <div class="stat-label">Conversations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="totalEssays">0</div>
                <div class="stat-label">Essays Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="totalWords">0</div>
                <div class="stat-label">Words Written</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="avgWordCount">0</div>
                <div class="stat-label">Avg Essay Length</div>
            </div>
        </div>
        
        <div class="section">
            <h2>💬 Recent Conversations</h2>
            <div id="conversations">
                <p style="text-align: center; color: #718096; padding: 40px;">No conversations yet. <a href="/voice-chat" style="color: #667eea;">Start your first session!</a></p>
            </div>
        </div>
        
        <div class="section">
            <h2>📝 Generated Essays</h2>
            <div id="essays">
                <p style="text-align: center; color: #718096; padding: 40px;">No essays yet. Complete a conversation to generate your first essay!</p>
            </div>
        </div>
    </div>

    <script>
        async function loadDashboard() {
            try {
                const [convResponse, essayResponse] = await Promise.all([
                    fetch('/api/conversations'),
                    fetch('/api/essays')
                ]);
                
                const conversations = await convResponse.json();
                const essays = await essayResponse.json();
                
                // Update stats
                document.getElementById('totalConversations').textContent = conversations.length;
                document.getElementById('totalEssays').textContent = essays.length;
                
                const totalWords = essays.reduce((sum, e) => sum + (e.word_count || 0), 0);
                document.getElementById('totalWords').textContent = totalWords;
                document.getElementById('avgWordCount').textContent = essays.length > 0 ? Math.round(totalWords / essays.length) : 0;
                
                // Display conversations
                const convDiv = document.getElementById('conversations');
                if (conversations.length > 0) {
                    convDiv.innerHTML = conversations.map(conv => `
                        <div class="conversation-item">
                            <div class="conversation-meta">
                                <strong>${conv.student_name || 'Anonymous Student'}</strong> • 
                                ${conv.message_count} messages • 
                                ${new Date(conv.created_at).toLocaleDateString()}
                            </div>
                            <div class="conversation-preview">
                                Stage: ${conv.stage || 'In Progress'} | 
                                Ready for essay: ${conv.message_count >= 8 ? '✅' : '⏳'}
                            </div>
                        </div>
                    `).join('');
                }
                
                // Display essays
                const essayDiv = document.getElementById('essays');
                if (essays.length > 0) {
                    essayDiv.innerHTML = essays.map(essay => `
                        <div class="essay-item">
                            <div class="essay-meta">
                                <strong>${essay.title}</strong> • 
                                ${essay.word_count} words • 
                                ${new Date(essay.created_at).toLocaleDateString()}
                            </div>
                            <div class="essay-preview">
                                ${essay.essay.substring(0, 250)}...
                            </div>
                        </div>
                    `).join('');
                }
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
            }
        }
        
        loadDashboard();
    </script>
</body>
</html>'''

# ============================================================================
# Enhanced API Routes
# ============================================================================

@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'voice_available': voice_processor.is_available(),
        'ai_available': ai_chat.is_available(),
        'version': '2.0.0',
        'features': {
            'speech_recognition': voice_processor.is_available(),
            'ai_chat': True,
            'essay_generation': True,
            'conversation_storage': True
        }
    })

@app.route('/api/voice-transcribe', methods=['POST'])
def voice_transcribe():
    """Enhanced voice transcription with better error handling"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if not voice_processor.is_available():
            return jsonify({'error': 'Voice processing not available'}), 503
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Transcribe with enhanced processor
            result = voice_processor.transcribe(temp_path)
            
            # Log successful transcription
            logger.info(f"Transcription successful: {result['text'][:50]}...")
            
            return jsonify({
                'success': True,
                'transcription': result['text'],
                'confidence': result['confidence'],
                'processing_time': result['processing_time'],
                'language': result.get('language', 'en')
            })
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice-respond', methods=['POST'])
def voice_respond():
    """Enhanced AI response with conversation flow"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        conversation_id = data.get('conversation_id', f'conv_{int(time.time())}')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get enhanced AI response
        ai_response = ai_chat.get_response(user_message, conversation_id)
        
        # Save conversation turn
        save_conversation_turn(conversation_id, user_message, ai_response['message'])
        
        logger.info(f"AI Response: {ai_response['message'][:50]}...")
        
        return jsonify({
            'success': True,
            'ai_response': ai_response['message'],
            'should_continue': ai_response['should_continue'],
            'conversation_stage': ai_response['stage'],
            'student_name': ai_response['student_name'],
            'conversation_id': conversation_id,
            'message_count': ai_response['message_count']
        })
        
    except Exception as e:
        logger.error(f"Voice response error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice-conversation', methods=['POST'])
def voice_conversation():
    """Complete voice conversation flow - transcribe and respond"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        conversation_id = request.form.get('conversation_id', f'conv_{int(time.time())}')
        
        # Save and transcribe audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Transcribe
            transcription_result = voice_processor.transcribe(temp_path)
            user_message = transcription_result['text']
            
            if not user_message.strip():
                return jsonify({'error': 'No speech detected'}), 400
            
            # Get AI response
            ai_response = ai_chat.get_response(user_message, conversation_id)
            
            # Save conversation
            save_conversation_turn(conversation_id, user_message, ai_response['message'])
            
            return jsonify({
                'success': True,
                'user_message': user_message,
                'transcription_confidence': transcription_result['confidence'],
                'ai_response': ai_response['message'],
                'should_continue': ai_response['should_continue'],
                'conversation_stage': ai_response['stage'],
                'conversation_id': conversation_id
            })
            
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Voice conversation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-essay', methods=['POST'])
def generate_essay():
    """Enhanced essay generation"""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id', 'default')
        word_count = data.get('word_count', 500)
        
        # Load conversation
        conversation = load_conversation(conversation_id)
        
        if not conversation or len(conversation['messages']) < 8:
            return jsonify({'error': 'Need more conversation content for essay generation'}), 400
        
        # Generate enhanced essay
        essay_result = ai_chat.generate_essay(conversation['messages'], word_count)
        
        # Save essay
        essay_id = save_essay(conversation_id, essay_result)
        
        logger.info(f"Essay generated: {essay_result['title']} ({essay_result['word_count']} words)")
        
        return jsonify({
            'success': True,
            'essay_id': essay_id,
            'title': essay_result['title'],
            'essay': essay_result['essay'],
            'word_count': essay_result['word_count'],
            'notes': essay_result['notes']
        })
        
    except Exception as e:
        logger.error(f"Essay generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations')
def get_conversations():
    """Get enhanced conversation list"""
    try:
        conversations_file = 'data/conversations.json'
        if not os.path.exists(conversations_file):
            return jsonify([])
        
        with open(conversations_file, 'r') as f:
            all_conversations = json.load(f)
        
        summaries = []
        for conv_id, conv_data in all_conversations.items():
            user_messages = [m for m in conv_data.get('messages', []) if m['role'] == 'user']
            summaries.append({
                'id': conv_id,
                'created_at': conv_data.get('created_at'),
                'message_count': len(conv_data.get('messages', [])),
                'user_message_count': len(user_messages),
                'student_name': conv_data.get('student_name', 'Unknown'),
                'stage': conv_data.get('stage', 'unknown'),
                'last_updated': conv_data.get('last_updated'),
                'ready_for_essay': len(user_messages) >= 4
            })
        
        # Sort by last updated
        summaries.sort(key=lambda x: x.get('last_updated', x.get('created_at', '')), reverse=True)
        
        return jsonify(summaries)
        
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return jsonify([])

@app.route('/api/essays')
def get_essays():
    """Get enhanced essay list"""
    try:
        essays_file = 'data/essays.json'
        if not os.path.exists(essays_file):
            return jsonify([])
        
        with open(essays_file, 'r') as f:
            essays = json.load(f)
        
        essay_list = list(essays.values())
        # Sort by created date
        essay_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify(essay_list)
        
    except Exception as e:
        logger.error(f"Get essays error: {e}")
        return jsonify([])

# ============================================================================
# Enhanced Helper Functions
# ============================================================================

def save_conversation_turn(conversation_id, user_message, ai_message):
    """Enhanced conversation saving with better structure"""
    try:
        conversations_file = 'data/conversations.json'
        
        if os.path.exists(conversations_file):
            with open(conversations_file, 'r') as f:
                conversations = json.load(f)
        else:
            conversations = {}
        
        if conversation_id not in conversations:
            conversations[conversation_id] = {
                'created_at': datetime.now().isoformat(),
                'messages': [],
                'student_name': None,
                'stage': 'greeting'
            }
        
        # Add messages with enhanced metadata
        conversations[conversation_id]['messages'].extend([
            {
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat(),
                'word_count': len(user_message.split())
            },
            {
                'role': 'assistant', 
                'content': ai_message,
                'timestamp': datetime.now().isoformat(),
                'word_count': len(ai_message.split())
            }
        ])
        
        # Update student name if found
        if not conversations[conversation_id]['student_name']:
            name = extract_name_from_message(user_message)
            if name:
                conversations[conversation_id]['student_name'] = name
        
        # Update conversation metadata
        conversations[conversation_id]['last_updated'] = datetime.now().isoformat()
        conversations[conversation_id]['message_count'] = len(conversations[conversation_id]['messages'])
        
        # Determine stage
        user_count = len([m for m in conversations[conversation_id]['messages'] if m['role'] == 'user'])
        if user_count <= 2:
            stage = 'exploration'
        elif user_count <= 4:
            stage = 'deep_dive'
        elif user_count <= 6:
            stage = 'synthesis'
        else:
            stage = 'conclusion'
        conversations[conversation_id]['stage'] = stage
        
        # Save to file
        with open(conversations_file, 'w') as f:
            json.dump(conversations, f, indent=2)
        
        logger.info(f"Conversation saved: {conversation_id} ({user_count} user messages)")
        
    except Exception as e:
        logger.error(f"Error saving conversation: {e}")

def load_conversation(conversation_id):
    """Enhanced conversation loading"""
    try:
        conversations_file = 'data/conversations.json'
        if not os.path.exists(conversations_file):
            return None
        
        with open(conversations_file, 'r') as f:
            conversations = json.load(f)
        
        return conversations.get(conversation_id)
        
    except Exception as e:
        logger.error(f"Error loading conversation: {e}")
        return None

def save_essay(conversation_id, essay_result):
    """Enhanced essay saving with metadata"""
    try:
        essays_file = 'data/essays.json'
        
        if os.path.exists(essays_file):
            with open(essays_file, 'r') as f:
                essays = json.load(f)
        else:
            essays = {}
        
        essay_id = f"essay_{len(essays) + 1}_{int(datetime.now().timestamp())}"
        
        essays[essay_id] = {
            'id': essay_id,
            'conversation_id': conversation_id,
            'title': essay_result['title'],
            'essay': essay_result['essay'],
            'word_count': essay_result['word_count'],
            'notes': essay_result['notes'],
            'created_at': datetime.now().isoformat(),
            'generated_at': essay_result.get('generated_at'),
            'version': '2.0'
        }
        
        with open(essays_file, 'w') as f:
            json.dump(essays, f, indent=2)
        
        logger.info(f"Essay saved: {essay_id}")
        return essay_id
        
    except Exception as e:
        logger.error(f"Error saving essay: {e}")
        return None

def extract_name_from_message(message):
    """Enhanced name extraction"""
    patterns = [
        r"my name is (\w+)",
        r"i'm (\w+)",
        r"i am (\w+)",
        r"call me (\w+)",
        r"name's (\w+)",
        r"this is (\w+)",
        r"hello,? (?:i'm |my name is )?(\w+)",
        r"hi,? (?:i'm |my name is )?(\w+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message.lower())
        if match:
            name = match.group(1).capitalize()
            # Filter out common false positives
            if name.lower() not in ['good', 'fine', 'well', 'okay', 'sure', 'yes', 'no']:
                return name
    
    return None

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# Static Files
# ============================================================================

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("\n🎤 Voice Essay AI - Enhanced Version 2.0")
    print("=" * 60)
    print(f"✅ Flask Server: Ready")
    print(f"🎤 Voice Processing: {'Available' if voice_processor.is_available() else 'Limited (install: pip install openai-whisper torch)'}")
    print(f"🤖 AI Chat: {'Ollama Connected' if ai_chat.is_available() else 'Enhanced Mode (install Ollama for advanced AI)'}")
    print(f"💾 Data Storage: Local JSON files")
    print("")
    print("🌐 Access your application:")
    print("   🏠 Home: http://localhost:5000")
    print("   🎤 Voice Chat: http://localhost:5000/voice-chat")
    print("   📊 Dashboard: http://localhost:5000/dashboard")
    print("")
    
    if not ai_chat.is_available():
        print("💡 For enhanced AI responses:")
        print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Start Ollama: ollama serve")
        print("   3. Download model: ollama pull mistral:7b")
        print("   4. Restart this application")
        print("")
    
    if not voice_processor.is_available():
        print("🎤 For voice processing:")
        print("   pip install openai-whisper torch")
        print("")
    
    print("📋 Quick Dependencies Install:")
    print("   pip install flask flask-cors openai-whisper torch requests")
    print("")
    print("🎯 Features:")
    print("   • Natural voice conversation with AI")
    print("   • Automatic speech-to-text transcription")
    print("   • AI text-to-speech responses")
    print("   • Smart note-taking and theme extraction")
    print("   • Professional essay generation")
    print("   • Conversation and essay management")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )