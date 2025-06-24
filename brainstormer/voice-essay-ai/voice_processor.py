# voice_processor.py - Voice Processing with Whisper and TTS
import os
import time
import subprocess
import platform
import logging
import tempfile
from typing import Dict, Any, Optional

try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoiceProcessor:
    """Simple voice processing with Whisper and system TTS"""
    
    def __init__(self, model_name="base"):
        self.model_name = model_name
        self.whisper_model = None
        self.tts_engine = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.initialize()
    
    def initialize(self):
        """Initialize voice processing components"""
        logger.info("Initializing voice processor...")
        
        # Initialize Whisper
        if WHISPER_AVAILABLE:
            try:
                logger.info(f"Loading Whisper model: {self.model_name}")
                self.whisper_model = whisper.load_model(self.model_name, device=self.device)
                logger.info("✅ Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                self.whisper_model = None
        else:
            logger.warning("Whisper not available - install with: pip install openai-whisper")
        
        # Initialize TTS
        self._setup_tts()
    
    def _setup_tts(self):
        """Setup text-to-speech based on platform"""
        system = platform.system().lower()
        
        if system == "linux":
            if self._command_exists("espeak"):
                self.tts_engine = "espeak"
                logger.info("✅ TTS: Using espeak")
            elif self._command_exists("spd-say"):
                self.tts_engine = "spd-say"
                logger.info("✅ TTS: Using speech-dispatcher")
            else:
                logger.warning("⚠️ TTS: No engine found. Install espeak: sudo apt install espeak")
        
        elif system == "darwin":  # macOS
            self.tts_engine = "say"
            logger.info("✅ TTS: Using macOS say")
        
        elif system == "windows":
            self.tts_engine = "sapi"
            logger.info("✅ TTS: Using Windows SAPI")
        
        else:
            logger.warning(f"⚠️ TTS: Unsupported system: {system}")
    
    def _command_exists(self, command):
        """Check if command exists in PATH"""
        try:
            result = subprocess.run([command, "--version"], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def is_available(self):
        """Check if voice processing is available"""
        return {
            'whisper': self.whisper_model is not None,
            'tts': self.tts_engine is not None,
            'device': self.device
        }
    
    def transcribe(self, audio_file_path: str) -> Dict[str, Any]:
        """Transcribe audio file to text using Whisper"""
        if not self.whisper_model:
            raise Exception("Whisper model not available")
        
        try:
            logger.info(f"Transcribing: {audio_file_path}")
            start_time = time.time()
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                audio_file_path,
                task="transcribe",
                language="en",
                fp16=False,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            text = result["text"].strip()
            confidence = self._calculate_confidence(result)
            
            logger.info(f"✅ Transcribed in {processing_time:.2f}s: {text[:50]}...")
            
            return {
                'text': text,
                'confidence': confidence,
                'processing_time': processing_time,
                'language': result.get('language', 'en')
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _calculate_confidence(self, whisper_result):
        """Calculate confidence score from Whisper result"""
        if "segments" in whisper_result and whisper_result["segments"]:
            confidences = []
            for segment in whisper_result["segments"]:
                if "avg_logprob" in segment:
                    # Convert log probability to confidence (0-1)
                    conf = min(1.0, max(0.0, segment["avg_logprob"] + 1.0))
                    confidences.append(conf)
            
            if confidences:
                return sum(confidences) / len(confidences)
        
        return 0.8  # Default confidence
    
    def speak(self, text: str, rate: int = 150) -> bool:
        """Convert text to speech"""
        if not self.tts_engine:
            logger.warning("TTS not available")
            return False
        
        try:
            clean_text = self._clean_text_for_speech(text)
            logger.info(f"🔊 Speaking: {clean_text[:30]}...")
            
            if self.tts_engine == "espeak":
                # Linux espeak
                subprocess.run([
                    "espeak", 
                    "-s", str(rate),      # Speed
                    "-v", "en+m3",        # Male voice
                    "-a", "200",          # Amplitude
                    clean_text
                ], timeout=30, check=True)
            
            elif self.tts_engine == "spd-say":
                # Speech dispatcher
                subprocess.run([
                    "spd-say",
                    "-r", str(int(rate/10)),  # Rate
                    "-v", "MALE1",            # Voice
                    clean_text
                ], timeout=30, check=True)
            
            elif self.tts_engine == "say":
                # macOS
                subprocess.run([
                    "say", 
                    "-r", str(rate),
                    "-v", "Alex",
                    clean_text
                ], timeout=30, check=True)
            
            elif self.tts_engine == "sapi":
                # Windows PowerShell
                ps_command = f'''
                Add-Type -AssemblyName System.Speech
                $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $speak.Rate = {int(rate/50)}
                $speak.Speak("{clean_text}")
                '''
                subprocess.run([
                    "powershell", "-Command", ps_command
                ], timeout=30, check=True)
            
            logger.info("✅ Speech completed")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("TTS timeout")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"TTS command failed: {e}")
            return False
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech output"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'#{1,6}\s*', '', text)         # Headers
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Replace symbols with words
        replacements = {
            '&': ' and ',
            '@': ' at ',
            '#': ' number ',
            '%': ' percent ',
            '$': ' dollars ',
            '+': ' plus ',
            '=': ' equals ',
            '/': ' or ',
            ' vs ': ' versus ',
            ' etc.': ' and so on',
            ' i.e.': ' that is',
            ' e.g.': ' for example'
        }
        
        for symbol, word in replacements.items():
            text = text.replace(symbol, word)
        
        # Clean up spacing and punctuation
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = text.strip()
        
        # Ensure text ends with punctuation for better speech rhythm
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def test_voice_system(self):
        """Test both speech recognition and TTS"""
        logger.info("Testing voice system...")
        
        results = {
            'whisper_available': self.whisper_model is not None,
            'tts_available': self.tts_engine is not None,
            'device': self.device,
            'tests': {}
        }
        
        # Test TTS
        if self.tts_engine:
            try:
                success = self.speak("Voice system test. Can you hear me?")
                results['tests']['tts'] = 'success' if success else 'failed'
            except Exception as e:
                results['tests']['tts'] = f'error: {e}'
        else:
            results['tests']['tts'] = 'not_available'
        
        # Test Whisper (would need actual audio file)
        if self.whisper_model:
            results['tests']['whisper'] = 'available'
        else:
            results['tests']['whisper'] = 'not_available'
        
        return results
    
    def create_test_audio(self, text="Hello, this is a test.", filename="test_audio.wav"):
        """Create test audio file using TTS (for testing purposes)"""
        if not self.tts_engine:
            return None
        
        try:
            # Create audio file using TTS
            if self.tts_engine == "espeak":
                subprocess.run([
                    "espeak", 
                    "-s", "150",
                    "-v", "en+m3",
                    "-w", filename,  # Write to file
                    text
                ], timeout=30, check=True)
                
                if os.path.exists(filename):
                    logger.info(f"✅ Test audio created: {filename}")
                    return filename
            
        except Exception as e:
            logger.error(f"Failed to create test audio: {e}")
        
        return None

# Test function
def test_voice_processor():
    """Test the voice processor"""
    print("🎤 Testing Voice Processor")
    print("=" * 30)
    
    processor = VoiceProcessor()
    
    # Check availability
    availability = processor.is_available()
    print(f"Whisper available: {availability['whisper']}")
    print(f"TTS available: {availability['tts']}")
    print(f"Device: {availability['device']}")
    
    # Run tests
    test_results = processor.test_voice_system()
    print("\nTest Results:")
    for test, result in test_results['tests'].items():
        print(f"  {test}: {result}")
    
    return test_results

if __name__ == "__main__":
    test_voice_processor()