# ai_chat.py - AI Conversation Manager with Short Responses
import requests
import logging
import json
import re
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AIChat:
    """Simple AI conversation manager with short responses"""
    
    def __init__(self, ollama_url="http://localhost:11434", model="mistral:7b"):
        self.ollama_url = ollama_url
        self.model = model
        self.conversations = {}  # Store conversation state
        
        # Check if Ollama is available
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m["name"] for m in models]
                logger.info(f"✅ Ollama connected. Models: {available_models}")
                
                if self.model not in available_models:
                    logger.warning(f"⚠️ Model {self.model} not found. Available: {available_models}")
                    # Try to use first available model
                    if available_models:
                        self.model = available_models[0]
                        logger.info(f"🔄 Using model: {self.model}")
            else:
                logger.error("❌ Ollama not responding properly")
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.error("Please start Ollama: ollama serve")
    
    def is_available(self):
        """Check if AI is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def get_response(self, user_message: str, conversation_id: str = "default") -> Dict[str, Any]:
        """Get AI response for user message"""
        
        # Initialize conversation if new
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                'messages': [],
                'stage': 'greeting',
                'student_name': None,
                'created_at': datetime.now().isoformat()
            }
        
        conv = self.conversations[conversation_id]
        
        # Add user message
        conv['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Extract student name if not already known
        if not conv['student_name']:
            name = self._extract_name(user_message)
            if name:
                conv['student_name'] = name
        
        # Determine conversation stage
        stage = self._determine_stage(conv['messages'], conv['student_name'])
        conv['stage'] = stage
        
        # Generate short response
        ai_message = self._generate_short_response(
            user_message, conv['messages'], stage, conv['student_name']
        )
        
        # Add AI response
        conv['messages'].append({
            'role': 'assistant',
            'content': ai_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check if conversation should continue
        should_continue = len(conv['messages']) < 16 and stage != 'conclusion'
        
        return {
            'message': ai_message,
            'should_continue': should_continue,
            'stage': stage,
            'student_name': conv['student_name'],
            'message_count': len(conv['messages'])
        }
    
    def _extract_name(self, message: str) -> str:
        """Extract student name from message"""
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
    
    def _determine_stage(self, messages: List[Dict], student_name: str) -> str:
        """Determine conversation stage"""
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        count = len(user_messages)
        
        if count == 0:
            return 'greeting'
        elif count == 1 and not student_name:
            return 'greeting'  # Still asking for name
        elif count <= 2:
            return 'exploration'
        elif count <= 4:
            return 'deep_dive'
        elif count <= 6:
            return 'synthesis'
        else:
            return 'conclusion'
    
    def _generate_short_response(self, user_message: str, messages: List[Dict], 
                                stage: str, student_name: str) -> str:
        """Generate short, contextual response"""
        
        name_part = f" {student_name}" if student_name else ""
        user_msg_lower = user_message.lower()
        
        # Template-based short responses
        if stage == 'greeting':
            if not student_name:
                return "Hi! I'm your AI essay coach. What's your name?"
            else:
                return f"Great to meet you{name_part}! Tell me about a meaningful challenge or achievement you've experienced."
        
        elif stage == 'exploration':
            # Reference what user mentioned
            if any(word in user_msg_lower for word in ['challenge', 'difficult', 'hard', 'struggle']):
                return f"That sounds tough{name_part}. How did you feel when it happened?"
            elif any(word in user_msg_lower for word in ['success', 'achievement', 'won', 'accomplished']):
                return f"That's impressive{name_part}! What made this so meaningful to you?"
            elif any(word in user_msg_lower for word in ['family', 'parent', 'mom', 'dad']):
                return f"Family experiences can be powerful{name_part}. How did this affect you?"
            elif any(word in user_msg_lower for word in ['school', 'teacher', 'class']):
                return f"School experiences shape us{name_part}. What did you learn from this?"
            else:
                return f"That's interesting{name_part}. What made this experience significant for you?"
        
        elif stage == 'deep_dive':
            # Emotional follow-up
            emotions = ['scared', 'nervous', 'excited', 'proud', 'worried', 'happy', 'sad', 'angry']
            if any(emotion in user_msg_lower for emotion in emotions):
                return f"I can understand that feeling{name_part}. What did you learn about yourself?"
            elif any(word in user_msg_lower for word in ['learned', 'realized', 'discovered']):
                return f"That's a valuable insight{name_part}. How has this changed you?"
            else:
                return f"That's really thoughtful{name_part}. How did this experience change your perspective?"
        
        elif stage == 'synthesis':
            # Connect to future
            if any(word in user_msg_lower for word in ['college', 'university', 'study']):
                return f"Great connection{name_part}! How will this experience help you in college?"
            elif any(word in user_msg_lower for word in ['goal', 'dream', 'future']):
                return f"Those are meaningful goals{name_part}. How does your experience prepare you?"
            else:
                return f"That's real growth{name_part}. What are your hopes for college?"
        
        else:  # conclusion
            return f"Amazing story{name_part}! You've shared incredible insights. Ready to create your essay?"
    
    def generate_essay(self, messages: List[Dict], word_count: int = 500) -> Dict[str, Any]:
        """Generate essay from conversation messages"""
        
        # Extract user messages only
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        user_content = '\n'.join(user_messages)
        
        if len(user_messages) < 3:
            raise Exception("Need more conversation content for essay generation")
        
        # Generate essay using Ollama
        essay_text = self._generate_essay_with_ollama(user_content, word_count)
        
        # Generate title
        title = self._generate_title(essay_text, user_content)
        
        # Extract notes
        notes = self._extract_notes_from_conversation(user_messages)
        
        return {
            'title': title,
            'essay': essay_text,
            'word_count': len(essay_text.split()),
            'notes': notes,
            'generated_at': datetime.now().isoformat()
        }
    
    def _generate_essay_with_ollama(self, user_content: str, word_count: int) -> str:
        """Generate essay using Ollama"""
        
        prompt = f"""You are an expert college admissions essay writer. Write a compelling {word_count}-word personal statement based on this student's authentic responses from a brainstorming conversation.

STUDENT'S ACTUAL WORDS:
{user_content}

REQUIREMENTS:
- {word_count} words
- Personal statement format
- Authentic voice matching the student's responses
- Compelling narrative with specific examples
- Show personal growth and character development
- Connect to college readiness

STRUCTURE:
- Engaging opening hook
- Detailed body with specific examples from their experience
- Meaningful conclusion connecting to their future

Write the complete essay based on their authentic story:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": word_count * 2,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                essay = result.get("response", "").strip()
                return self._clean_essay(essay)
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Essay generation failed: {e}")
            return self._generate_fallback_essay(user_content, word_count)
    
    def _generate_title(self, essay_text: str, user_content: str) -> str:
        """Generate essay title"""
        
        # Extract key themes for title
        content_sample = f"{essay_text[:200]}... {user_content[:200]}"
        
        title_prompt = f"""Based on this essay and student story, create a compelling 3-7 word title that captures the main theme:

{content_sample}

Generate just the title, nothing else:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": title_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 20,
                        "temperature": 0.8
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                title = result.get("response", "").strip()
                # Clean up title
                title = title.replace('"', '').replace("'", "").strip()
                if len(title) > 60:
                    title = title[:60] + "..."
                return title if title else "My Journey of Growth"
            
        except Exception as e:
            logger.error(f"Title generation failed: {e}")
        
        return "My Journey of Growth"
    
    def _clean_essay(self, essay: str) -> str:
        """Clean and format essay"""
        import re
        
        # Remove any instruction text that leaked through
        essay = re.sub(r'^.*?REQUIREMENTS:.*?\n', '', essay, flags=re.DOTALL)
        essay = re.sub(r'^.*?Write the complete.*?\n', '', essay, flags=re.DOTALL)
        
        # Clean up formatting
        essay = re.sub(r'\n\s*\n\s*\n', '\n\n', essay)  # Max 2 line breaks
        essay = re.sub(r'^\s+', '', essay, flags=re.MULTILINE)  # Remove leading spaces
        
        # Ensure proper paragraph structure
        paragraphs = [p.strip() for p in essay.split('\n\n') if p.strip()]
        essay = '\n\n'.join(paragraphs)
        
        return essay.strip()
    
    def _generate_fallback_essay(self, user_content: str, word_count: int) -> str:
        """Fallback essay when AI generation fails"""
        
        return f"""Based on our conversation, I can see you have a compelling story to tell. Your experiences have shaped you in meaningful ways, and the challenges you've faced have contributed to your personal growth.

{user_content[:300]}...

This essay would benefit from professional development to reach its full potential. The themes we discussed - your challenges, growth, and future aspirations - provide an excellent foundation for a college application essay.

[This is a fallback essay. Please check your AI connection for full essay generation.]"""
    
    def _extract_notes_from_conversation(self, user_messages: List[str]) -> Dict[str, List[str]]:
        """Extract key themes from conversation"""
        
        all_content = ' '.join(user_messages).lower()
        
        notes = {
            'challenges': [],
            'lessons': [],
            'emotions': [],
            'goals': []
        }
        
        # Simple keyword-based extraction
        if any(word in all_content for word in ['challenge', 'difficult', 'struggle', 'problem']):
            notes['challenges'].append('Discussed facing significant challenges')
        
        if any(word in all_content for word in ['learned', 'taught', 'realized', 'discovered']):
            notes['lessons'].append('Described important lessons and insights')
        
        if any(word in all_content for word in ['felt', 'emotion', 'scared', 'excited', 'proud']):
            notes['emotions'].append('Shared emotional experiences and growth')
        
        if any(word in all_content for word in ['college', 'future', 'goal', 'career', 'study']):
            notes['goals'].append('Connected experience to future aspirations')
        
        return notes
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get summary of conversation"""
        
        if conversation_id not in self.conversations:
            return {}
        
        conv = self.conversations[conversation_id]
        user_messages = [msg for msg in conv['messages'] if msg['role'] == 'user']
        
        return {
            'conversation_id': conversation_id,
            'student_name': conv['student_name'],
            'stage': conv['stage'],
            'message_count': len(conv['messages']),
            'user_message_count': len(user_messages),
            'created_at': conv['created_at'],
            'ready_for_essay': len(user_messages) >= 4
        }

# Test function
def test_ai_chat():
    """Test the AI chat system"""
    print("🤖 Testing AI Chat")
    print("=" * 20)
    
    chat = AIChat()
    
    # Check availability
    print(f"AI available: {chat.is_available()}")
    
    # Test conversation
    test_messages = [
        "My name is Alex",
        "I had to overcome a fear of public speaking when I joined the debate team",
        "I felt really nervous and scared at first",
        "I learned that I'm braver than I thought",
        "I want to study communications in college"
    ]
    
    conversation_id = "test_conv"
    
    for message in test_messages:
        response = chat.get_response(message, conversation_id)
        print(f"\nUser: {message}")
        print(f"AI: {response['message']}")
        print(f"Stage: {response['stage']}")
    
    # Test essay generation
    if conversation_id in chat.conversations:
        try:
            essay = chat.generate_essay(chat.conversations[conversation_id]['messages'])
            print(f"\nGenerated Essay Title: {essay['title']}")
            print(f"Word Count: {essay['word_count']}")
            print(f"Essay Preview: {essay['essay'][:200]}...")
        except Exception as e:
            print(f"Essay generation failed: {e}")

if __name__ == "__main__":
    test_ai_chat()