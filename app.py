from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for, flash
import json
import os
import uuid
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import google.generativeai as genai
import mimetypes
from PIL import Image
from pypdf import PdfReader
from docx import Document
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
import io
import re
import tempfile
import PyPDF2
from pdf2image import convert_from_bytes
from googleapiclient.discovery import build
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account

# Optional imports for enhanced processing
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Grade-Appropriate Rigor Level Guidelines for Pakistani ESL Students
GRADE_RIGOR_GUIDELINES = {
    "1-2": {
        "description": "Lower Primary (Ages 6-8)",
        "characteristics": [
            "Very simple vocabulary and concepts",
            "Short, clear instructions",
            "Concrete, hands-on activities", 
            "Basic foundational skills",
            "Heavy use of visuals and games",
            "Simple Pakistani cultural examples (family, home, friends)"
        ],
        "vocabulary_level": "Basic everyday words in both English and Urdu",
        "cognitive_complexity": "Recognition, recall, simple identification",
        "activity_style": "Play-based, movement activities, show-and-tell, pointing games"
    },
    "3": {
        "description": "Mid Primary (Age 9)",
        "characteristics": [
            "Building vocabulary with more detailed words",
            "Simple categorization and sorting",
            "Beginning analysis skills",
            "Mix of concrete and abstract concepts",
            "Introduction to simple rules and patterns"
        ],
        "vocabulary_level": "Expanded vocabulary with simple academic terms",
        "cognitive_complexity": "Understanding, simple analysis, basic categorization",
        "activity_style": "Group sorting, simple problem-solving, basic research"
    },
    "4-5": {
        "description": "Upper Primary (Ages 10-11)",
        "characteristics": [
            "More complex vocabulary and academic terms",
            "Abstract thinking and analysis",
            "Application and synthesis of concepts",
            "Critical thinking activities",
            "Independent research and exploration"
        ],
        "vocabulary_level": "Academic vocabulary with technical terms",
        "cognitive_complexity": "Analysis, synthesis, evaluation, application",
        "activity_style": "Research projects, debates, creative writing, complex problem-solving"
    }
}

SUBJECT_SPECIFIC_RIGOR = {
    "English": {
        "1-2": "Focus on basic phonics, simple words, listening skills",
        "3": "Simple grammar rules, reading comprehension, basic writing",
        "4-5": "Complex grammar, literature analysis, essay writing"
    },
    "Math": {
        "1-2": "Numbers 1-100, basic addition/subtraction, shapes",
        "3": "Multiplication tables, fractions, measurement",
        "4-5": "Advanced operations, problem-solving, geometry"
    },
    "Urdu": {
        "1-2": "Basic Urdu letters and sounds, simple words",
        "3": "Reading simple Urdu sentences, basic writing",
        "4-5": "Urdu literature, complex sentence structure, poetry"
    },
    "Science": {
        "1-2": "Observations about nature, five senses, basic animals/plants",
        "3": "Simple experiments, life cycles, weather",
        "4-5": "Scientific method, ecosystems, physical/chemical changes"
    }
}

# TEACHING METHODOLOGY GUIDE FOR U-DOST AI CHATBOT
UDOST_TEACHING_METHODOLOGY = """
## LESSON PLANNING STRUCTURE (6 Essential Steps):
• **RECALL:** Quick review of previous learning/prerequisite knowledge
• **HOOK:** Engaging activity to capture student interest and introduce topic  
• **EXPLAIN:** Clear explanation using visual aids, examples, and demonstrations
• **GUIDED PRACTICE:** Teacher-led practice with student participation
• **INDEPENDENT PRACTICE:** Students work from textbook exercises independently
• **QUICK CONCLUSION:** Brief summary and key takeaways

## TEACHING STRATEGIES BY SKILL:

### READING STRATEGIES:
- Echo Reading, Choral Reading, Paired Reading
- Picture Walk, Prediction, Think-Aloud
- Phonics Blending, Sight Word Recognition
- Reading Comprehension Questions

### WRITING STRATEGIES:
- Sentence Starters, Writing Frames
- Guided Writing, Shared Writing
- Grammar Integration, Vocabulary Building
- Peer Editing, Self-Correction

### ORAL COMMUNICATION STRATEGIES:
- Show and Tell, Role Play, Storytelling
- Question-Answer Sessions, Group Discussions
- Pronunciation Practice, Vocabulary Games
- Listen and Repeat Activities

### COMPREHENSION STRATEGIES:
- KWL Charts (Know-Want-Learn)
- Story Mapping, Sequence Activities
- Main Idea and Details, Cause and Effect
- Making Connections, Inference Skills

### GRAMMAR STRATEGIES:
- Grammar Games, Pattern Practice
- Sentence Building, Error Correction
- Visual Grammar Charts, Examples and Non-examples
- Contextual Grammar Teaching

### VOCABULARY STRATEGIES:
- Picture-Word Association, Word Maps
- Synonym/Antonym Games, Context Clues
- Word Families, Vocabulary Journals
- Total Physical Response (TPR)

## PAKISTANI ESL CONTEXT REQUIREMENTS:
- All content must match the SPECIFIC EXERCISE from the selected chapter/book
- Tailor difficulty to Pakistani ESL students (English as Second Language)
- Consider age-appropriate rigor for grades 1-5
- Use simple, clear Pakistani English context
- Include Urdu translation support when needed
- Reference local cultural examples and contexts (Pakistani names, foods, places)
- Adjust complexity based on grade level
- Ensure activities match the exact textbook content selected
- Use familiar cultural references
- Provide pronunciation guides for difficult English words
- Include mother tongue support strategies
- Consider limited English vocabulary of students
- Focus on practical, communicative English skills
"""

app = Flask(__name__)

# Require secure session secret in production
SESSION_SECRET = os.environ.get('SESSION_SECRET')
if not SESSION_SECRET:
    raise ValueError("SESSION_SECRET environment variable must be set for security")

app.secret_key = SESSION_SECRET
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12MB max file size

# Session configuration for proper persistence
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


# Initialize AI client - prefer OpenAI for reliability
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# PostHog configuration
POSTHOG_KEY = os.environ.get('VITE_PUBLIC_POSTHOG_KEY', 'phc_ygiCdZb8vwOkLO5WIdGvdxzugrlGnaFxkW0F73sHyBF')
POSTHOG_HOST = os.environ.get('VITE_PUBLIC_POSTHOG_HOST', 'https://app.posthog.com')

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')

# Removed Google Drive integration - using direct OpenAI responses

# Database connection helper
def get_db_connection():
    """Get database connection with RealDictCursor for easy column access"""
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error occurred")  # Don't log sensitive details
        return None

# Initialize both AI clients independently for reliable fallback
openai_client = None
gemini_model = None

# Initialize OpenAI if key is available
if OPENAI_API_KEY:
    try:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully")
    except ImportError:
        print("OpenAI library not available")
        openai_client = None
    except Exception as e:
        print(f"OpenAI client initialization failed: {e}")
        openai_client = None

# Initialize Gemini if key is available (independent of OpenAI)
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model initialized successfully")
    except Exception as e:
        print(f"Gemini model initialization failed: {e}")
        gemini_model = None

# Log available AI services
if openai_client and gemini_model:
    print("Both OpenAI and Gemini available")
elif openai_client:
    print("Only OpenAI available")
elif gemini_model:
    print("Only Gemini available")
else:
    print("No AI services available - using fallback responses")

# Upload configuration
UPLOAD_DIR = 'uploads'
AUDIO_DIR = os.path.join(UPLOAD_DIR, 'audio')
IMAGES_DIR = os.path.join(UPLOAD_DIR, 'images')
DOCS_DIR = os.path.join(UPLOAD_DIR, 'docs')
META_DIR = os.path.join(UPLOAD_DIR, 'meta')

# Allowed file types
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'}
ALLOWED_AUDIO_TYPES = {'audio/webm', 'audio/ogg', 'audio/wav', 'audio/mp3'}
ALLOWED_DOC_TYPES = {'application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}

def init_upload_directories():
    """Initialize upload directories"""
    for upload_dir in [UPLOAD_DIR, AUDIO_DIR, IMAGES_DIR, DOCS_DIR, META_DIR]:
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

# File handling utility functions
def get_file_mime_type(file_path):
    """Get MIME type of a file using Python's mimetypes module"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def generate_unique_filename(original_filename):
    """Generate a unique filename using UUID"""
    file_extension = os.path.splitext(original_filename)[1]
    unique_id = str(uuid.uuid4())
    return f"{unique_id}{file_extension}"

def save_file_metadata(file_id, original_name, mime_type, file_size, file_type):
    """Save file metadata for tracking and cleanup"""
    metadata = {
        'id': file_id,
        'original_name': original_name,
        'mime_type': mime_type,
        'size': file_size,
        'type': file_type,
        'session_id': session.get('session_id', 'anonymous'),
        'created_at': datetime.now().isoformat(),
        'accessed_at': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(META_DIR, f"{file_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return metadata

def get_file_metadata(file_id):
    """Get file metadata if it exists"""
    metadata_path = os.path.join(META_DIR, f"{file_id}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

# Removed JSON loading function - using direct OpenAI responses

# Removed chapter parsing function - using direct OpenAI responses

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return ""

def strip_image_metadata(image_path):
    """Remove EXIF data from images for privacy"""
    try:
        image = Image.open(image_path)
        data = list(image.getdata())
        image_without_exif = Image.new(image.mode, image.size)
        image_without_exif.putdata(data)
        image_without_exif.save(image_path)
    except Exception as e:
        print(f"Error stripping image metadata: {e}")

def cleanup_old_files():
    """Clean up files older than 7 days"""
    try:
        cutoff_time = datetime.now() - timedelta(days=7)
        
        for metadata_file in os.listdir(META_DIR):
            if metadata_file.endswith('.json'):
                metadata_path = os.path.join(META_DIR, metadata_file)
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    created_at = datetime.fromisoformat(metadata['created_at'])
                    if created_at < cutoff_time:
                        # Remove the actual file
                        file_id = metadata['id']
                        file_type = metadata['type']
                        
                        if file_type == 'audio':
                            file_path = os.path.join(AUDIO_DIR, f"{file_id}.webm")
                        elif file_type == 'image':
                            # Find the actual file (could have different extensions)
                            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                                file_path = os.path.join(IMAGES_DIR, f"{file_id}{ext}")
                                if os.path.exists(file_path):
                                    break
                        elif file_type == 'document':
                            # Find the actual file
                            for ext in ['.pdf', '.txt', '.docx']:
                                file_path = os.path.join(DOCS_DIR, f"{file_id}{ext}")
                                if os.path.exists(file_path):
                                    break
                        
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        
                        # Remove metadata file
                        os.remove(metadata_path)
                        
                except Exception as e:
                    print(f"Error cleaning up file {metadata_file}: {e}")
                    
    except Exception as e:
        print(f"Error during cleanup: {e}")

def get_ai_response(user_message, conversation_type="general", session_context=None):
    """Get AI-powered response using OpenAI or Gemini with grade-appropriate rigor levels"""
    
    # Build contextual information from session (new simplified context)
    context_info = ""
    grade_rigor_info = ""
    
    if session_context:
        # Get current session data (from simplified chatbot flow)
        grade = session_context.get('grade', '')
        subject = session_context.get('subject', '')
        selected_feature = session_context.get('selected_feature', '')
        activity_type = session_context.get('activity_type', '')
        
        if grade and subject:
            # Robust grade parsing - convert to integer and handle different formats
            grade_num = None
            try:
                if isinstance(grade, int):
                    grade_num = grade
                elif isinstance(grade, str):
                    # Extract digits from grade string (e.g., "Grade 1", "G1", "1")
                    import re
                    grade_match = re.search(r'(\d+)', str(grade))
                    if grade_match:
                        grade_num = int(grade_match.group(1))
                
                # Validate grade range (1-5 for Pakistani primary)
                if grade_num and 1 <= grade_num <= 5:
                    # Map to rigor levels
                    rigor_key = "1-2" if grade_num in [1, 2] else "3" if grade_num == 3 else "4-5"
                else:
                    # Default to simplest rigor for safety and clamp grade_num
                    rigor_key = "1-2"
                    grade_num = 1  # Clamp to valid range for consistent prompts
            except:
                # Fallback to simplest rigor for safety
                rigor_key = "1-2"
                grade_num = 1
            
            # Subject normalization - handle common variants
            subject_mapping = {
                "Mathematics": "Math",
                "General Science": "Science", 
                "Social Science": "Social Studies",
                "Islamic Studies": "Islamiyat"
            }
            normalized_subject = subject_mapping.get(subject, subject)
            
            rigor_guidelines = GRADE_RIGOR_GUIDELINES.get(rigor_key, {})
            subject_rigor = SUBJECT_SPECIFIC_RIGOR.get(normalized_subject, {}).get(rigor_key, f'Age-appropriate {normalized_subject} concepts for Grade {grade_num}')
            
            context_info = f"""

CURRENT EDUCATIONAL CONTEXT:
- Grade: {grade_num}
- Subject: {normalized_subject}
- Content Type: {selected_feature or activity_type or 'General'}
- Target Age: {rigor_guidelines.get('description', f'Grade {grade_num}')}

GRADE-APPROPRIATE RIGOR LEVEL ({rigor_key}):
- Vocabulary Level: {rigor_guidelines.get('vocabulary_level', 'Age-appropriate')}
- Cognitive Complexity: {rigor_guidelines.get('cognitive_complexity', 'Grade-appropriate')}
- Activity Style: {rigor_guidelines.get('activity_style', 'Engaging activities')}
- Subject Focus: {subject_rigor}

IMPORTANT: Match content complexity to Grade {grade_num} Pakistani ESL students. Use {rigor_guidelines.get('vocabulary_level', 'age-appropriate')} vocabulary and {rigor_guidelines.get('cognitive_complexity', 'appropriate')} thinking skills."""
            
            # Build detailed rigor characteristics
            characteristics = rigor_guidelines.get('characteristics', [])
            if characteristics:
                grade_rigor_info = f"""

GRADE {grade_num} RIGOR REQUIREMENTS:
""" + "\n".join([f"- {char}" for char in characteristics])
    
    # Create system prompt based on conversation type
    if conversation_type == "teaching":
        system_prompt = f"""You are U-DOST, a friendly Pakistani teacher assistant specifically designed for Pakistani primary education (grades 1-5). You MUST follow the Pakistani Teaching Methodology Guidelines:

{UDOST_TEACHING_METHODOLOGY}

CORE REQUIREMENTS:
- ALWAYS follow the 6-step lesson structure: RECALL → HOOK → EXPLAIN → GUIDED PRACTICE → INDEPENDENT PRACTICE → QUICK CONCLUSION
- Use skill-specific teaching strategies based on the focus area (Reading, Writing, Grammar, Vocabulary, etc.)
- Include Pakistani cultural examples and contexts (Pakistani names, foods, festivals, places)
- Provide Roman Urdu support for difficult English words (NO Arabic Urdu script - only Roman Urdu)
- Match content difficulty to Pakistani ESL students' level
- Reference the specific textbook content and chapter provided

CONVERSATIONAL STYLE (CRITICAL):
- Use natural Roman Urdu mixed with English (like Pakistani teachers speak)
- Be friendly and conversational: "Hello teacher! Bilkul easy hai!"
- Think in Roman Urdu: "Grade 1 ke bachon ke liye simple rakhna padega..."
- Use Roman Urdu words naturally: "bachon", "teacher", "bilkul", "acha", "kaisa laga"
- End with friendly questions: "Aur activities chahiye? Kaisa laga teacher? 😊"

EXAMPLES OF CORRECT STYLE:
✅ "Hello teacher! Grade 1 ke bachon ke liye nouns sikhana hai? Bilkul easy hai!"
✅ "Classroom ka khazana game try kariye - bachon ko bahut maza aayega"  
✅ "Ahmed aur Fatima ke examples use karke family members sikhayen"
❌ NEVER use Arabic Urdu script: "ارے ٹیچر صاحب"

RESPONSE GUIDELINES:
- Be encouraging and supportive like a Pakistani teacher speaking naturally
- Use conversational Roman Urdu + English mix throughout
- Include Pakistani cultural elements (biryani, chawal, Eid, cricket, etc.)
- Think aloud in Roman Urdu about teaching strategies
- Always end with friendly Roman Urdu phrases

WHEN CREATING EDUCATIONAL CONTENT:
- Lesson Plans: Follow 6-step structure with Roman Urdu explanations
- Teaching Strategies: Use conversational Roman Urdu style
- Activities: Mix Roman Urdu instructions with English content
- Assessments: Ask in Roman Urdu style: "Assessment ke liye kya banayein?"
- Definitions: Explain in Roman Urdu, never Arabic script

Remember: Speak like a friendly Pakistani teacher using natural Roman Urdu + English. NO Arabic Urdu script anywhere!""" + context_info + grade_rigor_info
    else:
        system_prompt = """You are a helpful, knowledgeable, and conversational AI assistant. Be friendly, professional, and approachable. Match the user's communication style, be concise but thorough, and help with any questions or tasks they have. Your goal is to be genuinely helpful while maintaining a natural, conversational tone.""" + context_info
    
    # Try OpenAI first with conversation history
    if openai_client:
        try:
            # Get chat history from session
            chat_history = session.get('chat_history', [])
            
            # Build messages with system prompt and history
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent chat history (keep last 15 exchanges to manage token limits)
            if len(chat_history) > 30:  # 30 messages = 15 exchanges
                chat_history = chat_history[-30:]
                session['chat_history'] = chat_history
                session.modified = True
            
            messages.extend(chat_history)
            messages.append({"role": "user", "content": user_message})
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": assistant_response})
            session['chat_history'] = chat_history
            session.modified = True
            
            return assistant_response
            
        except Exception as e:
            # Sanitize error logging to prevent API key leakage
            error_msg = str(e)
            if 'api key' in error_msg.lower() or 'authentication' in error_msg.lower():
                print("OpenAI API error: Authentication failed - please check API key configuration")
            else:
                print(f"OpenAI API error: {type(e).__name__}")
    
    # Fallback to Gemini with conversation history
    if gemini_model:
        try:
            # Get chat history for context
            chat_history = session.get('chat_history', [])
            
            # Build conversation context
            conversation_context = ""
            if chat_history:
                # Get last few exchanges for context
                recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
                for msg in recent_history:
                    role = "Assistant" if msg["role"] == "assistant" else "User"
                    conversation_context += f"{role}: {msg['content']}\n"
            
            full_prompt = f"{system_prompt}\n\nConversation History:\n{conversation_context}\nUser: {user_message}\n\nAssistant:"
            response = gemini_model.generate_content(full_prompt)
            
            assistant_response = response.text
            
            # Update chat history
            if 'chat_history' not in session:
                session['chat_history'] = []
            session['chat_history'].append({"role": "user", "content": user_message})
            session['chat_history'].append({"role": "assistant", "content": assistant_response})
            session.modified = True
            
            return assistant_response
            
        except Exception as e:
            # Sanitize error logging to prevent API key leakage
            error_msg = str(e)
            if 'api key' in error_msg.lower() or 'authentication' in error_msg.lower():
                print("Gemini API error: Authentication failed - please check API key configuration")
            else:
                print(f"Gemini API error: {type(e).__name__}")
    
    # Final fallback with Pakistani teacher responses
    return get_pakistani_teacher_fallback(user_message, session_context)

def transcribe_audio(audio_file_path):
    """Transcribe audio using OpenAI Whisper API with Urdu language support"""
    try:
        if openai_client:
            # Use OpenAI Whisper API for transcription
            with open(audio_file_path, 'rb') as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ur"  # Urdu language code, can auto-detect if omitted
                )
                return transcript.text
        else:
            # Fallback if OpenAI is not available
            return "Voice message received. (Speech-to-text service not available)"
    except Exception as e:
        print(f"Audio transcription error: {e}")
        # Try without language specification (auto-detect)
        try:
            if openai_client:
                with open(audio_file_path, 'rb') as audio_file:
                    transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=audio_file
                    )
                    return transcript.text
        except Exception as e2:
            print(f"Audio transcription fallback error: {e2}")
        return "Voice message received. (Transcription failed)"

# Predefined books from the curriculum (no uploads needed)
def get_predefined_books():
    """Returns the predefined books from the Pakistani curriculum"""
    return {
        'Grade 1': {
            'English': {'Grade 1 English': 'CHAPTER 1 FINAL.pdf'},
            'Mathematics': {'Grade 1 Maths': 'Grade1Math-ForFederal.pdf'},
            'Urdu': {'Grade 1 Urdu': 'Grade 1 compiled-001.pdf'},
            'General Knowledge': {
                'Grade 1 General Knowledge Science': 'GK 1 Science Indesign, Edited.pdf',
                'Grade 1 General Knowledge SST': 'GK 1 SST indesign_Edited.pdf'
            },
            'Islamiyat': {'Grade 1 Islamiat': 'IslGrade1 with nazra.pdf'}
        },
        'Grade 2': {
            'English': {'Grade 2 English': 'Grade 2 Edited.pdf'},
            'Mathematics': {'Grade 2 Math': 'Grade2Math.pdf'},
            'Urdu': {'Grade 2 Urdu': 'Grade 2a.pdf'},
            'General Knowledge': {
                'Grade 2 General Knowledge Science': 'GK 2 Science indesign.pdf',
                'Grade 2 General Knowledge SST': 'GK 2 SST indesign-001.pdf'
            },
            'Islamiyat': {'Grade 2 Islamiat': 'IslamiatG2 new.pdf'}
        },
        'Grade 3': {
            'English': {'Grade 3 English': 'English G3 updated.pdf'},
            'Mathematics': {'Grade 3 Math': 'Grade3MathChanges.pdf'},
            'Urdu': {'Grade 3 Urdu': 'G3.pdf'},
            'General Knowledge': {
                'Grade 3 General Knowledge Science': 'GK 3 Science.pdf',
                'Grade 3 General Knowledge SST': 'GK 3 SST indesign.pdf'
            },
            'Islamiyat': {'Grade 3 Islamiat': 'G3 new.pdf'}
        },
        'Grade 4': {
            'English': {'English Adventure: Learn, Express and Succeed!': 'Grade 4_1758655549357.pdf'},
            'Mathematics': {'Grade 4 Math': 'Grade4Mathupdatedwithout logo.pdf'},
            'Urdu': {'Grade 4 Urdu': 'GRADE 4.pdf'},
            'General Science': {'Grade 4 General Science': 'Science 4 indesign updated.pdf'},
            'Social Studies': {'Grade 4 Social Studies': 'sst 4 indesign updated.pdf'},
            'Islamiyat': {'Grade 4 Islamiat': 'Grade 4.pdf'}
        },
        'Grade 5': {
            'English': {'Grade 5 English': 'English G5 updated.pdf'},
            'Mathematics': {'Grade 5 Math': 'Grade5 Math - full book.pdf'},
            'Urdu': {'Grade 5 Urdu': 'Grade 5.pdf'},
            'General Science': {'Grade 5 General Science': 'Science 5 indesign.pdf'},
            'Social Studies': {'Grade 5 Social Studies': 'SST 5 indesign updated.pdf'},
            'Islamiyat': {'Grade 5 Islamiat': 'IslGrade5Complete.pdf'}
        }
    }

# ============= GOOGLE DRIVE INTEGRATION FUNCTIONS =============

def list_drive_files_paginated(query, fields="files(id, name)", page_size=100):
    """Helper function to list Google Drive files with pagination support"""
    if not drive_service:
        return []
    
    all_files = []
    page_token = None
    
    try:
        while True:
            request_params = {
                'q': query,
                'fields': f"nextPageToken, {fields}",
                'pageSize': page_size
            }
            
            if page_token:
                request_params['pageToken'] = page_token
            
            results = drive_service.files().list(**request_params).execute()
            
            files = results.get('files', [])
            all_files.extend(files)
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break
                
        return all_files
        
    except Exception as e:
        # Sanitize error logging to prevent credential leakage
        print(f"Error in paginated Drive listing: {type(e).__name__}")
        return []

# Removed Google Drive functions - using direct OpenAI responses

def extract_text_from_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes using multiple methods"""
    text_content = ""
    
    try:
        # Method 1: Try PyPDF2 first
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text.strip():
                text_content += page_text + "\n"
        
        # If PyPDF2 didn't extract much text, try OCR
        if len(text_content.strip()) < 100 and TESSERACT_AVAILABLE:
            text_content = extract_text_with_ocr(pdf_bytes)
            
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        if TESSERACT_AVAILABLE:
            text_content = extract_text_with_ocr(pdf_bytes)
    
    return text_content

def extract_text_with_ocr(pdf_bytes):
    """Extract text using OCR from PDF bytes"""
    if not TESSERACT_AVAILABLE:
        return ""
    
    try:
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=10)  # Limit to first 10 pages
        
        extracted_text = ""
        for i, image in enumerate(images):
            # Convert PIL image to text
            page_text = pytesseract.image_to_string(image)
            extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        return extracted_text
        
    except Exception as e:
        print(f"Error with OCR extraction: {e}")
        return ""

def is_title_candidate(line):
    """Check if line could be a story title based on format rules"""
    if not line or len(line) < 8 or len(line) > 60:
        return False
    
    # Count words (2-8 words for titles)
    words = line.split()
    if len(words) < 2 or len(words) > 8:
        return False
    
    # Must not start with digit or end with colon
    if line[0].isdigit() or line.endswith(':'):
        return False
    
    # Must not contain colon anywhere
    if ':' in line:
        return False
    
    # TOC filter: reject lines with dotted leaders and page numbers
    if re.search(r'\.\.{2,}\s*\d+$', line) or re.search(r'\s+\d{1,3}$', line):
        return False
    
    # Hard exclude prefixes/keywords (case-insensitive)
    blacklist_prefixes = [
        'activity', 'exercise', 'grammar', 'writing', 'reading', 'objectives',
        'learning outcomes', 'vocabulary', 'step', 'project', 'tips',
        'connect and create', 'choose your topic', 'state your opinion',
        'answer key', 'comprehension', 'unit', 'lesson', 'chapter'
    ]
    
    line_lower = line.lower()
    for prefix in blacklist_prefixes:
        if line_lower.startswith(prefix):
            return False
    
    # Must be Title Case or contain apostrophes (for story titles)
    has_title_case = any(word[0].isupper() for word in words if len(word) > 0)
    has_apostrophe = "'" in line
    
    return has_title_case or has_apostrophe

def is_heading_or_label(line):
    """Check if line is a heading or label that marks end of story"""
    if not line:
        return False
    
    line_lower = line.lower()
    
    # Section endings
    section_markers = [
        'comprehension', 'grammar', 'writing', 'activity', 'project', 'tips',
        'answer key', 'objectives', 'vocabulary', 'step', 'connect and create',
        'choose your topic', 'state your opinion', 'diving deeper'
    ]
    
    return any(marker in line_lower for marker in section_markers) or line.isupper()

def build_paragraphs(lines, start_idx):
    """Build paragraphs from lines starting at start_idx"""
    paragraphs = []
    current_paragraph = []
    
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        
        # Empty line or heading marks paragraph break
        if not line or is_heading_or_label(line):
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            if is_heading_or_label(line):
                break
        elif len(line) > 3:  # Ignore very short lines
            current_paragraph.append(line)
    
    # Add remaining content
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs

def validate_narrative(paragraph):
    """Check if paragraph contains narrative content"""
    if not paragraph or len(paragraph) < 80:
        return False
    
    words = paragraph.split()
    if len(words) < 15:
        return False
    
    # Count sentences (approximate)
    sentence_markers = paragraph.count('.') + paragraph.count('!') + paragraph.count('?')
    if sentence_markers < 3:
        return False
    
    # Check average sentence length
    avg_sentence_length = len(words) / max(sentence_markers, 1)
    if avg_sentence_length < 8:
        return False
    
    # Count pronouns
    pronouns = ['he', 'she', 'they', 'i', 'we', 'his', 'her', 'their', 'him']
    pronoun_count = sum(1 for word in words if word.lower() in pronouns)
    if pronoun_count < 3:
        return False
    
    # Count past-tense indicators
    past_tense_words = ['was', 'were', 'had', 'went', 'said', 'came', 'made', 'took', 'gave']
    past_tense_count = sum(1 for word in words if word.lower() in past_tense_words)
    past_tense_count += sum(1 for word in words if word.lower().endswith('ed'))
    
    if past_tense_count < 3:
        return False
    
    return True

def find_story_end(lines, start_idx):
    """Find where story content ends"""
    for i in range(start_idx + 50, min(len(lines), start_idx + 300)):
        line = lines[i].strip()
        if is_heading_or_label(line):
            return i
    return min(len(lines), start_idx + 300)

def categorize_text_into_chapters(text_content, grade, subject):
    """Automatically categorize extracted text into chapters and exercises"""
    
    if not text_content:
        return {}
    
    lines = text_content.split('\n')
    story_chapters = []
    
    print(f"Analyzing {len(lines)} lines for story content...")
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Stage 1: Check if line could be a title
        if not is_title_candidate(line):
            continue
        
        # Stage 2: Look for narrative content after title
        paragraphs = build_paragraphs(lines, i + 1)
        
        if not paragraphs:
            continue
        
        # Validate first paragraph for narrative content
        first_paragraph = paragraphs[0]
        if not validate_narrative(first_paragraph):
            continue
        
        # Found a valid story - extract full content
        story_end = find_story_end(lines, i)
        story_content = []
        
        for line_idx in range(i + 1, story_end):
            if line_idx < len(lines):
                content_line = lines[line_idx].strip()
                if content_line and len(content_line) > 3:
                    story_content.append(content_line)
        
        # Ensure substantial content
        total_words = sum(len(line.split()) for line in story_content)
        if total_words >= 200:
            story_chapters.append({
                'title': line,
                'content': story_content,
                'word_count': total_words
            })
            print(f"Found story: '{line}' ({total_words} words)")
    
    # Remove duplicates and limit to top stories by content length
    seen_titles = set()
    unique_stories = []
    
    for story in sorted(story_chapters, key=lambda x: x['word_count'], reverse=True):
        normalized_title = story['title'].lower().strip()
        if normalized_title not in seen_titles:
            seen_titles.add(normalized_title)
            unique_stories.append(story)
    
    # Create chapters from found stories (limit to 20)
    chapters = {}
    for i, story in enumerate(unique_stories[:20]):
        chapter_name = f"Chapter {i + 1}: {story['title']}"
        chapter_content = '\n'.join(story['content'])
        chapters[chapter_name] = categorize_chapter_content(chapter_content)
    
    print(f"Final result: {len(chapters)} chapters extracted")
    
    return chapters

def categorize_chapter_content(content):
    """Categorize content within a chapter into exercises by type"""
    
    exercises = {
        "Reading": [],
        "Writing": [], 
        "Grammar": [],
        "Activities": []
    }
    
    # Define exercise patterns for different types
    reading_patterns = [
        r'(?i)read(?:ing)?\s+(?:the\s+)?(?:story|text|passage)',
        r'(?i)comprehension',
        r'(?i)understand(?:ing)?',
        r'(?i)meaning',
    ]
    
    writing_patterns = [
        r'(?i)writ(?:e|ing)',
        r'(?i)compose',
        r'(?i)essay',
        r'(?i)paragraph',
    ]
    
    grammar_patterns = [
        r'(?i)grammar',
        r'(?i)noun|verb|adjective',
        r'(?i)sentence',
        r'(?i)punctuation',
    ]
    
    activity_patterns = [
        r'(?i)activit(?:y|ies)',
        r'(?i)exercise',
        r'(?i)practice',
        r'(?i)drill',
    ]
    
    # Split content into potential exercises
    lines = content.split('\n')
    current_exercise = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check patterns and categorize
        exercise_type = "Activities"  # Default
        
        for pattern in reading_patterns:
            if re.search(pattern, line):
                exercise_type = "Reading"
                break
        
        if exercise_type == "Activities":  # Still default, check others
            for pattern in writing_patterns:
                if re.search(pattern, line):
                    exercise_type = "Writing"
                    break
                    
        if exercise_type == "Activities":  # Still default, check grammar
            for pattern in grammar_patterns:
                if re.search(pattern, line):
                    exercise_type = "Grammar"
                    break
        
        # Add exercise to appropriate category
        if len(line) > 10:  # Only meaningful content
            exercise_title = line[:100] + "..." if len(line) > 100 else line
            exercises[exercise_type].append({
                "title": exercise_title,
                "type": exercise_type.lower()
            })
    
    # Ensure each category has at least some exercises
    for category in exercises:
        if not exercises[category]:
            exercises[category] = [
                {"title": f"Practice {category}", "type": category.lower()},
                {"title": f"{category} Exercise", "type": category.lower()}
            ]
    
    return exercises

def create_default_chapter_structure(grade, subject, content):
    """Create default chapter structure when automatic detection fails"""
    
    # Create 3-5 chapters based on content length
    content_length = len(content)
    num_chapters = min(5, max(3, content_length // 1000))
    
    chapters = {}
    
    for i in range(1, num_chapters + 1):
        chapter_title = f"Chapter {i}: {subject} Fundamentals {i}"
        chapters[chapter_title] = {
            "Reading": [
                {"title": f"Reading Comprehension {i}", "type": "reading"},
                {"title": f"Vocabulary Building {i}", "type": "vocabulary"}
            ],
            "Writing": [
                {"title": f"Writing Practice {i}", "type": "writing"},
                {"title": f"Creative Expression {i}", "type": "creative"}
            ],
            "Grammar": [
                {"title": f"Grammar Rules {i}", "type": "grammar"},
                {"title": f"Sentence Structure {i}", "type": "structure"}
            ],
            "Activities": [
                {"title": f"Practice Activities {i}", "type": "practice"},
                {"title": f"Review Exercises {i}", "type": "review"}
            ]
        }
    
    return chapters

def load_books_from_google_drive():
    """Main function to load all books from Google Drive and process them"""
    if not drive_service:
        print("Google Drive service not available - using static books")
        return get_predefined_books()
    
    try:
        # Scan Google Drive folder structure
        drive_books = scan_google_drive_folders()
        
        if not drive_books:
            print("No books found in Google Drive - using static books")
            return get_predefined_books()
        
        # Process each book and extract content
        processed_books = {}
        
        for grade_key, subjects in drive_books.items():
            processed_books[grade_key] = {}
            
            for subject_name, books in subjects.items():
                processed_books[grade_key][subject_name] = {}
                
                for book_title, file_id in books.items():
                    print(f"Processing {grade_key} - {subject_name} - {book_title}")
                    
                    # Download and process PDF
                    pdf_bytes = download_pdf_from_drive(file_id)
                    
                    if pdf_bytes:
                        # Extract text
                        text_content = extract_text_from_pdf_bytes(pdf_bytes)
                        
                        # Categorize into chapters and exercises
                        grade_num = int(re.search(r'(\d+)', grade_key).group(1))
                        chapters = categorize_text_into_chapters(text_content, grade_num, subject_name)
                        
                        processed_books[grade_key][subject_name][book_title] = {
                            'file_id': file_id,
                            'chapters': chapters,
                            'extracted_text': text_content[:1000] + "..." if len(text_content) > 1000 else text_content
                        }
                    else:
                        print(f"Failed to download {book_title}")
        
        print(f"Successfully processed {len(processed_books)} grades from Google Drive")
        return processed_books
        
    except Exception as e:
        # Sanitize error logging to prevent credential leakage
        print(f"Error loading books from Google Drive: {type(e).__name__}")
        return get_predefined_books()

def get_auto_loaded_book_content(grade, subject):
    """Auto-load book content with chapters and exercises based on grade and subject"""
    
    # PRIORITY: Special handling for Grade 4 English - use JSON file FIRST
    if str(grade) == "4" and subject and subject.lower() == "english":
        json_content = load_grade4_english_json()
        if json_content:
            debug_info = {
                'drive_status': '❌ Not Connected (JSON prioritized)',
                'pdf_status': '✅ Loaded from JSON file',
                'chapters_found': json_content.get('total_chapters', 0),
                'content_preview': json_content.get('extracted_text', '')[:100],
                'source': 'json_file'
            }
            json_content['debug_info'] = debug_info
            print(f"🎯 PRIORITY: Successfully loaded Grade 4 English from JSON file with {json_content.get('total_chapters', 0)} chapters")
            return json_content
        else:
            print("⚠️ Failed to load Grade 4 English JSON file, falling back to other methods")
    
    # Standard debug info for all other cases
    debug_info = {
        'drive_status': '❌ Not Connected',
        'pdf_status': '❌ No PDF Found',
        'chapters_found': 0,
        'content_preview': 'No content extracted',
        'source': 'unknown'
    }
    
    # Try Google Drive integration (for all other grades/subjects or if JSON fails)
    if drive_service:
        debug_info['drive_status'] = '✅ Connected'
        try:
            # Load books from Google Drive
            drive_books = load_books_from_google_drive()
            
            grade_key = f'Grade {grade}'
            if grade_key in drive_books and subject in drive_books[grade_key]:
                subject_books = drive_books[grade_key][subject]
                
                if subject_books:
                    # Get the first book for this subject
                    book_title = list(subject_books.keys())[0]
                    book_data = subject_books[book_title]
                    
                    debug_info['pdf_status'] = '✅ Loaded from Google Drive'
                    debug_info['chapters_found'] = len(book_data.get('chapters', {}))
                    debug_info['content_preview'] = book_data.get('extracted_text', '')[:100]
                    debug_info['source'] = 'google_drive'
                    
                    return {
                        'title': book_title,
                        'filename': book_data.get('file_id', ''),
                        'grade': grade,
                        'subject': subject,
                        'chapters': book_data.get('chapters', {}),
                        'total_chapters': len(book_data.get('chapters', {})),
                        'source': 'google_drive',
                        'file_id': book_data.get('file_id'),
                        'debug_info': debug_info
                    }
        except Exception as e:
            # Sanitize error logging to prevent credential leakage
            print(f"Error loading from Google Drive, falling back to static books: {type(e).__name__}")
    else:
        debug_info['drive_status'] = '❌ Service Account Not Configured'
    
    # Try to find and read actual PDF files from attached_assets
    try:
        import os
        pdf_dir = 'attached_assets'
        if os.path.exists(pdf_dir):
            # Look for PDFs that match this grade/subject
            grade_patterns = [f'grade {grade}', f'grade{grade}', f'g{grade}', str(grade)]
            subject_patterns = [subject.lower(), subject[:4].lower()]
            
            for filename in os.listdir(pdf_dir):
                if filename.lower().endswith('.pdf'):
                    filename_lower = filename.lower()
                    
                    # Check if this PDF matches the grade/subject
                    grade_match = any(pattern in filename_lower for pattern in grade_patterns)
                    subject_match = any(pattern in filename_lower for pattern in subject_patterns)
                    
                    if grade_match or subject_match:
                        pdf_path = os.path.join(pdf_dir, filename)
                        
                        # Extract actual text from the PDF
                        extracted_text = extract_text_from_pdf(pdf_path)
                        
                        if extracted_text:
                            # Extract real chapters from the PDF content
                            real_chapters = categorize_text_into_chapters(extracted_text, grade, subject)
                            
                            debug_info['pdf_status'] = '✅ Loaded from Local PDF'
                            debug_info['chapters_found'] = len(real_chapters) if real_chapters else 0
                            debug_info['content_preview'] = extracted_text[:100] + "..."
                            debug_info['source'] = 'local_pdf'
                            
                            if real_chapters:
                                return {
                                    'title': f'Grade {grade} {subject} Textbook',
                                    'filename': filename,
                                    'grade': grade,
                                    'subject': subject,
                                    'chapters': real_chapters,
                                    'total_chapters': len(real_chapters),
                                    'source': 'local_pdf',
                                    'extracted_text': extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text,
                                    'debug_info': debug_info
                                }
                        break
    except Exception as e:
        print(f"Error reading local PDFs: {type(e).__name__}")
    
    # Fallback to predefined static books
    predefined_books = get_predefined_books()
    
    # Get the appropriate book for this grade and subject
    grade_key = f'Grade {grade}'
    if grade_key not in predefined_books:
        debug_info['pdf_status'] = '❌ No PDF Found - No Predefined Book'
        debug_info['source'] = 'none'
        return None
        
    subject_books = predefined_books[grade_key].get(subject, {})
    if not subject_books:
        debug_info['pdf_status'] = '❌ No PDF Found - No Subject Book'
        debug_info['source'] = 'none'
        return None
    
    # Get the first (and usually only) book for this subject
    book_title = list(subject_books.keys())[0]
    book_filename = subject_books[book_title]
    
    # Generate structured content as if extracted via OCR
    book_content = generate_book_structure(grade, subject, book_title)
    
    debug_info['pdf_status'] = '⚠️ Using Static/Fake Data'
    debug_info['chapters_found'] = len(book_content['chapters'])
    debug_info['content_preview'] = 'Generated fake content for testing'
    debug_info['source'] = 'static'
    
    return {
        'title': book_title,
        'filename': book_filename,
        'grade': grade,
        'subject': subject,
        'chapters': book_content['chapters'],
        'total_chapters': len(book_content['chapters']),
        'source': 'static',
        'debug_info': debug_info
    }

def generate_book_structure(grade, subject, book_title):
    """Generate structured book content with chapters and exercises"""
    chapters = {}
    
    # Subject-specific chapter generation
    if subject == 'English':
        chapters = generate_english_chapters(grade)
    elif subject in ['Mathematics', 'Math']:
        chapters = generate_math_chapters(grade)
    elif subject == 'Urdu':
        chapters = generate_urdu_chapters(grade)
    elif subject == 'General Knowledge':
        chapters = generate_gk_chapters(grade)
    elif subject == 'General Science':
        chapters = generate_science_chapters(grade)
    elif subject == 'Social Studies':
        chapters = generate_sst_chapters(grade)
    elif subject == 'Islamiyat':
        chapters = generate_islamiyat_chapters(grade)
    else:
        # Default structure
        chapters = generate_default_chapters(grade, subject)
    
    return {'chapters': chapters}

def generate_english_chapters(grade):
    """Generate English chapter structure with categorized exercises"""
    
    # Special handler for Grade 4 with real textbook content
    if grade == 4:
        return generate_grade4_english_real_content()
    
    base_chapters = {
        1: {
            "Chapter 1: First Day at School": generate_exercises(["Meeting New Friends", "Classroom Rules", "School Places", "My Teacher and Me"]),
            "Chapter 2: Colors and Shapes": generate_exercises(["Red, Blue, Yellow", "Circle, Square, Triangle", "Big and Small", "Coloring and Drawing"]),
            "Chapter 3: Numbers and Counting": generate_exercises(["Numbers 1 to 10", "Counting Toys", "More and Less", "Number Songs"]),
            "Chapter 4: Family and Home": generate_exercises(["My Family Members", "Our House", "Things at Home", "Family Activities"]),
            "Chapter 5: Animals and Pets": generate_exercises(["Farm Animals", "Wild Animals", "Pet Care", "Animal Sounds"])
        },
        2: {
            "Chapter 1: Reading Stories": generate_exercises(["Simple Story Reading", "Story Characters", "Story Settings", "What Happens Next"]),
            "Chapter 2: My Community": generate_exercises(["People in Community", "Community Helpers", "Places in Community", "Community Rules"]),
            "Chapter 3: Seasons and Weather": generate_exercises(["Four Seasons", "Weather Changes", "Season Activities", "Weather Words"]),
            "Chapter 4: Food and Health": generate_exercises(["Healthy Foods", "Meal Times", "Good Habits", "Staying Clean"]),
            "Chapter 5: Transportation": generate_exercises(["Land Transport", "Water Transport", "Air Transport", "Safety Rules"])
        },
        3: {
            "Chapter 1: Comprehension": generate_exercises(["Reading for Understanding", "Main Ideas", "Supporting Details", "Story Sequence"]),
            "Chapter 2: Grammar Basics": generate_exercises(["Nouns and Verbs", "Adjectives", "Sentence Types", "Punctuation Marks"]),
            "Chapter 3: Creative Writing": generate_exercises(["Story Writing", "Descriptive Writing", "Letter Writing", "Diary Entries"]),
            "Chapter 4: Poetry and Rhymes": generate_exercises(["Reading Poems", "Rhyming Words", "Writing Simple Poems", "Action Songs"]),
            "Chapter 5: Speaking Skills": generate_exercises(["Show and Tell", "Conversations", "Presentations", "Drama Activities"])
        },
        4: {
            "Chapter 1: Literature": generate_exercises(["Story Analysis", "Character Development", "Plot and Setting", "Theme Understanding"]),
            "Chapter 2: Advanced Grammar": generate_exercises(["Parts of Speech", "Tenses", "Complex Sentences", "Paragraph Writing"]),
            "Chapter 3: Vocabulary Building": generate_exercises(["Word Meanings", "Synonyms and Antonyms", "Word Formation", "Context Clues"]),
            "Chapter 4: Reading Comprehension": generate_exercises(["Critical Reading", "Inference Skills", "Fact vs Opinion", "Reading Strategies"]),
            "Chapter 5: Communication Skills": generate_exercises(["Formal Speaking", "Debates", "Interviews", "Group Discussions"])
        },
        5: {
            "Chapter 1: Advanced Reading": generate_exercises(["Critical Reading", "Author's Purpose", "Fact vs Opinion", "Reading Between Lines"]),
            "Chapter 2: Essay Writing": generate_exercises(["Essay Structure", "Paragraph Development", "Argument Building", "Editing Skills"]),
            "Chapter 3: Literature Study": generate_exercises(["Poetry Analysis", "Story Elements", "Literary Devices", "Character Study"]),
            "Chapter 4: Research Skills": generate_exercises(["Information Gathering", "Source Evaluation", "Note Taking", "Report Writing"]),
            "Chapter 5: Presentation Skills": generate_exercises(["Public Speaking", "Visual Aids", "Persuasive Speaking", "Feedback Skills"])
        }
    }
    
    return base_chapters.get(grade, base_chapters[1])

def generate_grade4_english_real_content():
    """Generate real Grade 4 English content from the actual Pakistani curriculum textbook"""
    return {
        "Chapter 1: Pinky's Dental Dilemma": {
            "Reading": [
                {"title": "Journey through the text - Pinky has a bad tooth", "type": "reading_comprehension"},
                {"title": "Memory Lane - New words to know", "type": "vocabulary_reading"},
                {"title": "Character emotions and feelings analysis", "type": "literary_analysis"}
            ],
            "Writing": [
                {"title": "Activity: Sentence Creator", "type": "creative_writing"},
                {"title": "Express with Emotions - Draw and explain feelings", "type": "descriptive_writing"},
                {"title": "Write about a time you felt scared", "type": "personal_narrative"}
            ],
            "Grammar": [
                {"title": "Activity 1: Naming Words Adventure (Nouns)", "type": "noun_identification"},
                {"title": "Activity 2: Action Words Detective (Verbs)", "type": "verb_identification"},
                {"title": "Activity 3: Describe It (Adjectives)", "type": "adjective_practice"},
                {"title": "Activity 4: Regular vs. Irregular Nouns", "type": "noun_plurals"}
            ],
            "Vocabulary": [
                {"title": "New words: loaf, raisins, chewy, counter, spill", "type": "word_meanings"},
                {"title": "Activity 3: Arrange Adventures - Dictionary skills", "type": "dictionary_practice"},
                {"title": "Alphabetical ordering exercise", "type": "alphabetical_order"}
            ],
            "Comprehension": [
                {"title": "Activity 1: Discovery Quiz", "type": "reading_comprehension"},
                {"title": "Character analysis - How does Pinky feel?", "type": "character_analysis"},
                {"title": "Story sequence and plot understanding", "type": "story_analysis"}
            ],
            "Oral Communication": [
                {"title": "Activity 2: Sharing Strategies - Role-play with Pinky", "type": "role_play"},
                {"title": "Share and Sparkle - Express emotions discussion", "type": "group_discussion"},
                {"title": "Overcoming fears conversation practice", "type": "speaking_practice"}
            ]
        },
        "Chapter 2: Food and Friends!": {
            "Reading": [
                {"title": "Main story reading and comprehension", "type": "reading_comprehension"},
                {"title": "Food-related vocabulary in context", "type": "contextual_reading"},
                {"title": "Friendship themes analysis", "type": "thematic_reading"}
            ],
            "Writing": [
                {"title": "Write about your favorite food and friends", "type": "descriptive_writing"},
                {"title": "Create a menu with descriptions", "type": "functional_writing"},
                {"title": "Friendship story writing", "type": "narrative_writing"}
            ],
            "Grammar": [
                {"title": "Food-related nouns and their plurals", "type": "noun_practice"},
                {"title": "Cooking action words (verbs)", "type": "verb_practice"},
                {"title": "Describing food (adjectives)", "type": "adjective_practice"}
            ],
            "Vocabulary": [
                {"title": "Food and cooking vocabulary", "type": "thematic_vocabulary"},
                {"title": "Friendship-related words", "type": "social_vocabulary"},
                {"title": "Dictionary work with new terms", "type": "dictionary_skills"}
            ],
            "Comprehension": [
                {"title": "Story comprehension questions", "type": "reading_comprehension"},
                {"title": "Character relationships analysis", "type": "character_analysis"},
                {"title": "Cultural food practices discussion", "type": "cultural_analysis"}
            ],
            "Oral Communication": [
                {"title": "Discuss favorite foods with classmates", "type": "group_discussion"},
                {"title": "Recipe sharing activity", "type": "presentation"},
                {"title": "Role-play restaurant scenarios", "type": "role_play"}
            ]
        },
        "Chapter 3: Pinky and Jojo Write a Story": {
            "Reading": [
                {"title": "Story about story writing - meta narrative", "type": "reading_comprehension"},
                {"title": "Creative process understanding", "type": "process_reading"},
                {"title": "Character collaboration analysis", "type": "character_study"}
            ],
            "Writing": [
                {"title": "Collaborative story writing with partner", "type": "collaborative_writing"},
                {"title": "Create story outlines and plots", "type": "story_planning"},
                {"title": "Character development exercises", "type": "character_creation"}
            ],
            "Grammar": [
                {"title": "Story writing punctuation and dialogue", "type": "punctuation_practice"},
                {"title": "Past tense verbs in narratives", "type": "verb_tenses"},
                {"title": "Sentence variety in storytelling", "type": "sentence_structure"}
            ],
            "Vocabulary": [
                {"title": "Story writing vocabulary and terms", "type": "writing_vocabulary"},
                {"title": "Creative and imaginative words", "type": "descriptive_vocabulary"},
                {"title": "Literary terms for young writers", "type": "literary_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Understanding the writing process", "type": "process_comprehension"},
                {"title": "Story elements identification", "type": "literary_analysis"},
                {"title": "Creative collaboration benefits", "type": "analytical_thinking"}
            ],
            "Oral Communication": [
                {"title": "Share created stories with class", "type": "presentation"},
                {"title": "Discuss favorite story ideas", "type": "group_discussion"},
                {"title": "Story brainstorming sessions", "type": "collaborative_speaking"}
            ]
        },
        "Chapter 4: Heroes of History": {
            "Reading": [
                {"title": "Historical hero stories and biographies", "type": "biographical_reading"},
                {"title": "Pakistani historical figures", "type": "cultural_reading"},
                {"title": "Heroic qualities and characteristics", "type": "character_analysis"}
            ],
            "Writing": [
                {"title": "Write about your personal hero", "type": "descriptive_writing"},
                {"title": "Create hero biography summaries", "type": "biographical_writing"},
                {"title": "Heroic qualities essay", "type": "expository_writing"}
            ],
            "Grammar": [
                {"title": "Past tense for historical events", "type": "verb_tenses"},
                {"title": "Proper nouns for historical figures", "type": "capitalization"},
                {"title": "Descriptive language for hero qualities", "type": "adjective_usage"}
            ],
            "Vocabulary": [
                {"title": "Historical and heroic vocabulary", "type": "thematic_vocabulary"},
                {"title": "Pakistani history terms", "type": "cultural_vocabulary"},
                {"title": "Character trait descriptors", "type": "descriptive_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Historical timeline understanding", "type": "chronological_comprehension"},
                {"title": "Hero characteristics analysis", "type": "character_analysis"},
                {"title": "Cultural heritage appreciation", "type": "cultural_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Present historical hero research", "type": "presentation"},
                {"title": "Debate heroic qualities", "type": "debate"},
                {"title": "Interview a community hero role-play", "type": "role_play"}
            ]
        },
        "Chapter 5: Culture Craze with Pinky!": {
            "Reading": [
                {"title": "Pakistani cultural traditions and festivals", "type": "cultural_reading"},
                {"title": "Traditional clothing, food, and customs", "type": "informational_reading"},
                {"title": "Cultural diversity within Pakistan", "type": "comparative_reading"}
            ],
            "Writing": [
                {"title": "Describe your family's cultural traditions", "type": "cultural_writing"},
                {"title": "Festival celebration narratives", "type": "narrative_writing"},
                {"title": "Cultural comparison essays", "type": "comparative_writing"}
            ],
            "Grammar": [
                {"title": "Cultural terms and proper nouns", "type": "capitalization_practice"},
                {"title": "Present tense for traditions", "type": "verb_tenses"},
                {"title": "Cultural adjectives and descriptions", "type": "descriptive_grammar"}
            ],
            "Vocabulary": [
                {"title": "Pakistani cultural vocabulary", "type": "cultural_vocabulary"},
                {"title": "Festival and celebration terms", "type": "thematic_vocabulary"},
                {"title": "Traditional arts and crafts words", "type": "specialized_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Cultural practices understanding", "type": "cultural_comprehension"},
                {"title": "Tradition significance analysis", "type": "analytical_comprehension"},
                {"title": "Cultural diversity appreciation", "type": "social_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Share family cultural traditions", "type": "cultural_sharing"},
                {"title": "Cultural show-and-tell presentations", "type": "presentation"},
                {"title": "Traditional games and activities", "type": "interactive_communication"}
            ]
        },
        "Chapter 6: Tech Tales & Starry Sights": {
            "Reading": [
                {"title": "Technology in daily life stories", "type": "contemporary_reading"},
                {"title": "Space and astronomy information", "type": "scientific_reading"},
                {"title": "Future technology predictions", "type": "speculative_reading"}
            ],
            "Writing": [
                {"title": "Write about favorite technology", "type": "descriptive_writing"},
                {"title": "Create space adventure stories", "type": "science_fiction_writing"},
                {"title": "Technology invention descriptions", "type": "technical_writing"}
            ],
            "Grammar": [
                {"title": "Technology-related vocabulary and terms", "type": "technical_grammar"},
                {"title": "Future tense for predictions", "type": "verb_tenses"},
                {"title": "Scientific description language", "type": "specialized_grammar"}
            ],
            "Vocabulary": [
                {"title": "Technology and digital vocabulary", "type": "technical_vocabulary"},
                {"title": "Space and astronomy terms", "type": "scientific_vocabulary"},
                {"title": "Innovation and invention words", "type": "specialized_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Technology impact analysis", "type": "analytical_comprehension"},
                {"title": "Space facts and information", "type": "factual_comprehension"},
                {"title": "Future possibilities discussion", "type": "speculative_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Technology debate and discussion", "type": "debate"},
                {"title": "Space exploration presentations", "type": "scientific_presentation"},
                {"title": "Invention idea sharing", "type": "creative_communication"}
            ]
        },
        "Chapter 7: Pinky's Safety Squad!": {
            "Reading": [
                {"title": "Safety rules and guidelines", "type": "instructional_reading"},
                {"title": "Emergency procedures and protocols", "type": "safety_reading"},
                {"title": "Community safety helpers", "type": "informational_reading"}
            ],
            "Writing": [
                {"title": "Create safety rule posters", "type": "instructional_writing"},
                {"title": "Emergency action plan writing", "type": "procedural_writing"},
                {"title": "Safety story scenarios", "type": "narrative_writing"}
            ],
            "Grammar": [
                {"title": "Imperative sentences for instructions", "type": "command_structure"},
                {"title": "Safety vocabulary and terms", "type": "specialized_grammar"},
                {"title": "Warning and caution language", "type": "functional_grammar"}
            ],
            "Vocabulary": [
                {"title": "Safety and emergency vocabulary", "type": "safety_vocabulary"},
                {"title": "Community helpers terms", "type": "social_vocabulary"},
                {"title": "Warning and instruction words", "type": "functional_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Safety rule importance understanding", "type": "practical_comprehension"},
                {"title": "Emergency response comprehension", "type": "procedural_comprehension"},
                {"title": "Community safety awareness", "type": "social_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Safety rule discussions and sharing", "type": "instructional_communication"},
                {"title": "Emergency response role-play", "type": "safety_role_play"},
                {"title": "Community helper interviews", "type": "informational_speaking"}
            ]
        },
        "Chapter 8: Dream Town Builders!": {
            "Reading": [
                {"title": "Community planning and development", "type": "civic_reading"},
                {"title": "Architecture and building concepts", "type": "technical_reading"},
                {"title": "Urban planning stories", "type": "thematic_reading"}
            ],
            "Writing": [
                {"title": "Design your ideal community", "type": "creative_writing"},
                {"title": "Building description and planning", "type": "technical_writing"},
                {"title": "Community improvement proposals", "type": "persuasive_writing"}
            ],
            "Grammar": [
                {"title": "Descriptive language for buildings", "type": "descriptive_grammar"},
                {"title": "Spatial relationship words", "type": "prepositional_practice"},
                {"title": "Planning and future tense", "type": "verb_tenses"}
            ],
            "Vocabulary": [
                {"title": "Architecture and building vocabulary", "type": "technical_vocabulary"},
                {"title": "Community and civic terms", "type": "civic_vocabulary"},
                {"title": "Planning and development words", "type": "specialized_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Community planning understanding", "type": "civic_comprehension"},
                {"title": "Building and design concepts", "type": "technical_comprehension"},
                {"title": "Urban development analysis", "type": "analytical_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Present community design ideas", "type": "presentation"},
                {"title": "Debate community improvement plans", "type": "debate"},
                {"title": "Collaborative town planning", "type": "group_collaboration"}
            ]
        },
        "Chapter 9: Pinky's Personality Play": {
            "Reading": [
                {"title": "Character traits and personality", "type": "character_analysis"},
                {"title": "Emotions and feelings exploration", "type": "psychological_reading"},
                {"title": "Personal growth stories", "type": "developmental_reading"}
            ],
            "Writing": [
                {"title": "Describe your personality traits", "type": "self_reflective_writing"},
                {"title": "Character development exercises", "type": "character_writing"},
                {"title": "Personal growth narratives", "type": "reflective_narrative"}
            ],
            "Grammar": [
                {"title": "Personality-describing adjectives", "type": "descriptive_grammar"},
                {"title": "Emotion and feeling vocabulary", "type": "expressive_grammar"},
                {"title": "Self-description sentence patterns", "type": "personal_grammar"}
            ],
            "Vocabulary": [
                {"title": "Personality trait vocabulary", "type": "psychological_vocabulary"},
                {"title": "Emotions and feelings words", "type": "emotional_vocabulary"},
                {"title": "Character description terms", "type": "descriptive_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Personality development understanding", "type": "personal_comprehension"},
                {"title": "Character motivation analysis", "type": "character_analysis"},
                {"title": "Emotional intelligence development", "type": "social_emotional_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Share personality traits with peers", "type": "personal_sharing"},
                {"title": "Discuss character qualities", "type": "character_discussion"},
                {"title": "Personality-based role-playing", "type": "character_role_play"}
            ]
        },
        "Chapter 10: Wonders of the Wild!": {
            "Reading": [
                {"title": "Wildlife and nature stories", "type": "nature_reading"},
                {"title": "Animal habitats and behaviors", "type": "scientific_reading"},
                {"title": "Environmental conservation themes", "type": "environmental_reading"}
            ],
            "Writing": [
                {"title": "Animal fact files and descriptions", "type": "informational_writing"},
                {"title": "Nature adventure stories", "type": "adventure_writing"},
                {"title": "Environmental protection essays", "type": "persuasive_writing"}
            ],
            "Grammar": [
                {"title": "Animal names and classifications", "type": "scientific_grammar"},
                {"title": "Habitat description language", "type": "descriptive_grammar"},
                {"title": "Environmental action verbs", "type": "action_grammar"}
            ],
            "Vocabulary": [
                {"title": "Wildlife and animal vocabulary", "type": "scientific_vocabulary"},
                {"title": "Habitat and ecosystem terms", "type": "environmental_vocabulary"},
                {"title": "Conservation and protection words", "type": "advocacy_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Animal behavior understanding", "type": "scientific_comprehension"},
                {"title": "Ecosystem relationships analysis", "type": "environmental_comprehension"},
                {"title": "Conservation importance awareness", "type": "advocacy_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Animal research presentations", "type": "scientific_presentation"},
                {"title": "Environmental protection debates", "type": "advocacy_debate"},
                {"title": "Nature documentary discussions", "type": "analytical_discussion"}
            ]
        },
        "Chapter 11: Sands, Secrets, and Schooltime Surprises": {
            "Reading": [
                {"title": "Mystery and adventure stories", "type": "mystery_reading"},
                {"title": "School life and experiences", "type": "relatable_reading"},
                {"title": "Discovery and exploration themes", "type": "adventure_reading"}
            ],
            "Writing": [
                {"title": "Create mystery stories and puzzles", "type": "mystery_writing"},
                {"title": "School experience narratives", "type": "personal_narrative"},
                {"title": "Adventure tale writing", "type": "adventure_writing"}
            ],
            "Grammar": [
                {"title": "Mystery story structure and language", "type": "narrative_grammar"},
                {"title": "Question formation for mysteries", "type": "interrogative_grammar"},
                {"title": "Suspense-building sentence variety", "type": "stylistic_grammar"}
            ],
            "Vocabulary": [
                {"title": "Mystery and detective vocabulary", "type": "genre_vocabulary"},
                {"title": "School and education terms", "type": "academic_vocabulary"},
                {"title": "Adventure and exploration words", "type": "action_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Mystery plot and clue analysis", "type": "mystery_comprehension"},
                {"title": "School situation problem-solving", "type": "practical_comprehension"},
                {"title": "Adventure sequence understanding", "type": "narrative_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Share school experience stories", "type": "personal_storytelling"},
                {"title": "Mystery-solving group discussions", "type": "collaborative_problem_solving"},
                {"title": "Adventure planning conversations", "type": "planning_communication"}
            ]
        },
        "Chapter 12: Sharing is Caring": {
            "Reading": [
                {"title": "Stories about kindness and generosity", "type": "moral_reading"},
                {"title": "Community service and helping others", "type": "social_reading"},
                {"title": "Friendship and cooperation themes", "type": "social_emotional_reading"}
            ],
            "Writing": [
                {"title": "Write about acts of kindness", "type": "reflective_writing"},
                {"title": "Create sharing and caring stories", "type": "moral_narrative"},
                {"title": "Community service project proposals", "type": "persuasive_writing"}
            ],
            "Grammar": [
                {"title": "Kindness and helping verbs", "type": "action_grammar"},
                {"title": "Community service vocabulary", "type": "social_grammar"},
                {"title": "Positive emotion adjectives", "type": "emotional_grammar"}
            ],
            "Vocabulary": [
                {"title": "Kindness and caring vocabulary", "type": "moral_vocabulary"},
                {"title": "Community service terms", "type": "civic_vocabulary"},
                {"title": "Sharing and cooperation words", "type": "social_vocabulary"}
            ],
            "Comprehension": [
                {"title": "Moral lessons and values understanding", "type": "moral_comprehension"},
                {"title": "Community impact analysis", "type": "social_comprehension"},
                {"title": "Caring behavior consequences", "type": "cause_effect_comprehension"}
            ],
            "Oral Communication": [
                {"title": "Share kindness experiences", "type": "moral_sharing"},
                {"title": "Plan community service projects", "type": "project_planning"},
                {"title": "Discuss ways to help others", "type": "social_action_discussion"}
            ]
        }
    }

def generate_exercises(topics):
    """Generate categorized exercises for given topics"""
    exercises = {}
    categories = ['Reading', 'Writing', 'Oral Communication', 'Comprehension', 'Grammar', 'Vocabulary']
    
    for i, topic in enumerate(topics):
        # Distribute topics across categories
        category = categories[i % len(categories)]
        if category not in exercises:
            exercises[category] = []
        exercises[category].append({
            'title': topic,
            'type': category,
            'difficulty': 'basic' if len(topic) < 15 else 'intermediate'
        })
    
    return exercises

def generate_curriculum_data():
    """Generate structured curriculum data for grades 1-5"""
    return {
        "Grade 1": {
            "English": {
                "Lesson 1: First Day at School": [
                    "Meeting New Friends",
                    "Classroom Rules",
                    "School Places",
                    "My Teacher and Me"
                ],
                "Lesson 2: Colors and Shapes": [
                    "Red, Blue, Yellow",
                    "Circle, Square, Triangle",
                    "Big and Small",
                    "Coloring and Drawing"
                ],
                "Lesson 3: Numbers and Counting": [
                    "Numbers 1 to 10",
                    "Counting Toys",
                    "More and Less",
                    "Number Songs"
                ]
            },
            "Math": {
                "Unit 1: Numbers 1 to 9": [
                    "Recognizing Numbers",
                    "Writing Numbers",
                    "Counting Objects",
                    "Number Order"
                ],
                "Unit 2: Addition to 10": [
                    "Joining Groups",
                    "Addition Stories",
                    "Number Bonds",
                    "Adding with Pictures"
                ],
                "Unit 3: Subtraction from 10": [
                    "Taking Away",
                    "Subtraction Stories", 
                    "How Many Left?",
                    "Subtract with Pictures"
                ]
            },
            "Science": {
                "Topic 1: My Body": [
                    "Body Parts Names",
                    "What I Can Do",
                    "Keeping Clean",
                    "Healthy Food"
                ],
                "Topic 2: Around Me": [
                    "At Home",
                    "At School",
                    "In the Garden",
                    "On the Road"
                ],
                "Topic 3: Day and Night": [
                    "Morning Activities",
                    "Afternoon Fun",
                    "Evening Time",
                    "Night Sleep"
                ]
            },
            "Islamiyat": {
                "Chapter 1: Allah and His Creation": [
                    "Who is Allah?",
                    "Allah's Beautiful Names",
                    "Allah Created Everything",
                    "Thanking Allah"
                ],
                "Chapter 2: Prophet Muhammad (PBUH)": [
                    "Our Beloved Prophet",
                    "Prophet's Kindness",
                    "Following the Prophet",
                    "Prophet's Family"
                ],
                "Chapter 3: Basic Duas": [
                    "Bismillah",
                    "Alhamdulillah", 
                    "Assalamu Alaikum",
                    "Simple Prayers"
                ]
            }
        },
        "Grade 2": {
            "English": {
                "Chapter 1: Reading Stories": [
                    "Simple Story Reading",
                    "Story Characters",
                    "Story Settings",
                    "Story Events"
                ],
                "Chapter 2: Grammar Basics": [
                    "Nouns and Naming Words",
                    "Action Words (Verbs)",
                    "Describing Words",
                    "Simple Sentences"
                ],
                "Chapter 3: Writing Skills": [
                    "Letter Writing",
                    "Story Writing",
                    "Picture Description",
                    "Creative Writing"
                ]
            },
            "Math": {
                "Chapter 1: Numbers 1-100": [
                    "Two-Digit Numbers",
                    "Place Value (Tens and Ones)",
                    "Number Patterns",
                    "Greater Than, Less Than"
                ],
                "Chapter 2: Addition and Subtraction": [
                    "Two-Digit Addition",
                    "Two-Digit Subtraction",
                    "Word Problems",
                    "Mental Math"
                ],
                "Chapter 3: Time and Money": [
                    "Reading Clock",
                    "Time Concepts",
                    "Coins and Currency",
                    "Making Change"
                ]
            },
            "Science": {
                "Chapter 1: Human Body": [
                    "Body Parts",
                    "Five Senses",
                    "Healthy Habits",
                    "Body Functions"
                ],
                "Chapter 2: Water": [
                    "Sources of Water",
                    "Uses of Water",
                    "Clean and Dirty Water",
                    "Saving Water"
                ],
                "Chapter 3: Materials": [
                    "Natural Materials",
                    "Man-made Materials",
                    "Properties of Materials",
                    "Uses of Materials"
                ]
            },
            "Islamiyat": {
                "Chapter 1: Five Pillars of Islam": [
                    "Kalima (Declaration of Faith)",
                    "Salah (Prayer)",
                    "Zakat (Charity)",
                    "Hajj and Fasting"
                ],
                "Chapter 2: Good Manners": [
                    "Respect for Parents",
                    "Kindness to Others",
                    "Truthfulness",
                    "Islamic Greetings"
                ],
                "Chapter 3: Islamic Festivals": [
                    "Eid ul Fitr",
                    "Eid ul Adha",
                    "Islamic Calendar",
                    "Festival Celebrations"
                ]
            }
        },
        "Grade 3": {
            "English": {
                "Chapter 1: Comprehension": [
                    "Reading for Understanding",
                    "Main Ideas",
                    "Supporting Details",
                    "Making Inferences"
                ],
                "Chapter 2: Grammar": [
                    "Parts of Speech",
                    "Sentence Types",
                    "Punctuation",
                    "Capitalization"
                ],
                "Chapter 3: Creative Writing": [
                    "Paragraph Writing",
                    "Descriptive Writing",
                    "Story Elements",
                    "Poetry Basics"
                ]
            },
            "Math": {
                "Chapter 1: Large Numbers": [
                    "Numbers up to 1000",
                    "Place Value (Hundreds)",
                    "Number Comparison",
                    "Rounding Numbers"
                ],
                "Chapter 2: Multiplication": [
                    "Multiplication Concept",
                    "Times Tables 2-10",
                    "Multiplication Problems",
                    "Arrays and Groups"
                ],
                "Chapter 3: Measurement": [
                    "Length and Distance",
                    "Weight and Mass",
                    "Capacity and Volume",
                    "Units of Measurement"
                ]
            },
            "Science": {
                "Chapter 1: Animals and Habitats": [
                    "Animal Classifications",
                    "Animal Habitats",
                    "Animal Adaptations",
                    "Food Chains"
                ],
                "Chapter 2: States of Matter": [
                    "Solids, Liquids, Gases",
                    "Properties of Matter",
                    "Changing States",
                    "Examples in Daily Life"
                ],
                "Chapter 3: Earth and Space": [
                    "Day and Night",
                    "Seasons",
                    "Moon Phases",
                    "Solar System Basics"
                ]
            },
            "Islamiyat": {
                "Chapter 1: Quran and Sunnah": [
                    "What is the Quran?",
                    "What is Sunnah?",
                    "Following Quran and Sunnah",
                    "Benefits of Reading Quran"
                ],
                "Chapter 2: Prophets of Allah": [
                    "Prophet Adam (AS)",
                    "Prophet Noah (AS)",
                    "Prophet Ibrahim (AS)",
                    "Lessons from Prophets"
                ],
                "Chapter 3: Islamic Values": [
                    "Honesty and Truth",
                    "Patience and Perseverance", 
                    "Forgiveness",
                    "Helping Others"
                ]
            }
        },
        "Grade 4": {
            "English": {
                "Chapter 1: Literature": [
                    "Story Analysis",
                    "Character Development",
                    "Plot and Setting",
                    "Theme Identification"
                ],
                "Chapter 2: Advanced Grammar": [
                    "Tenses (Past, Present, Future)",
                    "Subject-Verb Agreement",
                    "Conjunctions",
                    "Complex Sentences"
                ],
                "Chapter 3: Research Skills": [
                    "Information Gathering",
                    "Note Taking",
                    "Report Writing",
                    "Presentation Skills"
                ]
            },
            "Math": {
                "Chapter 1: Division": [
                    "Division Concept",
                    "Long Division",
                    "Division with Remainders",
                    "Division Word Problems"
                ],
                "Chapter 2: Fractions": [
                    "Understanding Fractions",
                    "Equivalent Fractions",
                    "Adding Fractions",
                    "Comparing Fractions"
                ],
                "Chapter 3: Geometry": [
                    "2D and 3D Shapes",
                    "Angles and Lines",
                    "Symmetry",
                    "Area and Perimeter"
                ]
            },
            "Science": {
                "Chapter 1: Energy": [
                    "Forms of Energy",
                    "Energy Sources",
                    "Energy Transfer",
                    "Conservation of Energy"
                ],
                "Chapter 2: Plants and Photosynthesis": [
                    "Plant Structure",
                    "Photosynthesis Process",
                    "Plant Reproduction",
                    "Plant Adaptations"
                ],
                "Chapter 3: Simple Machines": [
                    "Lever and Fulcrum",
                    "Wheel and Axle",
                    "Inclined Plane",
                    "Machines in Daily Life"
                ]
            },
            "Islamiyat": {
                "Chapter 1: Worship in Islam": [
                    "Importance of Salah",
                    "Wudu (Ablution)",
                    "Times of Prayer",
                    "Masjid and Community"
                ],
                "Chapter 2: Islamic History": [
                    "Life in Makkah",
                    "Hijra to Madinah",
                    "Early Muslim Community",
                    "Lessons from History"
                ],
                "Chapter 3: Character Building": [
                    "Responsibility",
                    "Leadership Qualities",
                    "Justice and Fairness",
                    "Gratitude to Allah"
                ]
            }
        },
        "Grade 5": {
            "English": {
                "Chapter 1: Advanced Reading": [
                    "Critical Reading",
                    "Author's Purpose",
                    "Fact vs Opinion",
                    "Reading Strategies"
                ],
                "Chapter 2: Writing Mastery": [
                    "Essay Writing",
                    "Persuasive Writing", 
                    "Research Papers",
                    "Editing and Revision"
                ],
                "Chapter 3: Speaking and Listening": [
                    "Public Speaking",
                    "Debate Skills",
                    "Active Listening",
                    "Group Discussions"
                ]
            },
            "Math": {
                "Chapter 1: Decimals": [
                    "Understanding Decimals",
                    "Decimal Operations",
                    "Decimal and Fraction Relationship",
                    "Real-world Decimal Applications"
                ],
                "Chapter 2: Data and Statistics": [
                    "Collecting Data",
                    "Graphs and Charts",
                    "Mean, Median, Mode",
                    "Interpreting Data"
                ],
                "Chapter 3: Problem Solving": [
                    "Multi-step Problems",
                    "Logical Reasoning",
                    "Pattern Recognition",
                    "Mathematical Thinking"
                ]
            },
            "Science": {
                "Chapter 1: Human Systems": [
                    "Circulatory System",
                    "Respiratory System",
                    "Digestive System",
                    "Nervous System"
                ],
                "Chapter 2: Ecosystems": [
                    "Food Webs",
                    "Environmental Balance",
                    "Biodiversity",
                    "Conservation"
                ],
                "Chapter 3: Chemical Changes": [
                    "Physical vs Chemical Changes",
                    "Acids and Bases",
                    "Chemical Reactions",
                    "Safety in Science"
                ]
            },
            "Islamiyat": {
                "Chapter 1: Islamic Civilization": [
                    "Golden Age of Islam",
                    "Islamic Contributions to Science",
                    "Islamic Architecture",
                    "Scholars and Learning"
                ],
                "Chapter 2: Social Justice": [
                    "Rights and Responsibilities",
                    "Helping the Needy",
                    "Environmental Care",
                    "Community Service"
                ],
                "Chapter 3: Spiritual Development": [
                    "Dhikr and Remembrance",
                    "Self-reflection",
                    "Seeking Knowledge",
                    "Preparation for Adulthood"
                ]
            }
        }
    }

def generate_assessment_response(assessment_type):
    """Generate assessment content for chatbot responses"""
    if assessment_type == 'qna':
        return jsonify({
            'message': '''❓ **Quick Q&A Assessment Questions:**

**Question 1:** What is the main idea of today's lesson?
💡 *Look for: Accept answers that demonstrate understanding of the key concept*

**Question 2:** Can you give me an example of what we learned?
💡 *Look for: Real-world applications or connections*

**Question 3:** What was the most interesting part of the lesson?
💡 *Look for: Helps gauge engagement and memorable moments*

**Question 4:** Is there anything you'd like to know more about?
💡 *Look for: Identifies areas for follow-up or extension*

**Question 5:** How would you explain this to a friend?
💡 *Look for: Tests ability to communicate understanding clearly*

📋 **Instructions:** Use these questions to quickly assess student understanding. Mix and match based on your lesson!''',
            'options': ['📊 More Assessment Types', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'mcq':
        return jsonify({
            'message': '''🔤 **Multiple Choice Questions (MCQ):**

**Question 1:** What is the process by which plants make their own food?
A) Respiration  B) Photosynthesis  C) Digestion  D) Circulation
✅ *Answer: B) Photosynthesis - Plants use sunlight, water, and carbon dioxide to make food*

**Question 2:** Which planet is closest to the Sun?
A) Venus  B) Earth  C) Mercury  D) Mars
✅ *Answer: C) Mercury - Mercury is the smallest planet and closest to the Sun*

**Question 3:** What is the main source of energy for Earth?
A) The Moon  B) The Sun  C) Wind  D) Water
✅ *Answer: B) The Sun - The Sun provides light and heat energy for Earth*

**Question 4:** How many continents are there on Earth?
A) 5  B) 6  C) 7  D) 8
✅ *Answer: C) 7 - The seven continents are Asia, Africa, North America, South America, Antarctica, Europe, and Australia*

📋 **Instructions:** Read each question and ask students to choose the correct answer. Perfect for testing specific knowledge!''',
            'options': ['📊 More Assessment Types', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'comprehension':
        return jsonify({
            'message': '''📖 **Short Comprehension Questions:**

**Passage 1:** *"The butterfly starts its life as a tiny egg. Then it becomes a caterpillar that eats lots of leaves. Next, it forms a chrysalis around itself. Finally, it emerges as a beautiful butterfly."*

**Questions:**
1. What does the caterpillar eat? 💡 *Expected Answer: Leaves*
2. What forms around the caterpillar? 💡 *Expected Answer: A chrysalis*
3. What are the four stages mentioned? 💡 *Expected Answer: Egg, caterpillar, chrysalis, butterfly*

**Passage 2:** *"Rain is very important for our planet. It waters the plants and fills the rivers and lakes. When the sun heats up water, it turns into vapor and goes up into the sky. In the clouds, the vapor turns back into water drops that fall as rain."*

**Questions:**
1. Why is rain important? 💡 *Expected Answer: It waters plants and fills rivers and lakes*
2. What happens when the sun heats water? 💡 *Expected Answer: It turns into vapor and goes up into the sky*
3. Where does vapor turn back into water drops? 💡 *Expected Answer: In the clouds*

📋 **Instructions:** Read the passage aloud, then ask the comprehension questions. Great for reading and understanding skills!''',
            'options': ['📊 More Assessment Types', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'fill-blanks':
        return jsonify({
            'message': '''✏️ **Fill in the Blanks:**

**Question 1:** The _____ is the center of our solar system.
💡 *Hint: It gives us light and heat*
✅ *Answer: Sun*

**Question 2:** Plants need _____, water, and carbon dioxide to make food.
💡 *Hint: Something that comes from the sun*
✅ *Answer: sunlight/light*

**Question 3:** The _____ is the largest ocean on Earth.
💡 *Hint: It's between Asia and America*
✅ *Answer: Pacific*

**Question 4:** A _____ has three sides and three corners.
💡 *Hint: It's a shape*
✅ *Answer: triangle*

**Question 5:** We use our _____ to breathe air into our body.
💡 *Hint: They're inside your chest*
✅ *Answer: lungs*

📋 **Instructions:** Read each sentence and have students fill in the missing word. Give hints if needed!''',
            'options': ['📊 More Assessment Types', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'thumbs':
        return jsonify({
            'message': '''👍👎 **Thumbs Up/Down Assessment:**

**Statement 1:** "I understand today's main concept"
👍 *Thumbs Up = Agree | 👎 Thumbs Down = Disagree*

**Statement 2:** "I can explain this to someone else"
👍 *Thumbs Up = Agree | 👎 Thumbs Down = Disagree*

**Statement 3:** "I feel confident about this topic"
👍 *Thumbs Up = Agree | 👎 Thumbs Down = Disagree*

**Statement 4:** "I need more practice with this"
👍 *Thumbs Up = Agree | 👎 Thumbs Down = Disagree*

**Statement 5:** "I found today's lesson interesting"
👍 *Thumbs Up = Agree | 👎 Thumbs Down = Disagree*

**Statement 6:** "I can see how this connects to real life"
👍 *Thumbs Up = Agree | 👎 Thumbs Down = Disagree*

📋 **Instructions:** Read each statement and have students show thumbs up (agree) or thumbs down (disagree). Great for quick class pulse checks!''',
            'options': ['📊 More Assessment Types', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'statements':
        return jsonify({
            'message': '''📝 **True/False Statements:**

**Statement 1:** "Plants need sunlight to make their own food"
✅ *Answer: TRUE - Photosynthesis requires sunlight*

**Statement 2:** "All insects have 8 legs"
✅ *Answer: FALSE - Insects have 6 legs, spiders have 8*

**Statement 3:** "Water freezes at 0 degrees Celsius"
✅ *Answer: TRUE - This is the freezing point of water*

**Statement 4:** "The Earth is flat"
✅ *Answer: FALSE - The Earth is round/spherical*

**Statement 5:** "Reading helps improve vocabulary"
✅ *Answer: TRUE - Exposure to new words through reading expands vocabulary*

📋 **Instructions:** Read each statement and have students write T (True) or F (False). Perfect for science and general knowledge!''',
            'options': ['📊 More Assessment Types', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'exit-ticket':
        return jsonify({
            'message': '''🎫 **Exit Ticket Prompts:**

**Prompt 1:** "Today I learned..."
🎯 *Purpose: Identifies key takeaways*

**Prompt 2:** "I'm still wondering about..."
🎯 *Purpose: Reveals areas of confusion*

**Prompt 3:** "One thing I want to remember is..."
🎯 *Purpose: Highlights most important learning*

**Prompt 4:** "I can use this when..."
🎯 *Purpose: Shows real-world connections*

**Prompt 5:** "My favorite part was..."
🎯 *Purpose: Gauges engagement and interest*

📋 **Instructions:** Choose 2-3 prompts for students to complete before leaving class. Perfect for reflection and feedback!''',
            'options': ['📊 More Assessment Types', '← Back to Menu'],
            'show_menu': True
        })
    
    # Default fallback
    return jsonify({
        'message': '📊 Assessment feature is being prepared! Please try again or choose from the main menu.',
        'options': ['📊 Assessment', '← Back to Menu'],
        'show_menu': True
    })

def generate_curriculum_lesson_plan(grade, subject, chapter, topic):
    """Generate lesson plan for specific curriculum topic"""
    return jsonify({
        'message': f'''📝 **Lesson Plan Generated**

**Grade:** {grade}
**Subject:** {subject}  
**Chapter:** {chapter}
**Topic:** {topic}

## 🎯 **Learning Objectives:**
• Students will understand the key concepts of {topic}
• Students will be able to explain {topic} in their own words
• Students will apply knowledge of {topic} to real-world examples

## 📚 **Materials Needed:**
• Whiteboard/markers
• Student notebooks
• Visual aids/pictures
• Worksheets
• Interactive materials

## ⏰ **Lesson Duration:** 40 minutes

## 📋 **Lesson Structure:**

**Introduction (5 minutes):**
• Warm-up activity related to {topic}
• Ask students what they already know
• Introduce today's learning goal

**Main Teaching (25 minutes):**
• Explain {topic} with clear examples
• Use visual aids and interactive demonstrations
• Ask questions to check understanding
• Provide hands-on activities

**Practice (7 minutes):**
• Quick exercises for students to apply learning
• Pair/group work activities
• Individual practice time

**Wrap-up (3 minutes):**
• Summarize key points
• Ask students to share one thing they learned
• Preview next lesson

## 📊 **Assessment:**
• Observe student participation
• Check understanding through questions
• Review completed practice exercises

## 🏠 **Homework/Extension:**
• Simple practice worksheet
• Real-world observation activity
• Prepare for next lesson''',
        'options': [
            '📊 Create Assessment for this Topic',
            '🎮 Suggest Fun Activities', 
            '💡 Get Teaching Tips',
            '🔄 Choose Different Topic',
            '← Back to Menu'
        ],
        'show_menu': True
    })

def generate_curriculum_assessment_types(grade, subject, chapter, topic):
    """Show assessment type options for specific curriculum topic"""
    return jsonify({
        'message': f'''📊 **Assessment Types for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

Choose your assessment type:''',
        'options': [
            '❓ Quick Q&A',
            '🔤 Multiple Choice Questions (MCQ)',
            '📖 Short Comprehension Questions', 
            '👍👎 Thumbs Up/Down',
            '📝 True/False Statements',
            '✏️ Fill in the Blanks',
            '🎫 Exit Tickets',
            '🔄 Choose Different Topic',
            '← Back to Menu'
        ],
        'show_menu': True
    })

def generate_curriculum_assessment(grade, subject, chapter, topic):
    """Generate assessment questions for specific curriculum topic"""
    return jsonify({
        'message': f'''📊 **Assessment Questions Generated**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}  
**Topic:** {topic}

## ❓ **Quick Q&A Questions:**
1. What is {topic}? Explain in your own words.
2. Can you give an example of {topic}?
3. Why is {topic} important?
4. How does {topic} relate to what we learned before?

## 🔤 **Multiple Choice Questions:**
**Question 1:** Which of the following best describes {topic}?
A) Option A   B) Option B   C) Option C   D) Option D

**Question 2:** {topic} is most commonly found in:
A) Option A   B) Option B   C) Option C   D) Option D

## 📝 **True/False Statements:**
1. {topic} is an important concept in {subject}. (True/False)
2. Students should understand {topic} at {grade} level. (True/False)

## ✏️ **Fill in the Blanks:**
1. {topic} is related to _______ and _______.
2. The main idea of {topic} is _______.

## 👍👎 **Quick Assessment:**
Have students show thumbs up/down for:
- "I understand {topic}"
- "I can explain {topic} to someone else"
- "I need more practice with {topic}"

## 🎫 **Exit Ticket:**
Before leaving class, students complete:
"Today I learned that {topic} is..."
"One question I still have about {topic} is..."''',
        'options': [
            '📝 Generate Lesson Plan',
            '🎮 Suggest Fun Activities',
            '💡 Get Teaching Tips', 
            '🔄 Choose Different Topic',
            '← Back to Menu'
        ],
        'show_menu': True
    })

def generate_curriculum_activities(grade, subject, chapter, topic):
    """Generate fun activities for specific curriculum topic"""
    return jsonify({
        'message': f'''🎮 **Fun Activities for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

## 🎨 **Creative Activities:**

**Activity 1: {topic} Art Project**
• Students create drawings/posters about {topic}
• Use colors, symbols, and words to represent key concepts
• Display student work around the classroom

**Activity 2: {topic} Story Time**
• Students write short stories incorporating {topic}
• Share stories with the class
• Vote on most creative story

**Activity 3: {topic} Drama/Role Play**
• Students act out scenarios related to {topic}
• Use props and costumes
• Perform for other classes

## 🎯 **Interactive Games:**

**Game 1: {topic} Bingo**
• Create bingo cards with {topic}-related terms
• Call out definitions, students mark answers
• First to complete a line wins

**Game 2: {topic} Memory Match**
• Cards with {topic} terms and definitions
• Students match pairs
• Can be played individually or in groups

**Game 3: {topic} Scavenger Hunt**
• Hide clues around classroom/school
• Each clue teaches something about {topic}
• Teams work together to solve puzzles

## 🔬 **Hands-On Experiments:**

**Experiment 1: {topic} Investigation**
• Simple, safe experiment related to {topic}
• Students observe and record results
• Discuss findings as a class

**Experiment 2: {topic} Building Challenge**
• Use everyday materials to demonstrate {topic}
• Students work in teams
• Present creations to class

## 🎪 **Movement Activities:**

**Activity 1: {topic} Actions**
• Create movements that represent {topic}
• Students perform actions while learning
• Great for kinesthetic learners

**Activity 2: {topic} Dance/Song**
• Make up a simple song about {topic}
• Include hand motions and rhythm
• Perform for other classes''',
        'options': [
            '📝 Generate Lesson Plan',
            '📊 Create Assessment Questions',
            '💡 Get Teaching Tips',
            '🔄 Choose Different Topic', 
            '← Back to Menu'
        ],
        'show_menu': True
    })

def generate_curriculum_specific_assessment(assessment_type, grade, subject, chapter, topic):
    """Generate curriculum-specific assessments"""
    if assessment_type == 'qna':
        return jsonify({
            'message': f'''❓ **Quick Q&A for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

**Question 1:** What is {topic}? Explain in your own words.

**Question 2:** Can you give an example of {topic} from your daily life?

**Question 3:** Why is learning about {topic} important for {grade} students?

**Question 4:** How does {topic} connect to what we learned in previous lessons?

**Question 5:** What is the most interesting thing about {topic}?

📋 **Instructions:** Ask these questions one at a time and encourage students to explain their thinking. Great for checking understanding!''',
            'options': ['🔄 Try Different Assessment Type', '🔄 Choose Different Topic', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'mcq':
        return jsonify({
            'message': f'''🔤 **Multiple Choice Questions for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

**Question 1:** Which best describes {topic}?
A) Option related to basic concept
B) Option with correct answer about {topic}
C) Incorrect but plausible option
D) Another incorrect option
✅ *Answer: B*

**Question 2:** {topic} is most important because:
A) It helps with {subject} learning
B) Students need to understand it for {grade}
C) It connects to real life
D) All of the above
✅ *Answer: D*

**Question 3:** When learning about {topic}, students should focus on:
A) Memorizing facts only
B) Understanding concepts and examples
C) Just reading about it
D) Ignoring practical applications
✅ *Answer: B*

📋 **Instructions:** Read each question and have students choose the correct answer. Discuss why other options are incorrect!''',
            'options': ['🔄 Try Different Assessment Type', '🔄 Choose Different Topic', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'comprehension':
        return jsonify({
            'message': f'''📖 **Comprehension Questions for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

**Passage:** *"{topic} is an important concept that {grade} students learn in {subject}. Understanding {topic} helps students develop better knowledge and skills. When students learn about {topic}, they can apply this knowledge in many different situations and connect it to their daily experiences."*

**Questions:**
1. What subject do students learn {topic} in? 💡 *Expected Answer: {subject}*
2. Who learns about {topic}? 💡 *Expected Answer: {grade} students*
3. How can students use knowledge about {topic}? 💡 *Expected Answer: Apply it in different situations and connect to daily life*
4. Why is {topic} important for students? 💡 *Expected Answer: Helps develop better knowledge and skills*

📋 **Instructions:** Read the passage aloud, then ask the comprehension questions. Perfect for reading and understanding skills!''',
            'options': ['🔄 Try Different Assessment Type', '🔄 Choose Different Topic', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'fill-blanks':
        return jsonify({
            'message': f'''✏️ **Fill in the Blanks for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

**Question 1:** In {subject}, we learn that {topic} is _______.
💡 *Hint: What is the main concept?*
✅ *Answer: [key concept about the topic]*

**Question 2:** {grade} students should understand {topic} because it helps them _______.
💡 *Hint: Think about the benefits*
✅ *Answer: learn better/understand concepts/apply knowledge*

**Question 3:** When we study {topic}, we can see examples in _______.
💡 *Hint: Where do we find this in real life?*
✅ *Answer: daily life/real world/our environment*

**Question 4:** The most important thing about {topic} is _______.
💡 *Hint: What's the key takeaway?*
✅ *Answer: [main learning objective]*

📋 **Instructions:** Read each sentence and have students fill in the missing word. Give hints if needed!''',
            'options': ['🔄 Try Different Assessment Type', '🔄 Choose Different Topic', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'thumbs':
        return jsonify({
            'message': f'''👍👎 **Quick Check for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

**Statement 1:** "I understand what {topic} means"
👍 *Thumbs Up = I understand | 👎 Thumbs Down = I need help*

**Statement 2:** "I can give an example of {topic}"
👍 *Thumbs Up = I can | 👎 Thumbs Down = I'm not sure*

**Statement 3:** "I know why {topic} is important in {subject}"
👍 *Thumbs Up = I know why | 👎 Thumbs Down = I don't know*

**Statement 4:** "I feel confident about {topic}"
👍 *Thumbs Up = Very confident | 👎 Thumbs Down = Need more practice*

**Statement 5:** "I can connect {topic} to real life"
👍 *Thumbs Up = Yes, I can | 👎 Thumbs Down = Not really*

📋 **Instructions:** Read each statement and have students show thumbs up or down. Great for quick understanding checks!''',
            'options': ['🔄 Try Different Assessment Type', '🔄 Choose Different Topic', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'statements':
        return jsonify({
            'message': f'''📝 **True/False for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

**Statement 1:** {topic} is an important concept in {subject}. 
✅ *Answer: TRUE - It's part of the {grade} curriculum*

**Statement 2:** Only {grade} students need to learn about {topic}.
❌ *Answer: FALSE - Other grades may also learn this concept*

**Statement 3:** {topic} can be found in everyday life.
✅ *Answer: TRUE - Many concepts have real-world applications*

**Statement 4:** Understanding {topic} helps with other {subject} topics.
✅ *Answer: TRUE - Learning builds on previous knowledge*

**Statement 5:** {topic} is too difficult for {grade} students.
❌ *Answer: FALSE - It's designed for this grade level*

📋 **Instructions:** Read each statement and have students decide if it's true or false. Discuss the reasoning!''',
            'options': ['🔄 Try Different Assessment Type', '🔄 Choose Different Topic', '← Back to Menu'],
            'show_menu': True
        })
    
    elif assessment_type == 'exit-ticket':
        return jsonify({
            'message': f'''🎫 **Exit Ticket for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

**Before you leave today, please complete:**

**1. One thing I learned about {topic} today:**
_________________________________

**2. One question I still have about {topic}:**
_________________________________

**3. How I can use {topic} outside of school:**
_________________________________

**4. My confidence level with {topic} (1-5 stars):**
⭐ ⭐ ⭐ ⭐ ⭐

**5. I need more help with:**
□ Understanding the concept
□ Finding examples
□ Connecting to real life
□ Nothing - I feel confident!

📋 **Instructions:** Have students complete this before leaving class. Great for assessing learning and planning next steps!''',
            'options': ['🔄 Try Different Assessment Type', '🔄 Choose Different Topic', '← Back to Menu'],
            'show_menu': True
        })
    
    # Fallback
    return generate_assessment_response(assessment_type)

def generate_curriculum_tips(grade, subject, chapter, topic):
    """Generate teaching tips for specific curriculum topic"""
    return jsonify({
        'message': f'''💡 **Teaching Tips for {topic}**

**Grade:** {grade}
**Subject:** {subject}
**Chapter:** {chapter}
**Topic:** {topic}

## 🎯 **Before Teaching:**

**Preparation Tips:**
• Review {topic} concepts thoroughly yourself
• Gather visual aids, examples, and materials
• Plan for different learning styles (visual, auditory, kinesthetic)
• Prepare simple analogies students can relate to

**Know Your Students:**
• Assess prior knowledge about {topic}
• Consider students' attention spans ({grade} level)
• Plan for different ability levels in your class
• Have extra activities ready for fast finishers

## 🚀 **During Teaching:**

**Engagement Strategies:**
• Start with a question or surprising fact about {topic}
• Use real-world examples students can connect to
• Encourage student questions and discussions
• Break content into small, manageable chunks

**Clear Communication:**
• Use simple, age-appropriate language
• Repeat key concepts multiple times
• Check for understanding frequently ("Show me thumbs up if...")
• Use visual aids and gestures to support explanations

## 🔄 **Making It Stick:**

**Reinforcement Techniques:**
• Connect {topic} to previous learning
• Use storytelling to make concepts memorable
• Provide multiple practice opportunities
• Celebrate student success and progress

**Assessment Strategies:**
• Use quick formative assessments during lesson
• Observe student work and participation
• Ask students to explain concepts back to you
• Use peer teaching opportunities

## 🌟 **Differentiation Ideas:**

**For Advanced Students:**
• Provide extension questions about {topic}
• Let them help teach other students
• Give additional research projects
• Connect to more complex concepts

**For Struggling Students:**
• Break {topic} into smaller steps
• Provide additional visual supports
• Use peer buddies for support
• Give extra practice time

**For English Language Learners:**
• Use visual aids and gestures
• Provide key vocabulary beforehand
• Allow native language discussion
• Use translation tools when needed

## 💭 **Common Challenges:**

**If Students Seem Confused:**
• Slow down and re-explain using different words
• Use more concrete examples
• Ask students what specifically confuses them
• Try a different teaching approach

**If Students Seem Bored:**
• Add more interactive elements
• Connect to current events or popular culture
• Use humor appropriately
• Let students share their own examples''',
        'options': [
            '📝 Generate Lesson Plan',
            '📊 Create Assessment Questions', 
            '🎮 Suggest Fun Activities',
            '🔄 Choose Different Topic',
            '← Back to Menu'
        ],
        'show_menu': True
    })

def get_pakistani_teacher_fallback(user_message, session_context=None):
    """Robust fallback responses for Pakistani teachers when AI services fail"""
    
    # Get session data for context-aware responses
    grade = session_context.get('grade', 1) if session_context else 1
    subject = session_context.get('subject', 'English') if session_context else 'English'
    activity_type = session_context.get('activity_type', 'activities') if session_context else 'activities'
    selected_feature = session_context.get('selected_feature', 'lesson_plans') if session_context else 'lesson_plans'
    
    # Convert grade to number if it's a string
    try:
        if isinstance(grade, str):
            import re
            grade_match = re.search(r'(\d+)', str(grade))
            grade = int(grade_match.group(1)) if grade_match else 1
    except:
        grade = 1
    
    topic = user_message.lower().strip()
    
    # Debug information (logged, not shown to user)
    print(f"Fallback triggered - Grade: {grade}, Subject: {subject}, Topic: {topic}, Activity: {activity_type}")
    
    # Specific hardcoded responses for common topics with grade-based complexity
    if 'noun' in topic:
        if activity_type == 'pair_work':
            if grade == 1:
                return f"""👫 PAIR WORK - Nouns (Grade 1 English)

Hello teacher! Grade 1 ke liye simple activities:

🌟 Point and Say:
Ahmed chair dikhaega, Fatima 'chair' bolegi
Phir Fatima table dikhaegi, Ahmed 'table' bolega
Bilkul simple!

🌟 Show Me Game:
"Show me book" - partner book dikhaega
"Show me bag" - partner bag dikhaega
One word nouns only

🌟 Touch and Tell:
Partners classroom mein cheez touch karenge
"This is..." kehte jaenge
Door, window, floor - basic words

Grade 1 ke liye bilkul perfect level! Try kariye teacher! 😊"""

            elif grade == 3:
                return f"""👫 PAIR WORK - Nouns (Grade 3 English)

Hello teacher! Grade 3 ke liye yeh activities:

🌟 Noun Hunt:
Partners classroom mein common aur proper nouns dhundenge
Ahmed (proper), chair (common) - difference samjhayen

🌟 Person Place Thing Sort:
Cards banayiye: Ahmed (person), school (place), book (thing)
Partners turns lete kar categories batayenge

🌟 My Family Nouns:
"My Abbu is a teacher" - family words practice
Partners family members ke bare mein sentences banayenge

Grade 3 ke liye perfect complexity! Kaisa laga teacher? 😊"""

            elif grade >= 4:
                return f"""👫 PAIR WORK - Nouns (Grade {grade} English)

Hello teacher! Grade {grade} ke liye challenging activities:

🌟 Noun Analysis:
Partners sentences mein subject aur object nouns identify karenge
"Ahmed reads books" - Ahmed (subject), books (object)

🌟 Noun Functions:
Sentence building with different noun roles
"The teacher gave Ahmed a book" - multiple noun analysis

🌟 Abstract vs Concrete:
Advanced categorization: happiness (abstract), table (concrete)
Partners examples discuss karenge

Grade {grade} ke liye challenging level! Kaisa laga teacher? 😊"""

            else:
                return f"""👫 PAIR WORK - Nouns (Grade {grade} {subject})

Hello teacher! Grade {grade} ke bachon ke liye nouns activities:

🌟 Classroom Objects:
Partners object names practice karenge
Simple pointing and naming exercise

Kaisa laga teacher? Aur activities chahiye? 😊"""
        
        elif activity_type == 'group_work':
            return f"""👥 GROUP WORK - Nouns (Grade {grade} {subject})

Hello teacher! Groups mein nouns sikhana bahut maza aata hai!

🌟 Noun Detective Squad:
4-5 bachon ka group banayiye
Classroom mein noun hunt karte jayenge
Jo group zyada nouns dhunde, woh jeet gaya!

🌟 Pakistani Food Market:
Groups ko different Pakistani foods assign kariye
Biryani group, Karahi group, Haleem group
Har group apne food ke bare mein baat karega

🌟 Family Tree Activity:
Ahmed, Fatima, Ali, Ayesha ke families banayiye
"This is Ahmed's Abbu, This is Fatima's Ammi"

Grade {grade} ke liye bilkul perfect hai! Try kariye teacher! 😊"""
            
        else:
            return f"""📚 Nouns Activities (Grade {grade} {subject})

Hello teacher! Main samajh gayi hun aap ko nouns ke activities chahiye.
Grade {grade} ke liye kuch ideas suggest kar rahi hun:

🌟 Simple Noun Recognition:
Classroom objects dikhayen: chair, table, board, bag
Pakistani items bhi include kariye: dupatta, shalwar, kameez

🌟 Person, Place, Thing Categories:
Ahmed (person), Lahore (place), cricket (thing)
Bachon ko examples dene ko kahiye

🌟 My Family Nouns:
Abbu, Ammi, bhai, behen, nano, nana
English translation ke saath sikhayen

Kaisa laga teacher? Aur help chahiye? 😊"""
    
    elif 'verb' in topic:
        return f"""🏃 Verbs Activities (Grade {grade} {subject})

Hello teacher! Verbs ke liye action games best hain!

🌟 Action Time:
Jump, run, sit, stand - bachon ko action karwayiye
"Ahmed is running, Fatima is jumping"

🌟 Daily Activities:
Brush teeth, eat breakfast, play cricket
Pakistani bachon ke daily routine use kariye

🌟 Classroom Verbs:
Read, write, listen, speak, think
Simple actions jo har din karte hain

Grade {grade} ke bachon ko movement activities bahut pasand aati hain!
Try kariye teacher! 😊"""
    
    elif 'addition' in topic or 'add' in topic:
        return f"""➕ Addition Activities (Grade {grade} Math)

Hello teacher! Addition ke liye practical examples use kariye!

🌟 Pakistani Objects Counting:
Mangoes counting: 2 aam + 3 aam = 5 aam
Roti counting: 1 roti + 2 roti = 3 roti

🌟 Money Addition:
Pakistani rupees use kariye
5 rupees + 3 rupees = 8 rupees

🌟 Cricket Score:
Ahmed ne 2 runs banaye, Ali ne 3 runs
Total kitne runs? 2 + 3 = 5

Grade {grade} ke liye bilkul perfect level hai!
Bachon ko bahut samajh aayega! 😊"""
    
    # Assessment responses with grade-based complexity
    elif 'assessment' in topic or session_context.get('assessment_type'):
        assessment_type = session_context.get('assessment_type', 'general')
        
        if grade == 1:
            return f"""📝 Assessment - {topic.title()} (Grade 1 {subject})

Hello teacher! Grade 1 ke liye simple assessment:

🌟 Point and Choose:
Teacher picture dikhaega, bachon ko simple pointing
"Point to the book" - basic recognition only

🌟 Yes/No Questions:
"Is this a chair?" - thumbs up/down
Simple visual recognition

🌟 Circle the Right One:
Pictures mein se correct option circle karna
Very basic multiple choice

Grade 1 ke liye bilkul easy level! Samajh aayega teacher? 😊"""

        elif grade == 3:
            return f"""📝 Assessment - {topic.title()} (Grade 3 {subject})

Hello teacher! Grade 3 ke liye medium assessment:

🌟 Fill in the Blanks:
"Ahmed is a ___" (boy/teacher/student)
Simple sentence completion

🌟 Match the Columns:
Pictures ko words ke saath match karna
Person-Place-Thing categories

🌟 Short Answers:
"Name 3 classroom objects" - listing practice
Simple recall questions

Grade 3 ke liye perfect difficulty! Try kariye teacher! 😊"""

        elif grade >= 4:
            return f"""📝 Assessment - {topic.title()} (Grade {grade} {subject})

Hello teacher! Grade {grade} ke liye challenging assessment:

🌟 Multiple Choice with Reasoning:
"Which is the proper noun: a) book b) Ahmed c) happy"
"Explain why you chose this answer"

🌟 Sentence Analysis:
"Identify subject and object nouns in: Ahmed reads books"
Critical thinking required

🌟 Creative Application:
"Write 3 sentences using abstract and concrete nouns"
Higher-order thinking skills

Grade {grade} ke liye advanced level! Kaisa laga teacher? 😊"""

        else:
            return f"""📝 Assessment - {topic.title()} (Grade {grade} {subject})

Hello teacher! {topic} ke assessment ideas:

🌟 Basic Recognition
🌟 Simple Questions  
🌟 Easy Activities

Kaisa laga teacher? 😊"""

    # General fallback for any topic
    else:
        content_type = "activities" if activity_type else selected_feature
        return f"""Hello teacher! Main samajh gayi hun aap ko "{topic}" ke {content_type} chahiye.

Grade {grade} {subject} ke liye kuch general ideas:

🌟 Pakistani Context:
Local examples use kariye - Ahmed, Fatima, Ali, Ayesha
Familiar things: roti, chawal, cricket, Eid

🌟 Interactive Methods:
Pair work, group work, games aur movement
Bachon ko engage rakhne ke liye hands-on activities

🌟 Simple Language:
Urdu support dein difficult words ke liye
Step by step explain kariye

Grade {grade} ke bachon ke liye perfect level main adjust kar deti hun!
Kaisa laga teacher? Aur specific help chahiye? 😊"""

@app.route('/')
def index():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', posthog_key=POSTHOG_KEY, posthog_host=POSTHOG_HOST)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        phone_number = normalize_phone_number(request.form.get('phone_number', ''))
        password = request.form.get('password')
        
        if not phone_number or not password:
            flash('Please enter both phone number and password', 'error')
            return render_template('login.html')
        
        conn = get_db_connection()
        if not conn:
            flash('Database connection error', 'error')
            return render_template('login.html')
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, password_hash, name FROM users WHERE phone_number = %s", (phone_number,))
            user = cursor.fetchone()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                # Clear session to prevent session fixation
                session.clear()
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                session['phone_number'] = phone_number
                flash(f'Welcome back, {user["name"]}!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid phone number or password', 'error')
        except Exception as e:
            print(f"Login error occurred")  # Don't log sensitive details
            flash('Login failed. Please try again.', 'error')
        finally:
            conn.close()
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        phone_number = normalize_phone_number(request.form.get('phone_number', ''))
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([name, phone_number, password, confirm_password]):
            flash('Please fill in all required fields', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        conn = get_db_connection()
        if not conn:
            flash('Database connection error', 'error')
            return render_template('register.html')
        
        try:
            cursor = conn.cursor()
            # Check if phone number already exists
            cursor.execute("SELECT id FROM users WHERE phone_number = %s", (phone_number,))
            if cursor.fetchone():
                flash('Phone number already registered', 'error')
                return render_template('register.html')
            
            # Hash password and create user
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute("""
                INSERT INTO users (name, phone_number, password_hash) 
                VALUES (%s, %s, %s) RETURNING id
            """, (name, phone_number, password_hash))
            
            result = cursor.fetchone()
            if result:
                user_id = result['id']
                conn.commit()
                
                # Clear any existing session data and log in the new user
                session.clear()
                session['user_id'] = user_id
                session['user_name'] = name
                session['phone_number'] = phone_number
                flash(f'Registration successful! Welcome to USTAAD DOST, {name}!', 'success')
                return redirect(url_for('profile_setup'))
            else:
                flash('Registration failed. Please try again.', 'error')
                conn.rollback()
            
        except Exception as e:
            print(f"Registration error occurred")  # Don't log sensitive details
            flash('Registration failed. Please try again.', 'error')
            conn.rollback()
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/profile-setup', methods=['GET', 'POST'])
def profile_setup():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        cnic = request.form.get('cnic', '').strip() if request.form.get('cnic') else None
        email = request.form.get('email', '').strip() if request.form.get('email') else None
        address = request.form.get('address', '').strip() if request.form.get('address') else None
        
        # Handle file upload
        profile_photo_path = None
        if 'profile_photo' in request.files:
            file = request.files['profile_photo']
            if file and file.filename:
                filename = secure_filename(file.filename)
                if allowed_file(filename, {'png', 'jpg', 'jpeg'}):
                    # Create upload directory
                    upload_dir = Path('uploads/profiles')
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique filename
                    unique_filename = f"{session['user_id']}_{uuid.uuid4().hex[:8]}_{filename}"
                    file_path = upload_dir / unique_filename
                    
                    try:
                        file.save(file_path)
                        profile_photo_path = str(file_path)
                    except Exception as e:
                        print(f"File upload error occurred")  # Don't log sensitive details
                        flash('Photo upload failed. Please try again.', 'warning')
        
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                if profile_photo_path:
                    cursor.execute("""
                        UPDATE users SET cnic = %s, email = %s, address = %s, profile_photo = %s 
                        WHERE id = %s
                    """, (cnic, email, address, profile_photo_path, session['user_id']))
                else:
                    cursor.execute("""
                        UPDATE users SET cnic = %s, email = %s, address = %s 
                        WHERE id = %s
                    """, (cnic, email, address, session['user_id']))
                conn.commit()
                flash('Profile updated successfully!', 'success')
                return redirect(url_for('index'))
            except Exception as e:
                print(f"Profile update error occurred")  # Don't log sensitive details
                flash('Profile update failed. Please try again.', 'error')
                conn.rollback()
            finally:
                conn.close()
    
    return render_template('profile_setup.html', posthog_key=POSTHOG_KEY, posthog_host=POSTHOG_HOST)

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def normalize_phone_number(phone):
    """Normalize Pakistani phone number to standard format"""
    if not phone:
        return phone
    
    # Remove all non-digits
    digits_only = ''.join(filter(str.isdigit, phone))
    
    # Handle different Pakistani phone formats
    if digits_only.startswith('92'):
        # International format: +92 300 1234567 → 0300-1234567
        if len(digits_only) == 12:
            return f"0{digits_only[2:5]}-{digits_only[5:]}"
    elif digits_only.startswith('03') and len(digits_only) == 11:
        # Standard format: 03001234567 → 0300-1234567
        return f"{digits_only[:4]}-{digits_only[4:]}"
    elif len(digits_only) == 10 and digits_only.startswith('3'):
        # Missing leading zero: 3001234567 → 0300-1234567
        return f"0{digits_only[:3]}-{digits_only[3:]}"
    
    # Return as-is if format doesn't match
    return phone

@app.route('/yearly-planner')
def yearly_planner():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('yearly_planner.html', posthog_key=POSTHOG_KEY, posthog_host=POSTHOG_HOST)

@app.route('/grading-buddy')
def grading_buddy():
    # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('grading_buddy.html', posthog_key=POSTHOG_KEY, posthog_host=POSTHOG_HOST)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', posthog_key=POSTHOG_KEY, posthog_host=POSTHOG_HOST)

@app.route('/chat', methods=['POST'])
def chat():
    """Simplified chatbot with direct OpenAI responses"""
    # Guard against malformed requests
    if not request.json or 'message' not in request.json:
        return jsonify({'message': 'Invalid request format'})
    
    user_message = request.json.get('message', '').strip()
    
    # Handle Free Chat mode first - bypass all menu logic
    if session.get('selected_feature') == 'free_chat' and user_message.lower() not in ['menu', 'start', '← back to menu']:
        ai_response = get_ai_response(user_message, "general", session)
        return jsonify({'message': ai_response, 'is_markdown': True})
    
    # Handle special greetings and commands
    if user_message.lower() in ['hi', 'hello', 'hey', 'menu', 'start']:
        session.clear()
        session.modified = True
        return jsonify({
            'message': 'Welcome to U-Dost! What do you need help with?',
            'options': [
                '📚 Lesson Plans',
                '🎯 Teaching Strategies', 
                '🎮 Activities',
                '📖 Definitions',
                '📝 Assessments',
                '🎲 Hooks/Games',
                '💡 Examples',
                '💬 Free Chat'
            ],
            'show_menu': True
        })
    
    # Handle Free Chat selection
    if user_message.lower() in ['💬 free chat', 'free chat']:
        session['selected_feature'] = 'free_chat'
        session.modified = True
        return jsonify({
            'message': '💬 **Free Chat Mode Activated!** \n\nI\'m ready to help you with anything! Ask me questions about education, lesson planning, or any topic. Let\'s have a natural conversation! 🚀',
            'show_menu': False
        })
    
    # Handle main menu options
    menu_options = {
        '📚 lesson plans': 'lesson_plans',
        'lesson plans': 'lesson_plans',
        '🎯 teaching strategies': 'teaching_strategies', 
        'teaching strategies': 'teaching_strategies',
        '🎮 activities': 'activities',
        'activities': 'activities',
        '📖 definitions': 'definitions',
        'definitions': 'definitions',
        '📝 assessments': 'assessments',
        'assessments': 'assessments',
        '🎲 hooks/games': 'hooks_games',
        'hooks/games': 'hooks_games',
        'hooks': 'hooks_games',
        'games': 'hooks_games',
        '💡 examples': 'examples',
        'examples': 'examples'
    }
    
    # Handle Activities submenu
    if user_message.lower() in ['🎮 activities', 'activities']:
        session['selected_feature'] = 'activities'
        session.modified = True
        return jsonify({
            'message': 'You selected: **ACTIVITIES**\n\nChoose Activity Type:',
            'options': [
                '👥 Group Work Activities',
                '👫 Pair Work Activities', 
                '👤 Independent Work Activities',
                '📝 Assignment/Homework Activities',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    # Handle Assessments submenu
    if user_message.lower() in ['📝 assessments', 'assessments']:
        session['selected_feature'] = 'assessments'
        session.modified = True
        return jsonify({
            'message': '📝 Hello teacher! Assessment ke liye kya banana hai?\n\nChoose Assessment Type:',
            'options': [
                '🔹 MCQs (Multiple Choice Questions)',
                '🔹 Fill in the Blanks',
                '🔹 Short Comprehension Questions',
                '🔹 True/False Questions',
                '🔹 Exit Tickets',
                '🔹 Thumbs Up/Down',
                '🔹 Stand or Clap Activities',
                '🔹 Quick Quiz Games',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    # Handle other main menu selections (go directly to grade selection)
    if user_message.lower() in menu_options and user_message.lower() not in ['🎮 activities', 'activities', '📝 assessments', 'assessments']:
        session['selected_feature'] = menu_options[user_message.lower()]
        session['selected_feature_display'] = user_message
        session.modified = True
        return jsonify({
            'message': f'You selected: **{user_message.upper()}**\n\nSelect Grade:',
            'options': [
                '🔹 Grade 1',
                '🔹 Grade 2', 
                '🔹 Grade 3',
                '🔹 Grade 4',
                '🔹 Grade 5',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    # Handle Activity Type selections
    activity_types = {
        '👥 group work activities': 'group_work',
        'group work activities': 'group_work',
        '👫 pair work activities': 'pair_work', 
        'pair work activities': 'pair_work',
        '👤 independent work activities': 'independent_work',
        'independent work activities': 'independent_work',
        '📝 assignment/homework activities': 'homework',
        'assignment/homework activities': 'homework'
    }
    
    # Handle Assessment Type selections
    assessment_types = {
        '🔹 mcqs (multiple choice questions)': 'mcqs',
        'mcqs (multiple choice questions)': 'mcqs',
        'mcqs': 'mcqs',
        '🔹 fill in the blanks': 'fill_blanks',
        'fill in the blanks': 'fill_blanks',
        '🔹 short comprehension questions': 'comprehension',
        'short comprehension questions': 'comprehension',
        '🔹 true/false questions': 'true_false',
        'true/false questions': 'true_false',
        '🔹 exit tickets': 'exit_tickets',
        'exit tickets': 'exit_tickets',
        '🔹 thumbs up/down': 'thumbs_up_down',
        'thumbs up/down': 'thumbs_up_down',
        '🔹 stand or clap activities': 'stand_clap',
        'stand or clap activities': 'stand_clap',
        '🔹 quick quiz games': 'quiz_games',
        'quick quiz games': 'quiz_games'
    }
    
    if user_message.lower() in activity_types:
        session['activity_type'] = activity_types[user_message.lower()]
        session['selected_feature_display'] = user_message
        session.modified = True
        return jsonify({
            'message': f'You selected: **{user_message.upper()}**\n\nSelect Grade:',
            'options': [
                '🔹 Grade 1',
                '🔹 Grade 2', 
                '🔹 Grade 3',
                '🔹 Grade 4',
                '🔹 Grade 5',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    # Handle Assessment Type selections
    if user_message.lower() in assessment_types:
        session['assessment_type'] = assessment_types[user_message.lower()]
        session['selected_feature_display'] = user_message
        session.modified = True
        return jsonify({
            'message': f'You selected: **{user_message.upper()}**\n\nSelect Grade:',
            'options': [
                '🔹 Grade 1',
                '🔹 Grade 2', 
                '🔹 Grade 3',
                '🔹 Grade 4',
                '🔹 Grade 5',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    # Handle grade selection  
    grade_options = {
        '🔹 grade 1': 1, 'grade 1': 1,
        '🔹 grade 2': 2, 'grade 2': 2,
        '🔹 grade 3': 3, 'grade 3': 3,
        '🔹 grade 4': 4, 'grade 4': 4,
        '🔹 grade 5': 5, 'grade 5': 5
    }
    
    if user_message.lower() in grade_options and ('selected_feature' in session or 'activity_type' in session):
        grade = grade_options[user_message.lower()]
        session['grade'] = grade
        session.modified = True
        
        feature_display = session.get('selected_feature_display', 'Content')
        subjects = ['English', 'Math', 'Urdu', 'Islamiyat', 'General Knowledge', 'Social Studies', 'Science']
        
        return jsonify({
            'message': f'You selected: **{feature_display} for Grade {grade}**\n\nSelect Subject:',
            'options': [f'📚 {subject}' if subject == 'English' else 
                       f'🔢 {subject}' if subject == 'Math' else
                       f'🇵🇰 {subject}' if subject == 'Urdu' else
                       f'🕌 {subject}' if subject == 'Islamiyat' else
                       f'🌍 {subject}' if subject == 'General Knowledge' else
                       f'📊 {subject}' if subject == 'Social Studies' else
                       f'🔬 {subject}' for subject in subjects] + ['🔄 Change Grade', '← Back to Menu'],
            'show_menu': True
        })
    
    # Handle subject selection
    subject_mapping = {
        '📚 english': 'English', 'english': 'English',
        '🔢 math': 'Math', 'math': 'Math',
        '🇵🇰 urdu': 'Urdu', 'urdu': 'Urdu', 
        '🕌 islamiyat': 'Islamiyat', 'islamiyat': 'Islamiyat',
        '🌍 general knowledge': 'General Knowledge', 'general knowledge': 'General Knowledge',
        '📊 social studies': 'Social Studies', 'social studies': 'Social Studies',
        '🔬 science': 'Science', 'science': 'Science'
    }
    
    if user_message.lower() in subject_mapping and 'grade' in session:
        subject = subject_mapping[user_message.lower()]
        session['subject'] = subject
        session.modified = True
        
        grade = session['grade']
        feature_display = session.get('selected_feature_display', 'Content')
        
        return jsonify({
            'message': f'You selected: **{feature_display} for Grade {grade} {subject}**\n\n📝 Please enter the topic you want to teach:',
            'show_menu': False,
            'show_input': True,
            'input_placeholder': 'Enter topic (e.g., "Past Tense Verbs", "Addition and Subtraction", "Types of Animals")'
        })
    
    # Handle topic input and generate content
    if ('grade' in session and 'subject' in session and 
        ('selected_feature' in session or 'activity_type' in session) and 
        not user_message.lower().startswith(('🔹', '📚', '🔢', '🇵🇰', '🕌', '🌍', '📊', '🔬', '👥', '👫', '👤', '📝', '🎮', '← back', 'menu'))):
        
        topic = user_message.strip()
        grade = session['grade']
        subject = session['subject']
        
        # Determine content type
        if 'activity_type' in session:
            activity_type = session['activity_type']
            content_type = f"{activity_type}_activities"
            feature_display = session.get('selected_feature_display', 'Activities')
        else:
            content_type = session.get('selected_feature', 'lesson_plans')
            feature_display = session.get('selected_feature_display', 'Content')
        
        # Create context for AI generation
        prompt = f"""Create {feature_display} for Grade {grade} {subject} on the topic: "{topic}"

Grade: {grade}
Subject: {subject}
Topic: {topic}
Content Type: {content_type}

IMPORTANT: You are U-DOST, a friendly Pakistani teacher assistant. Generate content specifically for Pakistani ESL students with:
- Pakistani cultural examples and contexts (Pakistani names like Ahmed, Fatima, Ali, Ayesha)
- Local examples (biryani, cricket, Eid celebrations, etc.)
- Urdu translation support for difficult English words
- Content appropriate for Pakistani classroom settings
- Islamic values and Pakistani customs where relevant"""

        # Add specific instructions based on content type
        if 'group_work' in content_type:
            prompt += "\n\nGenerate GROUP WORK ACTIVITIES with 3-4 collaborative exercises that work well in Pakistani classrooms."
        elif 'pair_work' in content_type:
            prompt += "\n\nGenerate PAIR WORK ACTIVITIES with 3-4 partner-based exercises suitable for Grade {grade} students."
        elif 'independent_work' in content_type:
            prompt += "\n\nGenerate INDEPENDENT WORK ACTIVITIES with 3-4 self-directed exercises for individual practice."
        elif 'homework' in content_type:
            prompt += "\n\nGenerate ASSIGNMENT/HOMEWORK ACTIVITIES with 3-4 take-home exercises for reinforcement."
        elif content_type == 'lesson_plans':
            prompt += f"\n\n{UDOST_TEACHING_METHODOLOGY}"
        
        # Generate AI response
        ai_response = get_ai_response(prompt, "teaching", session)
        
        if not ai_response:
            ai_response = f"I'm ready to help with {feature_display} for Grade {grade} {subject} on '{topic}', but I'm having trouble connecting right now. Please try again!"
        
        return jsonify({
            'message': ai_response,
            'is_markdown': True,
            'options': ['🔄 Try Different Topic', '🔄 Change Subject', '🔄 Change Grade', '← Back to Menu'],
            'show_menu': True
        })
    
    # Handle back to menu
    if user_message.lower() in ['← back to menu', 'back to menu', 'menu']:
        session.clear()
        session.modified = True
        return jsonify({
            'message': 'Welcome to U-Dost! What do you need help with?',
            'options': [
                '📚 Lesson Plans',
                '🎯 Teaching Strategies', 
                '🎮 Activities',
                '📖 Definitions',
                '📝 Assessments',
                '🎲 Hooks/Games',
                '💡 Examples',
                '💬 Free Chat'
            ],
            'show_menu': True
        })
    
    # Default fallback
    return jsonify({
        'message': 'I didn\'t understand that. Please use the menu options to get started!',
        'options': ['← Back to Menu'],
        'show_menu': True
    })

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Handle file uploads for the chatbot"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file info
        original_filename = secure_filename(file.filename)
        file_size = len(file.read())
        file.seek(0)  # Reset file pointer
        
        # Check file size
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'File too large (max 12MB)'}), 400
        
        # Save the file temporarily to determine MIME type
        temp_path = os.path.join(UPLOAD_DIR, original_filename)
        file.save(temp_path)
        
        mime_type = get_file_mime_type(temp_path)
        
        # Determine file type and target directory
        if mime_type in ALLOWED_IMAGE_TYPES:
            file_type = 'image'
            target_dir = IMAGES_DIR
            # Strip metadata from images
            strip_image_metadata(temp_path)
        elif mime_type in ALLOWED_DOC_TYPES:
            file_type = 'document'
            target_dir = DOCS_DIR
        else:
            os.remove(temp_path)
            return jsonify({'error': f'Unsupported file type: {mime_type}'}), 400
        
        # Generate unique filename and move to proper directory
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(original_filename)[1]
        final_filename = f"{file_id}{file_extension}"
        final_path = os.path.join(target_dir, final_filename)
        
        os.rename(temp_path, final_path)
        
        # Extract text from documents
        extracted_text = ""
        if file_type == 'document':
            if mime_type == 'application/pdf':
                extracted_text = extract_text_from_pdf(final_path)
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                extracted_text = extract_text_from_docx(final_path)
            elif mime_type == 'text/plain':
                with open(final_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
        
        # Save metadata
        metadata = save_file_metadata(file_id, original_filename, mime_type, file_size, file_type)
        if extracted_text:
            metadata['extracted_text'] = extracted_text
            # Update metadata file
            metadata_path = os.path.join(META_DIR, f"{file_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        # Schedule cleanup
        cleanup_old_files()
        
        return jsonify({
            'file_id': file_id,
            'filename': original_filename,
            'size': file_size,
            'type': file_type,
            'mime_type': mime_type
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': 'Upload failed'}), 500

def generate_math_chapters(grade):
    """Generate Mathematics chapter structure"""
    base_chapters = {
        1: {
            "Chapter 1: Numbers 1-20": generate_exercises(["Counting Objects", "Number Recognition", "Number Writing", "Number Sequence"]),
            "Chapter 2: Addition": generate_exercises(["Adding Objects", "Addition Facts", "Word Problems", "Mental Math"]),
            "Chapter 3: Subtraction": generate_exercises(["Taking Away", "Subtraction Facts", "Simple Problems", "Comparison"]),
            "Chapter 4: Shapes": generate_exercises(["Basic Shapes", "Shape Properties", "Shape Sorting", "Drawing Shapes"]),
            "Chapter 5: Measurement": generate_exercises(["Long and Short", "Heavy and Light", "Big and Small", "Comparing Size"])
        },
        2: {
            "Chapter 1: Numbers 1-100": generate_exercises(["Two-digit Numbers", "Place Value", "Number Patterns", "Skip Counting"]),
            "Chapter 2: Addition and Subtraction": generate_exercises(["Two-digit Addition", "Regrouping", "Word Problems", "Estimation"]),
            "Chapter 3: Multiplication": generate_exercises(["Groups and Arrays", "Times Tables 2-5", "Multiplication Facts", "Problem Solving"]),
            "Chapter 4: Geometry": generate_exercises(["2D Shapes", "3D Shapes", "Symmetry", "Patterns"]),
            "Chapter 5: Measurement": generate_exercises(["Length", "Weight", "Capacity", "Time"])
        },
        3: {
            "Chapter 1: Numbers to 1000": generate_exercises(["Three-digit Numbers", "Place Value", "Rounding", "Number Comparison"]),
            "Chapter 2: Operations": generate_exercises(["Addition with Regrouping", "Subtraction with Borrowing", "Multiplication Tables", "Division Basics"]),
            "Chapter 3: Fractions": generate_exercises(["Parts of Whole", "Comparing Fractions", "Adding Fractions", "Fraction Problems"]),
            "Chapter 4: Geometry": generate_exercises(["Angles", "Lines", "Polygons", "Area and Perimeter"]),
            "Chapter 5: Data": generate_exercises(["Graphs", "Charts", "Data Collection", "Interpretation"])
        },
        4: {
            "Chapter 1: Large Numbers": generate_exercises(["Numbers to 10,000", "Place Value", "Rounding", "Number Operations"]),
            "Chapter 2: Multiplication and Division": generate_exercises(["Multi-digit Multiplication", "Long Division", "Word Problems", "Factors and Multiples"]),
            "Chapter 3: Fractions and Decimals": generate_exercises(["Equivalent Fractions", "Decimal Numbers", "Converting Forms", "Operations"]),
            "Chapter 4: Geometry": generate_exercises(["Quadrilaterals", "Triangles", "Circles", "Transformations"]),
            "Chapter 5: Measurement": generate_exercises(["Metric System", "Area", "Perimeter", "Volume"])
        },
        5: {
            "Chapter 1: Advanced Numbers": generate_exercises(["Large Numbers", "Prime Numbers", "Factors", "Multiples"]),
            "Chapter 2: Operations": generate_exercises(["Advanced Multiplication", "Long Division", "Order of Operations", "Problem Solving"]),
            "Chapter 3: Fractions and Decimals": generate_exercises(["Mixed Numbers", "Decimal Operations", "Percentage", "Ratio"]),
            "Chapter 4: Algebra Basics": generate_exercises(["Patterns", "Variables", "Simple Equations", "Expressions"]),
            "Chapter 5: Statistics": generate_exercises(["Data Analysis", "Mean and Mode", "Probability", "Graphs"])
        }
    }
    
    return base_chapters.get(grade, base_chapters[1])

def generate_urdu_chapters(grade):
    """Generate Urdu chapter structure"""
    base_chapters = {
        1: {
            "سبق ۱: حروف تہجی": generate_exercises(["الف سے ے تک", "حروف کی پہچان", "حروف لکھنا", "آوازیں"]),
            "سبق ۲: آسان الفاظ": generate_exercises(["روزمرہ الفاظ", "الفاظ پڑھنا", "الفاظ لکھنا", "معنی سمجھنا"]),
            "سبق ۳: خاندان": generate_exercises(["والدین", "بہن بھائی", "رشتہ دار", "احترام"]),
            "سبق ۴: گھر": generate_exercises(["گھر کے کمرے", "سامان", "صفائی", "ذمہ داریاں"]),
            "سبق ۵: دوست": generate_exercises(["دوستی", "کھیل", "مدد", "شائستگی"])
        }
    }
    
    return base_chapters.get(grade, generate_default_chapters(grade, 'Urdu'))

def generate_gk_chapters(grade):
    """Generate General Knowledge chapter structure"""
    base_chapters = {
        1: {
            "Chapter 1: About Me": generate_exercises(["My Name", "My Age", "My Family", "My School"]),
            "Chapter 2: My Body": generate_exercises(["Body Parts", "Five Senses", "Keeping Clean", "Staying Healthy"]),
            "Chapter 3: Animals": generate_exercises(["Pet Animals", "Farm Animals", "Wild Animals", "Animal Homes"]),
            "Chapter 4: Plants": generate_exercises(["Trees", "Flowers", "Fruits", "Plant Parts"]),
            "Chapter 5: My Country": generate_exercises(["Pakistan", "Our Flag", "National Symbols", "Famous Places"])
        }
    }
    
    return base_chapters.get(grade, generate_default_chapters(grade, 'General Knowledge'))

def generate_science_chapters(grade):
    """Generate Science chapter structure"""
    base_chapters = {
        4: {
            "Chapter 1: Living Things": generate_exercises(["Plants and Animals", "Life Processes", "Habitats", "Food Chains"]),
            "Chapter 2: Human Body": generate_exercises(["Body Systems", "Nutrition", "Exercise", "Health and Hygiene"]),
            "Chapter 3: Matter": generate_exercises(["Solids, Liquids, Gases", "Properties", "Changes", "Materials"]),
            "Chapter 4: Forces and Motion": generate_exercises(["Push and Pull", "Movement", "Simple Machines", "Energy"]),
            "Chapter 5: Our Environment": generate_exercises(["Weather", "Seasons", "Water Cycle", "Conservation"])
        },
        5: {
            "Chapter 1: Cells and Organisms": generate_exercises(["Cell Structure", "Microorganisms", "Classification", "Microscopy"]),
            "Chapter 2: Ecosystems": generate_exercises(["Food Webs", "Biodiversity", "Adaptation", "Conservation"]),
            "Chapter 3: Physical Science": generate_exercises(["Light", "Sound", "Heat", "Electricity"]),
            "Chapter 4: Earth Science": generate_exercises(["Rocks and Soil", "Water Bodies", "Weather Patterns", "Natural Resources"]),
            "Chapter 5: Space": generate_exercises(["Solar System", "Day and Night", "Seasons", "Moon Phases"])
        }
    }
    
    return base_chapters.get(grade, generate_default_chapters(grade, 'Science'))

def generate_sst_chapters(grade):
    """Generate Social Studies chapter structure"""
    base_chapters = {
        4: {
            "Chapter 1: Our Community": generate_exercises(["Community Helpers", "Local Government", "Rules and Laws", "Civic Duties"]),
            "Chapter 2: Geography": generate_exercises(["Maps and Globes", "Landforms", "Climate", "Natural Resources"]),
            "Chapter 3: History": generate_exercises(["Past and Present", "Historical Figures", "Important Events", "Cultural Heritage"]),
            "Chapter 4: Pakistan Studies": generate_exercises(["Provinces", "Cities", "Culture", "National Identity"]),
            "Chapter 5: Global Awareness": generate_exercises(["Countries", "Cultures", "International Relations", "Global Issues"])
        }
    }
    
    return base_chapters.get(grade, generate_default_chapters(grade, 'Social Studies'))

def generate_islamiyat_chapters(grade):
    """Generate Islamiyat chapter structure"""
    base_chapters = {
        1: {
            "سبق ۱: کلمہ طیبہ": generate_exercises(["کلمے کا اردو ترجمہ", "کلمے کی اہمیت", "یاد کرنا", "سمجھنا"]),
            "سبق ۲: نماز": generate_exercises(["نماز کی اہمیت", "وضو", "نماز کے اوقات", "قبلہ"]),
            "سبق ۳: دعائیں": generate_exercises(["روزانہ دعائیں", "کھانے کی دعا", "سونے کی دعا", "اٹھنے کی دعا"]),
            "سبق ۴: اخلاق": generate_exercises(["سچ بولنا", "والدین کا احترام", "بزرگوں کی عزت", "دوسروں سے اچھا برتاؤ"]),
            "سبق ۵: پیغمبر": generate_exercises(["حضرت محمد ﷺ", "آپ کی زندگی", "آپ کی تعلیمات", "آپ سے محبت"])
        }
    }
    
    return base_chapters.get(grade, generate_default_chapters(grade, 'Islamiyat'))

def generate_default_chapters(grade, subject):
    """Generate default chapter structure for any subject"""
    chapters = {}
    for i in range(1, 6):  # 5 chapters
        chapter_title = f"Chapter {i}: {subject} Basics {i}"
        chapters[chapter_title] = generate_exercises([
            f"Topic {i}.1", f"Topic {i}.2", f"Topic {i}.3", f"Topic {i}.4"
        ])
    return chapters

# Book upload functionality removed - now using predefined curriculum books

def generate_udost_content(feature_type, curriculum_selection, session_data=None):
    """Generate curriculum-specific content for U-DOST system"""
    grade = curriculum_selection.get('grade')
    subject = curriculum_selection.get('subject')
    book = curriculum_selection.get('book', 'textbook')
    chapter = curriculum_selection.get('chapter')
    skill_category = curriculum_selection.get('skill_category')
    exercise = curriculum_selection.get('exercise', 'General Exercise')
    
    # Get actual exercise content from JSON (for Grade 4 English)
    exercise_content = ""
    if grade == 4 and subject and subject.lower() == 'english':
        print(f"🔍 Fetching actual exercise content for: {exercise}")
        book_content = get_auto_loaded_book_content(grade, subject)
        if book_content and 'chapters' in book_content:
            # Find the matching chapter key
            matching_chapter = None
            for chapter_key in book_content['chapters'].keys():
                if chapter in chapter_key:
                    matching_chapter = chapter_key
                    break
            
            if matching_chapter and skill_category in book_content['chapters'][matching_chapter]:
                exercises = book_content['chapters'][matching_chapter][skill_category]
                # Find the specific exercise
                for ex in exercises:
                    if isinstance(ex, dict) and exercise in ex.get('title', ''):
                        exercise_content = f"\n\nEXERCISE DETAILS:\nTitle: {ex.get('title')}\nType: {ex.get('type')}\nCategory: {skill_category}"
                        print(f"✅ Found exercise content: {ex.get('title')}")
                        break
    
    # Create context for AI with Pakistani Teaching Methodology
    context = f"""
    You are U-DOST, a friendly Pakistani teacher assistant. Generate content for:
    
    Grade: {grade}
    Subject: {subject}  
    Book: {book}
    Chapter: {chapter}
    Skill Focus: {skill_category}
    Selected Exercise: {exercise}
    {exercise_content}
    
    IMPORTANT: You MUST follow this Pakistani Teaching Methodology:
    {UDOST_TEACHING_METHODOLOGY}
    
    Remember: All content must match the SPECIFIC EXERCISE from the selected chapter/book, be tailored for Pakistani ESL students, and use local cultural examples.
    """
    
    content_prompts = {
        'lesson_plans': f"""Create a detailed lesson plan for Grade {grade} {subject}, Chapter {chapter}, focusing on {skill_category}.

MANDATORY: Follow the 6-step lesson structure exactly:

## 📝 **LESSON PLAN**

### **Learning Objectives:**
- Clear, measurable goals appropriate for Grade {grade}

### **Materials Needed:**
- List required materials (textbook, whiteboard, charts, etc.)

### **Lesson Duration:** 40 minutes

### **LESSON STRUCTURE (6 Essential Steps):**

**1. RECALL (5 minutes):**
- Quick review of previous learning/prerequisite knowledge
- Connect to what students already know

**2. HOOK (5 minutes):**
- Engaging activity to capture student interest and introduce topic
- Make it exciting and relevant to Pakistani children

**3. EXPLAIN (15 minutes):**
- Clear explanation using visual aids, examples, and demonstrations
- Use simple Pakistani English with cultural examples

**4. GUIDED PRACTICE (10 minutes):**
- Teacher-led practice with student participation
- Interactive exercises together

**5. INDEPENDENT PRACTICE (3 minutes):**
- Students work from textbook exercises independently
- Individual application of learning

**6. QUICK CONCLUSION (2 minutes):**
- Brief summary and key takeaways
- What did we learn today?

### **Pakistani ESL Considerations:**
- Use Pakistani names (Ahmed, Fatima, Ali, Ayesha)
- Include local examples (biryani, cricket, Eid, etc.)
- Provide Urdu translations for difficult words
- Consider limited English vocabulary

Make it practical and ready to use in Pakistani classrooms.""",

        'teaching_strategies': f"""Provide effective teaching strategies for Grade {grade} {subject}, Chapter {chapter}, focusing on {skill_category}.

## 🎯 **SKILL-SPECIFIC STRATEGIES**

### **For {skill_category} Skills, use these proven methods:**

**If focusing on READING:**
- Echo Reading, Choral Reading, Paired Reading
- Picture Walk, Prediction, Think-Aloud
- Phonics Blending, Sight Word Recognition
- Reading Comprehension Questions

**If focusing on WRITING:**
- Sentence Starters, Writing Frames
- Guided Writing, Shared Writing
- Grammar Integration, Vocabulary Building
- Peer Editing, Self-Correction

**If focusing on ORAL COMMUNICATION:**
- Show and Tell, Role Play, Storytelling
- Question-Answer Sessions, Group Discussions
- Pronunciation Practice, Vocabulary Games
- Listen and Repeat Activities

**If focusing on COMPREHENSION:**
- KWL Charts (Know-Want-Learn)
- Story Mapping, Sequence Activities
- Main Idea and Details, Cause and Effect
- Making Connections, Inference Skills

**If focusing on GRAMMAR:**
- Grammar Games, Pattern Practice
- Sentence Building, Error Correction
- Visual Grammar Charts, Examples and Non-examples
- Contextual Grammar Teaching

**If focusing on VOCABULARY:**
- Picture-Word Association, Word Maps
- Synonym/Antonym Games, Context Clues
- Word Families, Vocabulary Journals
- Total Physical Response (TPR)

### **Pakistani ESL Adaptations:**
- Use familiar cultural references (Pakistani foods, festivals, places)
- Provide pronunciation guides for difficult English words
- Include mother tongue support strategies
- Consider limited English vocabulary of students
- Focus on practical, communicative English skills

Make strategies practical and ready to implement in Pakistani classrooms.""",

        'activities': f"""Design 6 engaging activities for Grade {grade} {subject}, Chapter {chapter}, {skill_category}.

## 🎨 **CULTURALLY-ADAPTED ACTIVITIES**

### **Activity 1: Independent Work**
- Individual practice that matches exact textbook content
- Use Pakistani names (Ahmed, Fatima, Ali, Ayesha)
- Include familiar objects (rickshaw, chapati, mangoes)

### **Activity 2: Group Activity** 
- Collaborative learning with Islamic values emphasis
- Reference Pakistani festivals (Eid, Independence Day, Jashn-e-Baharan)
- Use cricket or other familiar sports as examples

### **Activity 3: Assignment/Homework**
- Take-home practice involving family members
- Connect to home life in Pakistani context
- Include respect for elders and family values

### **Activity 4: Pair Work**
- Partner activities promoting cooperation
- Use local Pakistani contexts (bazaar, masjid, school)
- Include helping and sharing concepts

### **Activity 5: Creative Activity**
- Arts, crafts using Pakistani cultural elements
- Traditional patterns, local animals, or foods
- Express creativity while staying culturally appropriate

### **Activity 6: Assessment Activity**
- Fun way to check learning through games
- Use Pakistani context (cricket scoring, counting rotis)
- Make assessment enjoyable and stress-free

### **Each Activity Includes:**
- **Clear Instructions:** Step-by-step in simple English
- **Time Required:** Realistic timing for Pakistani classrooms
- **Materials Needed:** Available local materials
- **Learning Outcomes:** What students will achieve
- **Urdu Support:** Translation hints for difficult words
- **Cultural Connection:** How it relates to students' lives

**Pakistani ESL Adaptations:**
- Use simple vocabulary appropriate for Grade {grade}
- Include pronunciation guides for difficult English words
- Connect activities to real Pakistani student experiences
- Ensure activities work in typical Pakistani classroom settings""",

        'definitions': f"""Provide clear, age-appropriate definitions and explanations for key concepts in Grade {grade} {subject}, Chapter {chapter}, {skill_category}.

Include:
1. **Key Terms** (Main vocabulary with simple definitions)
2. **Concept Explanations** (Break down complex ideas)
3. **Pakistani Examples** (Local context for better understanding)
4. **Urdu Translations** (Where helpful for comprehension)
5. **Memory Aids** (Mnemonics, rhymes, visual associations)
6. **Practice Questions** (Simple questions to check understanding)

Make definitions simple and relatable for Grade {grade} Pakistani students.""",

        'assessment_tools': f"""Create comprehensive assessment tools for Grade {grade} {subject}, Chapter {chapter}, {skill_category}.

Include:
1. **Quick Quiz** (5 multiple choice questions with answers)
2. **True/False Statements** (5 statements with answers)
3. **Thumbs Up/Down Activities** (5 interactive checks)
4. **Exit Tickets** (3 reflection questions)
5. **Fill in the Blanks** (5 sentences with answers)
6. **Comprehension Questions** (3-5 open-ended questions)
7. **Assessment Rubric** (Simple scoring guide)

Provide answer keys for all assessments. Make them engaging and age-appropriate.""",

        'educational_games': f"""Design 5 fun educational games and hooks for Grade {grade} {subject}, Chapter {chapter}, {skill_category}.

Include:
1. **Warm-up Game** (Start class with energy)
2. **Review Game** (Reinforce previous learning)
3. **Practice Game** (Make drilling fun)
4. **Group Competition** (Team-based activity)
5. **Closing Hook** (End class memorably)

For each game provide:
- Rules and setup
- Time required  
- Materials needed
- Learning objectives
- Pakistani cultural elements (where appropriate)
- Variations for different skill levels""",

        'examples_practice': f"""Provide detailed examples and practice exercises for Grade {grade} {subject}, Chapter {chapter}, {skill_category}.

Include:
1. **Worked Examples** (Step-by-step solutions)
2. **Pakistani Context Examples** (Local scenarios, familiar settings)
3. **Practice Problems** (5-10 exercises with answers)
4. **Common Mistakes** (What to watch out for)
5. **Tips and Tricks** (Memory aids, shortcuts)
6. **Extension Activities** (For advanced students)
7. **Real-world Applications** (How this applies to daily life in Pakistan)

Make examples relatable and practice progressive from easy to challenging."""
    }
    
    if feature_type not in content_prompts:
        return jsonify({
            'message': '❌ Invalid feature type selected.',
            'options': ['← Back to Menu'],
            'show_menu': True
        })
    
    prompt = context + content_prompts[feature_type]
    
    # Get AI response with session context
    try:
        print(f"🤖 Calling AI with prompt for {feature_type}...")
        ai_response = get_ai_response(prompt, "teaching", session_data)
        
        if ai_response and ai_response.strip():
            print(f"✅ AI generated content successfully ({len(ai_response)} characters)")
            # Format according to user requirements
            feature_display = feature_type.replace('_', ' ').upper()
            chapter_display = chapter if chapter else "Chapter"
            skill_display = skill_category if skill_category else "Exercise"
            
            header = f"✅ Generating {feature_display} for:\nGrade {grade} → {subject} → {chapter_display} → {skill_display}\n\n"
            
            # Add specific emojis based on feature type
            if feature_type == 'activities':
                header += f"🎮 CLASSROOM ACTIVITIES:\n"
            elif feature_type == 'lesson_plans':
                header += f"📚 LESSON PLAN:\n"
            elif feature_type == 'teaching_strategies':
                header += f"🎯 TEACHING STRATEGIES:\n"
            elif feature_type == 'assessment_tools':
                header += f"📝 ASSESSMENT TOOLS:\n"
            elif feature_type == 'definitions':
                header += f"📖 DEFINITIONS:\n"
            elif feature_type == 'educational_games':
                header += f"🎲 EDUCATIONAL GAMES:\n"
            elif feature_type == 'examples_practice':
                header += f"💡 EXAMPLES & PRACTICE:\n"
            else:
                header += f"📚 {feature_display}:\n"
            
            return header + ai_response
        else:
            print("❌ AI returned empty response")
            return f"✅ Generating {feature_type.replace('_', ' ').upper()} for:\nGrade {grade} → {subject} → {chapter} → {skill_category}\n\n❌ Content generation failed - AI returned empty response. Please try again."
        
    except Exception as e:
        print(f"❌ AI content generation error: {str(e)}")
        return f"✅ **Exercise Selected: {exercise}**\n\n❌ Sorry, I encountered an error generating content. Please ensure AI services are properly configured with valid API keys."

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Handle audio uploads for voice messages"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.webm"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        # Save audio file
        audio_file.save(file_path)
        file_size = os.path.getsize(file_path)
        
        # For now, we'll save metadata without transcription
        # In a production app, you would integrate with a speech-to-text service
        metadata = save_file_metadata(file_id, 'voice_message.webm', 'audio/webm', file_size, 'audio')
        
        # Real transcription using OpenAI Whisper API with Urdu support
        transcription = transcribe_audio(file_path)
        metadata['extracted_text'] = transcription
        
        # Update metadata file
        metadata_path = os.path.join(META_DIR, f"{file_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Schedule cleanup
        cleanup_old_files()
        
        return jsonify({
            'audio_id': file_id,
            'filename': 'voice_message.webm',
            'size': file_size,
            'transcription': transcription
        })
        
    except Exception as e:
        print(f"Audio upload error: {e}")
        return jsonify({'error': 'Audio upload failed'}), 500

@app.route('/media/<file_id>')
def serve_media(file_id):
    """Serve uploaded media files (audio, images, documents)"""
    try:
        # Get metadata to determine file type and location
        metadata = get_file_metadata(file_id)
        if not metadata:
            return "File not found", 404
        
        file_type = metadata['type']
        file_path = None
        
        if file_type == 'audio':
            file_path = os.path.join(AUDIO_DIR, f"{file_id}.webm")
        elif file_type == 'image':
            # Find the actual image file
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                potential_path = os.path.join(IMAGES_DIR, f"{file_id}{ext}")
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
        elif file_type == 'document':
            # Find the actual document file
            for ext in ['.pdf', '.txt', '.docx']:
                potential_path = os.path.join(DOCS_DIR, f"{file_id}{ext}")
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
        
        if file_path and os.path.exists(file_path):
            return send_file(file_path)
        else:
            return "File not found", 404
            
    except Exception as e:
        print(f"Media serving error: {e}")
        return "File serving failed", 500

@app.route('/remove_file', methods=['POST'])
def remove_file():
    """Remove uploaded file"""
    try:
        file_id = request.json.get('file_id')
        if not file_id:
            return jsonify({'error': 'No file ID provided'}), 400
        
        # Get metadata
        metadata = get_file_metadata(file_id)
        if not metadata:
            return jsonify({'error': 'File not found'}), 404
        
        # Remove the actual file
        file_type = metadata['type']
        if file_type == 'audio':
            file_path = os.path.join(AUDIO_DIR, f"{file_id}.webm")
        elif file_type == 'image':
            # Find the actual file
            file_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                potential_path = os.path.join(IMAGES_DIR, f"{file_id}{ext}")
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
        elif file_type == 'document':
            # Find the actual file
            file_path = None
            for ext in ['.pdf', '.txt', '.docx']:
                potential_path = os.path.join(DOCS_DIR, f"{file_id}{ext}")
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
        
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove metadata file
        metadata_path = os.path.join(META_DIR, f"{file_id}.json")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"File removal error: {e}")
        return jsonify({'error': 'File removal failed'}), 500

if __name__ == '__main__':
    # Initialize upload directories
    init_upload_directories()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)