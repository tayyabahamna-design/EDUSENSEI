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

app = Flask(__name__)

# Require secure session secret in production
SESSION_SECRET = os.environ.get('SESSION_SECRET')
if not SESSION_SECRET:
    raise ValueError("SESSION_SECRET environment variable must be set for security")

app.secret_key = SESSION_SECRET
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12MB max file size

# Security configurations (adjusted for development)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # No JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

# Initialize AI client - prefer OpenAI for reliability
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# PostHog configuration
POSTHOG_KEY = os.environ.get('VITE_PUBLIC_POSTHOG_KEY', 'phc_ygiCdZb8vwOkLO5WIdGvdxzugrlGnaFxkW0F73sHyBF')
POSTHOG_HOST = os.environ.get('VITE_PUBLIC_POSTHOG_HOST', 'https://app.posthog.com')

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')

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
    """Get AI-powered response using OpenAI or Gemini with book-specific context"""
    
    # Build contextual information from session
    context_info = ""
    if session_context and session_context.get('curriculum_selection'):
        selection = session_context['curriculum_selection']
        grade = selection.get('grade', '')
        subject = selection.get('subject', '')
        book = selection.get('book', '')
        chapter = selection.get('chapter', '')
        topic = selection.get('topic', '')
        
        if book:
            context_info = f"""

CURRENT EDUCATIONAL CONTEXT:
- Grade: {grade}
- Subject: {subject}
- Textbook: {book}
- Chapter: {chapter if chapter else 'Not specified'}
- Topic: {topic if topic else 'Not specified'}

IMPORTANT: Base all educational content generation on this specific textbook and context. When creating lesson plans, activities, assessments, or other educational materials, reference this specific book and make the content appropriate for {grade} students studying {subject} from "{book}". If asked to create examples or definitions, make them relevant to the current chapter and topic context."""
    
    # Create system prompt based on conversation type
    if conversation_type == "teaching":
        system_prompt = """You are a helpful, knowledgeable, and conversational AI assistant. Follow these guidelines:

PERSONALITY & TONE:
- Be friendly, professional, and approachable
- Match the user's energy level and communication style
- Be concise but thorough - provide complete answers without being overly verbose
- Show enthusiasm when appropriate
- Be patient and understanding

RESPONSE STYLE:
- Always aim to be helpful and provide accurate information
- If you're uncertain about something, acknowledge it honestly
- Break down complex topics into easy-to-understand explanations
- Use examples when they help clarify your points
- Ask follow-up questions when you need more context

CAPABILITIES:
- Help with coding, writing, analysis, math, creative tasks, and general questions
- Provide step-by-step instructions when needed
- Offer multiple approaches or solutions when applicable
- Admit when something is outside your knowledge or capabilities

SAFETY & ETHICS:
- Prioritize user safety and well-being
- Decline requests for harmful, illegal, or unethical content
- Respect privacy and confidentiality
- Be honest and transparent

Remember: Your goal is to be genuinely helpful while maintaining a natural, conversational tone. Adapt your communication style to what works best for each user.""" + context_info
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
            print(f"OpenAI API error: {e}")
    
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
            print(f"Gemini API error: {e}")
    
    # Final fallback
    return get_general_guidance_fallback(user_message)

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
            'English': {'Grade 4 English': 'Grade 4.pdf'},
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

def get_auto_loaded_book_content(grade, subject):
    """Auto-load book content with chapters and exercises based on grade and subject"""
    predefined_books = get_predefined_books()
    
    # Get the appropriate book for this grade and subject
    grade_key = f'Grade {grade}'
    if grade_key not in predefined_books:
        return None
        
    subject_books = predefined_books[grade_key].get(subject, {})
    if not subject_books:
        return None
    
    # Get the first (and usually only) book for this subject
    book_title = list(subject_books.keys())[0]
    book_filename = subject_books[book_title]
    
    # Generate structured content as if extracted via OCR
    book_content = generate_book_structure(grade, subject, book_title)
    
    return {
        'title': book_title,
        'filename': book_filename,
        'grade': grade,
        'subject': subject,
        'chapters': book_content['chapters'],
        'total_chapters': len(book_content['chapters'])
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

def get_general_guidance_fallback(question):
    """Fallback guidance when AI services are not available"""
    question_lower = question.lower()
    
    # Provide helpful responses for common topics
    if any(word in question_lower for word in ['code', 'coding', 'programming', 'python', 'javascript', 'html', 'css']):
        return """I'd be happy to help with coding questions! While I'm currently unable to access my full capabilities, here are some general programming tips:

• Break down complex problems into smaller, manageable parts
• Use clear, descriptive variable and function names
• Comment your code to explain the "why," not just the "what"
• Test your code frequently with small inputs
• Don't be afraid to look up documentation and examples

For specific coding help, I recommend checking Stack Overflow, official documentation, or online coding communities."""
    
    elif any(word in question_lower for word in ['write', 'writing', 'essay', 'story', 'creative']):
        return """I'd love to help with your writing! Here are some general writing tips:

• Start with a clear outline or structure
• Write a compelling opening that hooks your reader
• Use specific details and examples to support your points
• Keep your audience in mind throughout
• Read your work aloud to catch awkward phrasing
• Don't worry about perfection in your first draft - focus on getting ideas down

What type of writing are you working on? I can provide more specific guidance once my services are fully available."""
    
    elif any(word in question_lower for word in ['math', 'mathematics', 'calculate', 'equation', 'problem']):
        return """I'm here to help with math! Some general problem-solving strategies:

• Read the problem carefully and identify what you're looking for
• Write down what information you have
• Consider what formulas or concepts might apply
• Work through simpler examples first
• Check your answer by substituting back or using estimation

What specific math topic are you working with? I'll be able to provide more detailed help once my full capabilities are restored."""
    
    else:
        # General helpful response
        return f"""Thanks for your question! I'm currently experiencing some technical difficulties, but I'm designed to help with a wide variety of topics including:

• Coding and programming
• Writing and creative tasks
• Math and analysis
• General questions and research
• Step-by-step problem solving

I aim to be helpful, accurate, and conversational in my responses. Once my services are fully restored, I'll be able to provide more detailed assistance with "{question}" and any other questions you might have.

Is there anything specific you'd like help with in the meantime?"""

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

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', posthog_key=POSTHOG_KEY, posthog_host=POSTHOG_HOST)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot messages with multimodal inputs"""
    # Guard against malformed requests
    if not request.json or 'message' not in request.json:
        return jsonify({'message': 'Invalid request format'})
    
    user_message = request.json.get('message', '').strip()
    file_ids = request.json.get('file_ids', [])
    audio_id = request.json.get('audio_id', None)
    
    # Handle Free Chat mode first - bypass all menu logic
    if session.get('selected_feature') == 'free_chat' and user_message.lower() not in ['menu', 'start', '← back to menu']:
        # Process uploaded files and generate AI response directly
        full_text = user_message
        
        # Add audio transcript if available
        if audio_id:
            audio_metadata = get_file_metadata(audio_id)
            if audio_metadata and 'extracted_text' in audio_metadata:
                audio_text = audio_metadata['extracted_text']
                if audio_text:
                    full_text = f"[Voice Message]: {audio_text}\n{full_text}".strip()
        
        # Process uploaded files
        for file_id in file_ids:
            file_metadata = get_file_metadata(file_id)
            if file_metadata:
                file_type = file_metadata['type']
                
                if file_type == 'image':
                    full_text = f"User uploaded an image. {full_text}".strip()
                    
                elif file_type == 'document' and 'extracted_text' in file_metadata:
                    extracted_content = file_metadata['extracted_text']
                    if extracted_content:
                        full_text = f"[Document Content]: {extracted_content}\n\n{full_text}".strip()
        
        # Get AI response directly with session context
        ai_response = get_ai_response(full_text, "general", session)
        return jsonify({'message': ai_response, 'is_markdown': True})
    
    # Handle special greetings and commands
    if user_message.lower() in ['hi', 'hello', 'hey', 'menu', 'start']:
        return jsonify({
            'message': '🌟 **السلام علیکم! Hello!** 🌟\n\nI\'m **U-DOST** 🤖✨ - Your friendly Pakistani teacher assistant! Ready to help you with curriculum-based educational content for grades 1-5.\n\n**Choose how I can help:**',
            'options': [
                '📚 Lesson Plans',
                '🎯 Teaching Strategies', 
                '🎲 Activities',
                '📖 Definitions',
                '📊 Assessment Tools',
                '🎮 Educational Games/Hooks',
                '📝 Examples & Practice',
                '💬 Free Chat'
            ],
            'show_menu': True
        })
    
    # Handle Free Chat selection from menu
    if user_message.lower() in ['💬 free chat', 'free chat']:
        session['selected_feature'] = 'free_chat'
        # Clear any curriculum selection to avoid conflicts
        if 'curriculum_selection' in session:
            del session['curriculum_selection']
        session.modified = True
        return jsonify({
            'message': '💬 **Free Chat Mode Activated!** \n\nI\'m ready to help you with anything! Ask me about coding, writing, analysis, creative tasks, or any questions you have. Let\'s have a natural conversation! 🚀',
            'show_menu': False
        })
    
    # Handle main menu options - all lead to grade selection
    menu_options = {
        '📚 lesson plans': 'lesson_plans',
        'lesson plans': 'lesson_plans',
        '🎯 teaching strategies': 'teaching_strategies', 
        'teaching strategies': 'teaching_strategies',
        '🎲 activities': 'activities',
        'activities': 'activities',
        '📖 definitions': 'definitions',
        'definitions': 'definitions',
        '📊 assessment tools': 'assessment_tools',
        'assessment tools': 'assessment_tools',
        '🎮 educational games/hooks': 'educational_games',
        'educational games/hooks': 'educational_games',
        'educational games': 'educational_games',
        '📝 examples & practice': 'examples_practice',
        'examples & practice': 'examples_practice',
        'examples and practice': 'examples_practice'
    }
    
    if user_message.lower() in menu_options:
        session['selected_feature'] = menu_options[user_message.lower()]
        return jsonify({
            'message': f'**{user_message}** 📖\n\nFirst, select your grade level:',
            'options': [
                '1️⃣ Grade 1',
                '2️⃣ Grade 2', 
                '3️⃣ Grade 3',
                '4️⃣ Grade 4',
                '5️⃣ Grade 5',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    # Handle "Specific Topic Assessment" option (legacy - for backward compatibility)
    if user_message.lower() in ['specific topic assessment', '📋 specific topic assessment']:
        # Redirect to new U-DOST flow
        return jsonify({
            'message': 'Please use the new enhanced U-DOST system! Select from the main menu options.',
            'options': ['← Back to Menu'],
            'show_menu': True
        })

    # Handle individual assessment types - now with curriculum context
    if 'curriculum_selection' in session and 'topic' in session['curriculum_selection']:
        current_grade = session['curriculum_selection']['grade']
        current_subject = session['curriculum_selection']['subject']
        current_chapter = session['curriculum_selection'].get('chapter', 'Chapter 1')
        current_topic = session['curriculum_selection']['topic']
        
        if user_message.lower() in ['quick q&a', '❓ quick q&a', 'qna']:
            return generate_curriculum_specific_assessment('qna', current_grade, current_subject, current_chapter, current_topic)
        
        if user_message.lower() in ['multiple choice questions (mcq)', '🔤 multiple choice questions (mcq)', 'mcq', 'multiple choice']:
            return generate_curriculum_specific_assessment('mcq', current_grade, current_subject, current_chapter, current_topic)
        
        if user_message.lower() in ['short comprehension questions', '📖 short comprehension questions', 'comprehension']:
            return generate_curriculum_specific_assessment('comprehension', current_grade, current_subject, current_chapter, current_topic)
        
        if user_message.lower() in ['thumbs up/down', '👍👎 thumbs up/down', 'thumbs']:
            return generate_curriculum_specific_assessment('thumbs', current_grade, current_subject, current_chapter, current_topic)
        
        if user_message.lower() in ['true/false statements', '📝 true/false statements', 'true false']:
            return generate_curriculum_specific_assessment('statements', current_grade, current_subject, current_chapter, current_topic)
        
        if user_message.lower() in ['fill in the blanks', '✏️ fill in the blanks', 'fill blanks']:
            return generate_curriculum_specific_assessment('fill-blanks', current_grade, current_subject, current_chapter, current_topic)
        
        if user_message.lower() in ['exit tickets', '🎫 exit tickets', 'exit ticket']:
            return generate_curriculum_specific_assessment('exit-ticket', current_grade, current_subject, current_chapter, current_topic)
    
    # Fallback to generic assessment types if no curriculum context
    if user_message.lower() in ['quick q&a', '❓ quick q&a', 'qna']:
        return generate_assessment_response('qna')
    
    if user_message.lower() in ['multiple choice questions (mcq)', '🔤 multiple choice questions (mcq)', 'mcq', 'multiple choice']:
        return generate_assessment_response('mcq')
    
    if user_message.lower() in ['short comprehension questions', '📖 short comprehension questions', 'comprehension']:
        return generate_assessment_response('comprehension')
    
    if user_message.lower() in ['thumbs up/down', '👍👎 thumbs up/down', 'thumbs']:
        return generate_assessment_response('thumbs')
    
    if user_message.lower() in ['true/false statements', '📝 true/false statements', 'true false']:
        return generate_assessment_response('statements')
    
    if user_message.lower() in ['fill in the blanks', '✏️ fill in the blanks', 'fill blanks']:
        return generate_assessment_response('fill-blanks')
    
    if user_message.lower() in ['exit tickets', '🎫 exit tickets', 'exit ticket']:
        return generate_assessment_response('exit-ticket')
    
    # Handle grade selection
    grade_options = {
        '1️⃣ grade 1': 1, 'grade 1': 1,
        '2️⃣ grade 2': 2, 'grade 2': 2,
        '3️⃣ grade 3': 3, 'grade 3': 3,
        '4️⃣ grade 4': 4, 'grade 4': 4,
        '5️⃣ grade 5': 5, 'grade 5': 5
    }
    
    if user_message.lower() in grade_options and 'selected_feature' in session:
        grade = grade_options[user_message.lower()]
        if 'curriculum_selection' not in session:
            session['curriculum_selection'] = {}
        session['curriculum_selection']['grade'] = grade
        session.modified = True
        
        # Pakistani curriculum subjects for grades 1-5
        subjects = ['English', 'Urdu', 'Mathematics', 'Science', 'Islamiyat', 'Social Studies', 'General Knowledge']
        
        return jsonify({
            'message': f'**Grade {grade}** 📚\n\nSelect your subject:',
            'options': [f'📖 {subject}' for subject in subjects] + ['🔄 Change Grade', '← Back to Menu'],
            'show_menu': True
        })
    
    # Handle subject selection with auto-loading book content
    if 'curriculum_selection' in session and 'grade' in session['curriculum_selection'] and 'selected_feature' in session:
        subjects = ['english', 'urdu', 'mathematics', 'science', 'islamiyat', 'social studies', 'general knowledge']
        subject_message = user_message.lower().replace('📖 ', '')
        
        if subject_message in subjects:
            session['curriculum_selection']['subject'] = subject_message.title()
            session.modified = True
            grade = session['curriculum_selection']['grade']
            subject = session['curriculum_selection']['subject']
            
            # AUTO-LOAD BOOK based on grade + subject
            book_content = get_auto_loaded_book_content(grade, subject)
            
            if book_content:
                # Store auto-loaded book info in session
                session['curriculum_selection']['book'] = book_content['title']
                session['curriculum_selection']['book_filename'] = book_content['filename']
                session.modified = True
                
                # Display auto-loaded book with chapters
                chapter_options = list(book_content['chapters'].keys())
                
                return jsonify({
                    'message': f'**📖 {book_content["title"]}** \n\n*Auto-loaded from Pakistani curriculum*\n\n**Available Chapters:** ({len(chapter_options)} chapters found)',
                    'options': [f'📄 {chapter}' for chapter in chapter_options[:10]] + (['📚 Show More Chapters'] if len(chapter_options) > 10 else []) + ['🔄 Change Subject', '← Back to Menu'],
                    'show_menu': True
                })
            else:
                # Fallback if no predefined book found
                return jsonify({
                    'message': f'**Grade {grade} - {subject}** 📚\n\nNo curriculum book available for this combination. Please try a different subject.',
                    'options': ['🔄 Change Subject', '🔄 Change Grade', '← Back to Menu'],
                    'show_menu': True
                })
    
    # Handle chapter selection (from auto-loaded book)
    if ('curriculum_selection' in session and 'book' in session['curriculum_selection'] 
        and 'selected_feature' in session and user_message.lower().startswith('📄 ')):
        
        chapter_title = user_message[3:].strip()  # Remove emoji
        grade = session['curriculum_selection']['grade']
        subject = session['curriculum_selection']['subject']
        
        # Get auto-loaded book content again for chapter details
        book_content = get_auto_loaded_book_content(grade, subject)
        
        if book_content and chapter_title in book_content['chapters']:
            # Store chapter selection
            session['curriculum_selection']['chapter'] = chapter_title
            session.modified = True
            
            # Get exercises for this chapter, categorized by skill type
            chapter_exercises = book_content['chapters'][chapter_title]
            
            # Display exercise categories
            exercise_categories = list(chapter_exercises.keys())
            
            return jsonify({
                'message': f'**📖 {book_content["title"]}**\n**📄 {chapter_title}**\n\nSelect exercise category:',
                'options': [f'🎯 {category} ({len(chapter_exercises[category])} exercises)' for category in exercise_categories] + ['📚 Show All Exercises', '🔄 Change Chapter', '← Back to Menu'],
                'show_menu': True
            })
        else:
            return jsonify({
                'message': 'Chapter not found. Please select a valid chapter.',
                'options': ['🔄 Change Chapter', '← Back to Menu'],
                'show_menu': True
            })
    
    # Handle exercise category selection
    if ('curriculum_selection' in session and 'chapter' in session['curriculum_selection'] 
        and 'selected_feature' in session and user_message.lower().startswith('🎯 ')):
        
        category_with_count = user_message[3:].strip()  # Remove emoji
        # Extract category name (before the count in parentheses)
        category = category_with_count.split(' (')[0]
        
        grade = session['curriculum_selection']['grade']
        subject = session['curriculum_selection']['subject']
        chapter = session['curriculum_selection']['chapter']
        
        # Store exercise category selection
        session['curriculum_selection']['skill_category'] = category
        session.modified = True
        
        # Get specific exercises for this category
        book_content = get_auto_loaded_book_content(grade, subject)
        if book_content and chapter in book_content['chapters']:
            exercises = book_content['chapters'][chapter].get(category, [])
            
            if exercises:
                # Display individual exercises
                exercise_options = [f'✏️ {exercise["title"]}' for exercise in exercises[:8]]  # Limit to 8 exercises
                
                return jsonify({
                    'message': f'**📖 {book_content["title"]}**\n**📄 {chapter}**\n**🎯 {category} Exercises**\n\nSelect an exercise to generate content:',
                    'options': exercise_options + (['📝 Show More Exercises'] if len(exercises) > 8 else []) + [f'🎯 Generate All {category} Content', '🔄 Change Category', '← Back to Menu'],
                    'show_menu': True
                })
            else:
                return jsonify({
                    'message': f'No {category} exercises found for this chapter.',
                    'options': ['🔄 Change Category', '🔄 Change Chapter', '← Back to Menu'],
                    'show_menu': True
                })
        else:
            return jsonify({
                'message': 'Chapter content not found.',
                'options': ['🔄 Change Chapter', '← Back to Menu'],
                'show_menu': True
            })
    
    # Handle individual exercise selection and content generation
    if ('curriculum_selection' in session and 'skill_category' in session['curriculum_selection'] 
        and 'selected_feature' in session and user_message.lower().startswith('✏️ ')):
        
        exercise_title = user_message[3:].strip()  # Remove emoji
        
        # Store exercise selection
        session['curriculum_selection']['exercise'] = exercise_title
        session.modified = True
        
        # Generate AI content using the complete curriculum context
        curriculum_selection = session['curriculum_selection']
        feature_type = session['selected_feature']
        
        # Generate curriculum-specific content
        ai_content = generate_udost_content(feature_type, curriculum_selection)
        
        return jsonify({
            'message': ai_content,
            'is_markdown': True,
            'options': ['🔄 Try Different Exercise', '🎯 Change Category', '📄 Change Chapter', '← Back to Menu'],
            'show_menu': True
        })
    
    # Handle "Generate All Category Content" option
    if ('curriculum_selection' in session and 'skill_category' in session['curriculum_selection'] 
        and user_message.lower().startswith('🎯 generate all ')):
        
        curriculum_selection = session['curriculum_selection']
        feature_type = session['selected_feature']
        category = curriculum_selection['skill_category']
        
        # Generate comprehensive content for the entire category
        curriculum_selection['exercise'] = f'All {category} exercises'
        ai_content = generate_udost_content(feature_type, curriculum_selection)
        
        return jsonify({
            'message': ai_content,
            'is_markdown': True,
            'options': ['🎯 Try Different Category', '📄 Change Chapter', '🔄 Change Subject', '← Back to Menu'],
            'show_menu': True
        })
        
    # Legacy skill category selection handler (keeping for backward compatibility)
    if ('curriculum_selection' in session and 'chapter_number' in session['curriculum_selection']
        and 'selected_feature' in session and user_message.lower().startswith('🎯 ')):
        
        skill_category = user_message[3:].strip()  # Remove emoji
        session['curriculum_selection']['skill_category'] = skill_category
        session.modified = True
        
        # Generate content based on selected feature and curriculum context
        return generate_udost_content(session['selected_feature'], session['curriculum_selection'], session)
    
    if user_message.lower() in ['← back to menu', 'back to menu', 'menu']:
        # Clear all session data when returning to main menu
        session.pop('curriculum_selection', None)
        session.pop('selected_feature', None)
        return jsonify({
            'message': '🌟 **السلام علیکم! Hello!** 🌟\n\nI\'m **U-DOST** 🤖✨ - Your friendly Pakistani teacher assistant! Ready to help you with curriculum-based educational content for grades 1-5.\n\n**Choose how I can help:**',
            'options': [
                '📚 Lesson Plans',
                '🎯 Teaching Strategies', 
                '🎲 Activities',
                '📖 Definitions',
                '📊 Assessment Tools',
                '🎮 Educational Games/Hooks',
                '📝 Examples & Practice',
                '💬 Free Chat'
            ],
            'show_menu': True
        })
    
    if user_message.lower() in ['📊 more assessment types', 'more assessment types']:
        return jsonify({
            'message': '📊 Choose your assessment type! Pick the perfect question format for your classroom:',
            'options': [
                '❓ Quick Q&A',
                '🔤 Multiple Choice Questions (MCQ)',
                '📖 Short Comprehension Questions', 
                '👍👎 Thumbs Up/Down',
                '📝 True/False Statements',
                '✏️ Fill in the Blanks',
                '🎫 Exit Tickets',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    # Handle main menu options - start with curriculum selection
    if user_message.lower() in ['lesson planning help', '📝 lesson planning help']:
        session['selected_feature'] = 'lesson_planning'
        return jsonify({
            'message': '📝 **Lesson Planning Help** - First, select your grade level:',
            'options': [
                '1️⃣ Grade 1',
                '2️⃣ Grade 2', 
                '3️⃣ Grade 3',
                '4️⃣ Grade 4',
                '5️⃣ Grade 5',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    if user_message.lower() in ['fun classroom activities', '🎮 fun classroom activities']:
        session['selected_feature'] = 'activities'
        return jsonify({
            'message': '🎮 **Fun Classroom Activities** - First, select your grade level:',
            'options': [
                '1️⃣ Grade 1',
                '2️⃣ Grade 2', 
                '3️⃣ Grade 3',
                '4️⃣ Grade 4',
                '5️⃣ Grade 5',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    if user_message.lower() in ['teaching tips & advice', '💡 teaching tips & advice', 'teaching tips']:
        session['selected_feature'] = 'teaching_tips'
        return jsonify({
            'message': '💡 **Teaching Tips & Advice** - First, select your grade level:',
            'options': [
                '1️⃣ Grade 1',
                '2️⃣ Grade 2', 
                '3️⃣ Grade 3',
                '4️⃣ Grade 4',
                '5️⃣ Grade 5',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    # Handle curriculum navigation - only if a feature is selected
    if 'selected_feature' in session:
        curriculum_data = generate_curriculum_data()
        
        # Store selection in session for navigation with proper persistence
        if 'curriculum_selection' not in session:
            session['curriculum_selection'] = {}
            session.modified = True
        
        # Grade 1-5 selections
        for grade_num in range(1, 6):
            grade_text = f'grade {grade_num}'
            grade_emoji = f'{grade_num}️⃣ grade {grade_num}'
            
            if user_message.lower() in [grade_text, grade_emoji.lower()]:
                curriculum_selection = session.get('curriculum_selection', {})
                curriculum_selection['grade'] = f'Grade {grade_num}'
                session['curriculum_selection'] = curriculum_selection
                session.modified = True
                subjects = list(curriculum_data[f'Grade {grade_num}'].keys())
                feature_name = {
                    'lesson_planning': 'Lesson Planning Help',
                    'assessment': 'Assessment',
                    'activities': 'Fun Classroom Activities',
                    'teaching_tips': 'Teaching Tips & Advice'
                }.get(session['selected_feature'], 'Selected Feature')
                
                return jsonify({
                    'message': f'📚 **{feature_name} - Grade {grade_num} Subjects** - Choose a subject:',
                    'options': [f'📖 {subject}' for subject in subjects] + ['🔄 Change Grade', '← Back to Menu'],
                    'show_menu': True
                })
        
        # Handle Subject selections
        if 'grade' in session.get('curriculum_selection', {}):
            current_grade = session['curriculum_selection']['grade']
            subjects = list(curriculum_data[current_grade].keys())
            
            for subject in subjects:
                subject_text = subject.lower()
                subject_emoji = f'📖 {subject}'.lower()
                
                if user_message.lower() in [subject_text, subject_emoji.lower()]:
                    curriculum_selection = session.get('curriculum_selection', {})
                    curriculum_selection['subject'] = subject
                    session['curriculum_selection'] = curriculum_selection
                    session.modified = True
                    chapters = list(curriculum_data[current_grade][subject].keys())
                    feature_name = {
                        'lesson_planning': 'Lesson Planning Help',
                        'assessment': 'Assessment',
                        'activities': 'Fun Classroom Activities',
                        'teaching_tips': 'Teaching Tips & Advice'
                    }.get(session['selected_feature'], 'Selected Feature')
                    
                    # For Assessment feature, show assessment types directly after subject selection
                    if session.get('selected_feature') == 'assessment':
                        # Set generic values for quick assessment
                        chapters = list(curriculum_data[current_grade][subject].keys())
                        curriculum_selection['chapter'] = chapters[0] if chapters else 'Chapter 1'
                        curriculum_selection['topic'] = f'All {subject} Topics'
                        session['curriculum_selection'] = curriculum_selection
                        session.modified = True
                        
                        return jsonify({
                            'message': f'📊 **Assessment Types for {current_grade} - {subject}**\n\nChoose your assessment type:',
                            'options': [
                                '❓ Quick Q&A',
                                '🔤 Multiple Choice Questions (MCQ)',
                                '📖 Short Comprehension Questions',
                                '👍👎 Thumbs Up/Down',
                                '📝 True/False Statements',
                                '✏️ Fill in the Blanks',
                                '🎫 Exit Tickets',
                                '📋 Specific Topic Assessment',
                                '🔄 Change Subject',
                                '← Back to Menu'
                            ],
                            'show_menu': True
                        })
                    else:
                        # Get predefined books for this grade and subject
                        predefined_books = get_predefined_books()
                        available_books = predefined_books.get(current_grade, {}).get(subject, {})
                        
                        if available_books:
                            return jsonify({
                                'message': f'📚 **{feature_name} - {current_grade} - {subject}** - Choose a textbook:',
                                'options': [f'📖 {book}' for book in available_books.keys()] + ['🔄 Change Subject', '← Back to Menu'],
                                'show_menu': True
                            })
                        else:
                            # Fallback if no books available for this subject
                            return jsonify({
                                'message': f'📚 **{current_grade} - {subject}** - No textbooks available yet for this subject.',
                                'options': ['🔄 Change Subject', '← Back to Menu'],
                                'show_menu': True
                            })
        
        # Handle Book selections
        if 'grade' in session.get('curriculum_selection', {}) and 'subject' in session.get('curriculum_selection', {}):
            current_grade = session['curriculum_selection']['grade']
            current_subject = session['curriculum_selection']['subject']
            predefined_books = get_predefined_books()
            available_books = predefined_books.get(current_grade, {}).get(current_subject, {})
            
            for book_title in available_books.keys():
                book_text = book_title.lower()
                book_emoji = f'📖 {book_title}'.lower()
                
                if user_message.lower() in [book_text, book_emoji.lower()]:
                    curriculum_selection = session.get('curriculum_selection', {})
                    curriculum_selection['book'] = book_title
                    session['curriculum_selection'] = curriculum_selection
                    session.modified = True
                    
                    # Now show chapters from the curriculum data
                    chapters = list(curriculum_data[current_grade][current_subject].keys())
                    feature_name = {
                        'lesson_planning': 'Lesson Planning Help',
                        'assessment': 'Assessment',
                        'activities': 'Fun Classroom Activities',
                        'teaching_tips': 'Teaching Tips & Advice'
                    }.get(session['selected_feature'], 'Selected Feature')
                    
                    return jsonify({
                        'message': f'📖 **{feature_name} - {book_title}** - Choose a chapter:',
                        'options': [f'📄 {chapter}' for chapter in chapters] + ['🔄 Change Book', '← Back to Menu'],
                        'show_menu': True
                    })

        # Handle Chapter selections  
        if 'grade' in session.get('curriculum_selection', {}) and 'subject' in session.get('curriculum_selection', {}) and 'book' in session.get('curriculum_selection', {}):
            current_grade = session['curriculum_selection']['grade']
            current_subject = session['curriculum_selection']['subject']
            chapters = list(curriculum_data[current_grade][current_subject].keys())
            
            for chapter in chapters:
                chapter_text = chapter.lower()
                chapter_emoji = f'📄 {chapter}'.lower()
                
                if user_message.lower() in [chapter_text, chapter_emoji.lower()]:
                    curriculum_selection = session.get('curriculum_selection', {})
                    curriculum_selection['chapter'] = chapter
                    session['curriculum_selection'] = curriculum_selection
                    session.modified = True
                    topics = curriculum_data[current_grade][current_subject][chapter]
                    feature_name = {
                        'lesson_planning': 'Lesson Planning Help',
                        'assessment': 'Assessment',
                        'activities': 'Fun Classroom Activities',
                        'teaching_tips': 'Teaching Tips & Advice'
                    }.get(session['selected_feature'], 'Selected Feature')
                    
                    return jsonify({
                        'message': f'📝 **{feature_name} - {current_grade} - {current_subject}** \n**{chapter}** - Choose a topic:',
                        'options': [f'✏️ {topic}' for topic in topics] + ['🔄 Change Chapter', '← Back to Menu'],
                        'show_menu': True
                    })
        
        # Handle Topic selections
        if all(key in session.get('curriculum_selection', {}) for key in ['grade', 'subject', 'chapter']):
            current_grade = session['curriculum_selection']['grade']
            current_subject = session['curriculum_selection']['subject'] 
            current_chapter = session['curriculum_selection']['chapter']
            topics = curriculum_data[current_grade][current_subject][current_chapter]
            
            for topic in topics:
                topic_text = topic.lower()
                topic_emoji = f'✏️ {topic}'.lower()
                
                if user_message.lower() in [topic_text, topic_emoji]:
                    curriculum_selection = session.get('curriculum_selection', {})
                    curriculum_selection['topic'] = topic
                    session['curriculum_selection'] = curriculum_selection
                    session.modified = True
                    # Directly proceed to the selected feature instead of showing action menu
                    selected_feature = session.get('selected_feature')
                    
                    if selected_feature == 'lesson_planning':
                        return generate_curriculum_lesson_plan(current_grade, current_subject, current_chapter, topic)
                    elif selected_feature == 'assessment':
                        return generate_curriculum_assessment_types(current_grade, current_subject, current_chapter, topic)
                    elif selected_feature == 'activities':
                        return generate_curriculum_activities(current_grade, current_subject, current_chapter, topic)
                    elif selected_feature == 'teaching_tips':
                        return generate_curriculum_tips(current_grade, current_subject, current_chapter, topic)
                    else:
                        # Fallback to action menu if no feature selected
                        return jsonify({
                            'message': f'''🎯 **Selected Topic:**
**Grade:** {current_grade}
**Subject:** {current_subject}
**Chapter:** {current_chapter}
**Topic:** {topic}

What would you like me to create for this topic?''',
                            'options': [
                                '📝 Generate Lesson Plan',
                                '📊 Create Assessment Questions',
                                '🎮 Suggest Fun Activities',
                                '💡 Teaching Tips for this Topic',
                                '🔄 Choose Different Topic',
                                '← Back to Menu'
                            ],
                            'show_menu': True
                        })
    
    # Handle curriculum action selections based on selected feature (legacy fallback)
    if 'topic' in session.get('curriculum_selection', {}) and 'selected_feature' in session:
        current_grade = session['curriculum_selection']['grade']
        current_subject = session['curriculum_selection']['subject']
        current_chapter = session['curriculum_selection']['chapter']
        current_topic = session['curriculum_selection']['topic']
        selected_feature = session['selected_feature']
        
        if selected_feature == 'lesson_planning':
            return generate_curriculum_lesson_plan(current_grade, current_subject, current_chapter, current_topic)
        
        elif selected_feature == 'assessment':
            return generate_curriculum_assessment_types(current_grade, current_subject, current_chapter, current_topic)
        
        elif selected_feature == 'activities':
            return generate_curriculum_activities(current_grade, current_subject, current_chapter, current_topic)
        
        elif selected_feature == 'teaching_tips':
            return generate_curriculum_tips(current_grade, current_subject, current_chapter, current_topic)
    
    # Handle navigation options
    if user_message.lower() in ['🔄 change grade', 'change grade']:
        session['curriculum_selection'] = {}
        return jsonify({
            'message': '📖 **Curriculum Navigator** - Choose your grade level to explore subjects, chapters, and topics!',
            'options': [
                '1️⃣ Grade 1',
                '2️⃣ Grade 2', 
                '3️⃣ Grade 3',
                '4️⃣ Grade 4',
                '5️⃣ Grade 5',
                '← Back to Menu'
            ],
            'show_menu': True
        })
    
    if user_message.lower() in ['🔄 change subject', 'change subject'] and 'grade' in session.get('curriculum_selection', {}):
        # Keep grade, reset others
        grade = session['curriculum_selection']['grade']
        session['curriculum_selection'] = {'grade': grade}
        subjects = list(curriculum_data[grade].keys())
        return jsonify({
            'message': f'📚 **{grade} Subjects** - Choose a subject to explore chapters and topics:',
            'options': [f'📖 {subject}' for subject in subjects] + ['🔄 Change Grade', '← Back to Menu'],
            'show_menu': True
        })
    
    if user_message.lower() in ['🔄 change book', 'change book'] and 'subject' in session.get('curriculum_selection', {}):
        # Keep grade and subject, reset book and others
        grade = session['curriculum_selection']['grade']
        subject = session['curriculum_selection']['subject']
        session['curriculum_selection'] = {'grade': grade, 'subject': subject}
        
        # Get predefined books for this grade and subject
        predefined_books = get_predefined_books()
        available_books = predefined_books.get(grade, {}).get(subject, {})
        
        feature_name = {
            'lesson_planning': 'Lesson Planning Help',
            'assessment': 'Assessment', 
            'activities': 'Fun Classroom Activities',
            'teaching_tips': 'Teaching Tips & Advice'
        }.get(session['selected_feature'], 'Selected Feature')
        
        if available_books:
            return jsonify({
                'message': f'📚 **{feature_name} - {grade} - {subject}** - Choose a textbook:',
                'options': [f'📖 {book}' for book in available_books.keys()] + ['🔄 Change Subject', '← Back to Menu'],
                'show_menu': True
            })
        else:
            return jsonify({
                'message': f'📚 **{grade} - {subject}** - No textbooks available yet for this subject.',
                'options': ['🔄 Change Subject', '← Back to Menu'],
                'show_menu': True
            })
    
    if user_message.lower() in ['🔄 change chapter', 'change chapter'] and 'book' in session.get('curriculum_selection', {}):
        # Keep grade, subject, and book, reset chapter and others
        grade = session['curriculum_selection']['grade']
        subject = session['curriculum_selection']['subject']  
        book = session['curriculum_selection']['book']
        session['curriculum_selection'] = {'grade': grade, 'subject': subject, 'book': book}
        session.modified = True
        
        # Get chapters from auto-loaded book
        book_content = get_auto_loaded_book_content(grade, subject)
        if book_content:
            chapters = list(book_content['chapters'].keys())
            feature_name = {
                'lesson_planning': 'Lesson Planning Help',
                'assessment': 'Assessment',
                'activities': 'Fun Classroom Activities',
                'teaching_tips': 'Teaching Tips & Advice'
            }.get(session['selected_feature'], 'Selected Feature')
            
            return jsonify({
                'message': f'📖 **{feature_name} - {book}** - Choose a chapter:',
                'options': [f'📄 {chapter}' for chapter in chapters[:10]] + (['📚 Show More Chapters'] if len(chapters) > 10 else []) + ['🔄 Change Subject', '← Back to Menu'],
                'show_menu': True
            })
        else:
            return jsonify({
                'message': 'Book content not found.',
                'options': ['🔄 Change Subject', '← Back to Menu'],
                'show_menu': True
            })
    
    if user_message.lower() in ['🔄 change category', 'change category', '🎯 try different category', 'try different category'] and 'chapter' in session.get('curriculum_selection', {}):
        # Keep grade, subject, book and chapter, reset skill category
        grade = session['curriculum_selection']['grade']
        subject = session['curriculum_selection']['subject']
        book = session['curriculum_selection']['book']
        chapter = session['curriculum_selection']['chapter']
        session['curriculum_selection'] = {'grade': grade, 'subject': subject, 'book': book, 'chapter': chapter}
        session.modified = True
        
        # Get exercise categories for this chapter
        book_content = get_auto_loaded_book_content(grade, subject)
        if book_content and chapter in book_content['chapters']:
            chapter_exercises = book_content['chapters'][chapter]
            exercise_categories = list(chapter_exercises.keys())
            
            return jsonify({
                'message': f'**📖 {book}**\n**📄 {chapter}**\n\nSelect exercise category:',
                'options': [f'🎯 {category} ({len(chapter_exercises[category])} exercises)' for category in exercise_categories] + ['📚 Show All Exercises', '🔄 Change Chapter', '← Back to Menu'],
                'show_menu': True
            })
        else:
            return jsonify({
                'message': 'Chapter content not found.',
                'options': ['🔄 Change Chapter', '← Back to Menu'],
                'show_menu': True
            })
    
    # Show All Exercises handler
    if user_message.lower() in ['📚 show all exercises', 'show all exercises'] and 'chapter' in session.get('curriculum_selection', {}):
        grade = session['curriculum_selection']['grade']
        subject = session['curriculum_selection']['subject'] 
        chapter = session['curriculum_selection']['chapter']
        
        book_content = get_auto_loaded_book_content(grade, subject)
        if book_content and chapter in book_content['chapters']:
            all_exercises = book_content['chapters'][chapter]
            
            # Create a comprehensive overview of all exercises
            exercise_overview = f'**📖 {book_content["title"]}**\n**📄 {chapter}**\n\n**All Available Exercises:**\n\n'
            
            for category, exercises in all_exercises.items():
                exercise_overview += f'**🎯 {category}** ({len(exercises)} exercises)\n'
                for i, exercise in enumerate(exercises[:3], 1):  # Show first 3 per category
                    exercise_overview += f'{i}. {exercise["title"]}\n'
                if len(exercises) > 3:
                    exercise_overview += f'   ... and {len(exercises) - 3} more\n'
                exercise_overview += '\n'
            
            return jsonify({
                'message': exercise_overview,
                'is_markdown': True,
                'options': ['🎯 Select Category', '📄 Change Chapter', '🔄 Change Subject', '← Back to Menu'],
                'show_menu': True
            })
        else:
            return jsonify({
                'message': 'Chapter content not found.',
                'options': ['📄 Change Chapter', '← Back to Menu'],
                'show_menu': True
            })
            
    # Select Category handler (from Show All Exercises view)
    if user_message.lower() in ['🎯 select category', 'select category'] and 'chapter' in session.get('curriculum_selection', {}):
        grade = session['curriculum_selection']['grade']
        subject = session['curriculum_selection']['subject']
        book = session['curriculum_selection']['book']
        chapter = session['curriculum_selection']['chapter']
        
        # Get exercise categories for this chapter
        book_content = get_auto_loaded_book_content(grade, subject)
        if book_content and chapter in book_content['chapters']:
            chapter_exercises = book_content['chapters'][chapter]
            exercise_categories = list(chapter_exercises.keys())
            
            return jsonify({
                'message': f'**📖 {book}**\n**📄 {chapter}**\n\nSelect exercise category:',
                'options': [f'🎯 {category} ({len(chapter_exercises[category])} exercises)' for category in exercise_categories] + ['📚 Show All Exercises', '🔄 Change Chapter', '← Back to Menu'],
                'show_menu': True
            })
        else:
            return jsonify({
                'message': 'Chapter content not found.',
                'options': ['🔄 Change Chapter', '← Back to Menu'],
                'show_menu': True
            })
            
    # Legacy topic selection handler (keeping for backward compatibility)
    if user_message.lower() in ['🔄 choose different topic', 'choose different topic'] and 'chapter' in session.get('curriculum_selection', {}):
        # Keep grade, subject, and chapter, reset topic
        grade = session['curriculum_selection']['grade']
        subject = session['curriculum_selection']['subject']
        chapter = session['curriculum_selection']['chapter']
        session['curriculum_selection'] = {'grade': grade, 'subject': subject, 'chapter': chapter}
        topics = curriculum_data[grade][subject][chapter]
        return jsonify({
            'message': f'📝 **{grade} - {subject}** \n**{chapter}** - Choose a topic:',
            'options': [f'✏️ {topic}' for topic in topics] + ['🔄 Change Chapter', '← Back to Menu'],
            'show_menu': True
        })
    
    # Handle multimodal content
    content_parts = []
    full_text = user_message
    
    # Add audio transcript if available
    if audio_id:
        audio_metadata = get_file_metadata(audio_id)
        if audio_metadata and 'extracted_text' in audio_metadata:
            audio_text = audio_metadata['extracted_text']
            if audio_text:
                full_text = f"[Voice Message]: {audio_text}\n{full_text}".strip()
    
    # Process uploaded files
    for file_id in file_ids:
        file_metadata = get_file_metadata(file_id)
        if file_metadata:
            file_type = file_metadata['type']
            if file_type == 'image':
                # Add image to content parts for Gemini
                image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
                for ext in image_extensions:
                    image_path = os.path.join(IMAGES_DIR, f"{file_id}{ext}")
                    if os.path.exists(image_path):
                        try:
                            if gemini_model:  # Only for Gemini
                                image = Image.open(image_path)
                                content_parts.append(image)
                            full_text += f"\n[Image attached: {file_metadata['original_name']}]"
                            break
                        except Exception as e:
                            print(f"Error loading image {file_id}: {e}")
            
            elif file_type == 'document':
                # Add document text
                if 'extracted_text' in file_metadata:
                    doc_text = file_metadata['extracted_text']
                    if doc_text:
                        full_text += f"\n[Document: {file_metadata['original_name']}]\n{doc_text}"
    
    if not full_text and not content_parts:
        return jsonify({'message': 'Please provide a message or upload a file to chat with me!'})
    
    # Generate AI response
    try:
        if content_parts and gemini_model:
            # Multimodal content with images (Gemini only)
            response = gemini_model.generate_content([full_text] + content_parts)
            ai_response = response.text
        else:
            # Text-only content (works with OpenAI or Gemini) with session context
            ai_response = get_ai_response(full_text, "general", session)
    except Exception as e:
        print(f"Error generating AI response: {e}")
        ai_response = "I'm sorry, I encountered an error processing your request. Please try again."
    
    return jsonify({'message': ai_response})

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
    
    # Create context for AI
    context = f"""
    You are U-DOST, a friendly Pakistani teacher assistant. Generate content for:
    
    Grade: {grade}
    Subject: {subject}  
    Book: {book}
    Chapter: {chapter}
    Skill Focus: {skill_category}
    
    Pakistani Education Context: This is for Pakistani primary education (grades 1-5) following the local curriculum.
    Language: Provide content in English but include Urdu terms where appropriate for Pakistani context.
    """
    
    content_prompts = {
        'lesson_plans': f"""Create a detailed lesson plan for Grade {grade} {subject}, Chapter {chapter}, focusing on {skill_category}.

Include:
1. **Learning Objectives** (Clear, measurable goals)
2. **Materials Needed** (Textbook, whiteboard, etc.)
3. **Introduction** (5-10 minutes warm-up activity)
4. **Main Activity** (20-25 minutes structured learning)
5. **Practice Session** (10-15 minutes hands-on work)
6. **Assessment** (How to check understanding)
7. **Homework/Extension** (Optional follow-up activities)
8. **Pakistani Context** (Local examples, cultural references)

Make it engaging and age-appropriate for Grade {grade} students in Pakistan.""",

        'teaching_strategies': f"""Suggest 5-7 effective teaching strategies for Grade {grade} {subject}, Chapter {chapter}, {skill_category} focus.

Include:
1. **Interactive Methods** (Group work, pair activities)
2. **Visual Aids** (Charts, pictures, demonstrations) 
3. **Local Examples** (Pakistani context, familiar scenarios)
4. **Differentiated Approaches** (For different learning styles)
5. **Assessment Techniques** (Quick checks, formative assessment)
6. **Classroom Management Tips** (Keeping students engaged)
7. **Cultural Sensitivity** (Islamic values, Pakistani customs)

Make strategies practical and easy to implement.""",

        'activities': f"""Design 6 engaging activities for Grade {grade} {subject}, Chapter {chapter}, {skill_category}.

Provide activities for:
1. **Independent Work** (Individual practice)
2. **Group Activity** (Collaborative learning) 
3. **Assignment/Homework** (Take-home practice)
4. **Pair Work** (Partner activities)
5. **Creative Activity** (Arts, crafts, creative expression)
6. **Assessment Activity** (Fun way to check learning)

Each activity should include:
- Clear instructions
- Time required
- Materials needed
- Learning outcomes
- Pakistani cultural context where relevant""",

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
        ai_response = get_ai_response(prompt, "educational_content", session_data)
        
        return jsonify({
            'message': f"**📚 {feature_type.replace('_', ' ').title()} - Grade {grade} {subject}**\n\n" + ai_response,
            'is_markdown': True,
            'return_to_menu': True,
            'breadcrumb': f"Grade {grade} › {subject} › {book} › Chapter {chapter} › {skill_category}",
            'suggestions': [
                '🔄 Generate More Content',
                '📤 Save to Files', 
                '🎯 Change Skill Category',
                '📖 Change Chapter',
                '← Back to Menu'
            ]
        })
        
    except Exception as e:
        print(f"AI content generation error occurred")
        return jsonify({
            'message': '❌ Sorry, I encountered an error generating content. Please ensure AI services are properly configured with valid API keys.',
            'options': ['🔄 Try Again', '← Back to Menu'],
            'show_menu': True
        })

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