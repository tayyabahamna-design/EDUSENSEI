from flask import Flask, render_template, request, jsonify, session
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

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'ai-assistant-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12MB max file size

# Initialize AI client - prefer OpenAI for reliability
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Try OpenAI first, fallback to Gemini
openai_client = None
gemini_model = None

if OPENAI_API_KEY:
    try:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        openai_client = None

if not openai_client and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception:
        gemini_model = None

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

def get_ai_response(user_message, conversation_type="general"):
    """Get AI-powered response using OpenAI or Gemini"""
    # Create system prompt based on conversation type
    if conversation_type == "teaching":
        system_prompt = """You are a helpful AI teaching assistant for primary school teachers (grades 1-5). 
        You provide practical, actionable advice about classroom management, lesson planning, student engagement, 
        assessment, and parent communication. Keep responses friendly, supportive, and focused on elementary education. 
        Use bullet points and clear formatting when helpful."""
    else:
        system_prompt = """You are a friendly, helpful AI assistant for primary school teachers. 
        You can discuss teaching topics, answer general questions, and have natural conversations. 
        Be warm, supportive, and helpful while maintaining a professional but friendly tone."""
    
    # Try OpenAI first
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
    
    # Fallback to Gemini
    if gemini_model:
        try:
            full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
            response = gemini_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
    
    # Final fallback
    return get_teaching_guidance_fallback(user_message)

def get_teaching_guidance_fallback(question):
    """Fallback teaching guidance when AI is not available"""
    question_lower = question.lower()
    
    # Common teaching topics and responses
    if any(word in question_lower for word in ['classroom management', 'behavior', 'discipline']):
        return """**Classroom Management Tips:**
        
• Set clear, consistent rules and expectations from day one
• Use positive reinforcement more than negative consequences  
• Create engaging activities to prevent boredom-related issues
• Build relationships with students - they behave better for teachers they like
• Use non-verbal cues like hand signals for quiet redirection
• Have a calm, consistent response to disruptions

For persistent issues, involve parents and school counselors as partners in supporting the student."""
    
    elif any(word in question_lower for word in ['lesson plan', 'planning', 'curriculum']):
        return """**Lesson Planning Best Practices:**
        
• Start with clear learning objectives - what should students know/do by the end?
• Include a hook or engaging opening to capture attention
• Break content into 10-15 minute chunks for elementary students
• Plan interactive activities, not just lectures
• Include multiple ways to practice the skill (visual, auditory, kinesthetic)
• End with a quick assessment or summary
• Always have backup activities ready

Remember: Good planning prevents poor performance!"""
    
    elif any(word in question_lower for word in ['motivation', 'engage', 'interest']):
        return """**Student Engagement Strategies:**
        
• Connect lessons to students' real lives and interests
• Use games, movement, and hands-on activities
• Give students choices when possible (topics, seating, partners)
• Celebrate effort and improvement, not just achievement
• Use technology thoughtfully to enhance learning
• Break up long activities with brain breaks
• Tell stories and use humor appropriately

Engaged students learn better and cause fewer problems!"""
    
    else:
        # General teaching advice
        return f"""Thanks for your question about: "{question}"
        
**General Teaching Tips:**
        
• Build positive relationships with all students
• Keep lessons interactive and hands-on
• Use clear expectations and consistent routines
• Differentiate instruction for different learning styles
• Communicate regularly with parents
• Take care of yourself - you can't pour from an empty cup!

For specific guidance on this topic, I recommend consulting with your school's instructional coach, mentor teacher, or educational resources like Edutopia.org or your district's curriculum materials."""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot messages with multimodal inputs"""
    # Guard against malformed requests
    if not request.json or 'message' not in request.json:
        return jsonify({'message': 'Invalid request format'})
    
    user_message = request.json.get('message', '').strip()
    file_ids = request.json.get('file_ids', [])
    audio_id = request.json.get('audio_id', None)
    
    # Handle special greetings and commands
    if user_message.lower() in ['hi', 'hello', 'hey', 'menu', 'start']:
        return jsonify({
            'message': '🎉 Welcome! I\'m your AI Teaching Assistant! Here\'s how I can help you:',
            'options': [
                '📝 Lesson Planning Help',
                '🎮 Fun Classroom Activities', 
                '💡 Teaching Tips & Advice',
                '📚 Educational Resources',
                '❓ Answer Questions',
                '💬 Free Chat'
            ],
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
            # Text-only content (works with OpenAI or Gemini)
            ai_response = get_ai_response(full_text)
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
        
        # Mock transcription for demo (replace with actual speech-to-text service)
        mock_transcription = "Voice message uploaded successfully. (Transcription feature would be implemented with a speech-to-text service)"
        metadata['extracted_text'] = mock_transcription
        
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
            'transcription': mock_transcription
        })
        
    except Exception as e:
        print(f"Audio upload error: {e}")
        return jsonify({'error': 'Audio upload failed'}), 500

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