from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import json
import os
import uuid
from datetime import datetime, date, timedelta
import calendar
from werkzeug.utils import secure_filename
import requests
import re
import google.generativeai as genai
import mimetypes
import hashlib
from PIL import Image, ExifTags
from pypdf import PdfReader
from docx import Document

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'tutor-assistant-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12MB max file size

# Initialize Gemini client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

# Data file paths
DATA_DIR = 'data'
STUDENTS_FILE = os.path.join(DATA_DIR, 'students.json')
ATTENDANCE_FILE = os.path.join(DATA_DIR, 'attendance.json')
SCHEDULE_FILE = os.path.join(DATA_DIR, 'schedule.json')
GRADES_FILE = os.path.join(DATA_DIR, 'grades.json')
CLASSES_FILE = os.path.join(DATA_DIR, 'classes.json')

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

# Ensure data directory and files exist
def init_data_files():
    # Create data directory
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Create upload directories
    for upload_dir in [UPLOAD_DIR, AUDIO_DIR, IMAGES_DIR, DOCS_DIR, META_DIR]:
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
    
    # Initialize empty data files if they don't exist
    default_data = {
        STUDENTS_FILE: [],
        ATTENDANCE_FILE: {},
        SCHEDULE_FILE: {},
        GRADES_FILE: {},
        CLASSES_FILE: {}
    }
    
    for file_path, default_content in default_data.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(default_content, f)

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

# Load data from JSON files
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if 'attendance' in file_path or 'schedule' in file_path or 'grades' in file_path or 'classes' in file_path else []

# Save data to JSON files
def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Student profile helper functions
def create_student_profile(name, guardian_name="", guardian_phone="", address="", profile_picture="", gender="", class_id=""):
    """Create a new student profile object"""
    return {
        "id": f"student_{uuid.uuid4().hex[:8]}",
        "name": name.strip(),
        "guardian_name": guardian_name.strip(),
        "guardian_phone": guardian_phone.strip(),
        "address": address.strip(),
        "profile_picture": profile_picture,
        "gender": gender.strip(),
        "class_id": class_id.strip()
    }

def get_student_by_name(students, name):
    """Find a student by name from the students list"""
    for student in students:
        if isinstance(student, dict) and student.get('name') == name:
            return student
    return None

def get_student_by_id(students, student_id):
    """Find a student by ID from the students list"""
    for student in students:
        if isinstance(student, dict) and student.get('id') == student_id:
            return student
    return None

def get_student_names(students):
    """Extract list of student names from student objects"""
    names = []
    for student in students:
        if isinstance(student, dict):
            names.append(student.get('name', ''))
        else:
            # Handle legacy string format
            names.append(str(student))
    return names

def migrate_legacy_students(students):
    """Convert legacy string-based student list to object format"""
    migrated = []
    for i, student in enumerate(students):
        if isinstance(student, str):
            # Convert string to object
            migrated.append(create_student_profile(student))
        elif isinstance(student, dict):
            # Already in new format, ensure class_id exists
            if 'class_id' not in student:
                student['class_id'] = ""
            migrated.append(student)
    return migrated

# Class management helper functions
def create_class(grade, section):
    """Create a new class object"""
    class_id = f"grade_{grade}{section.lower()}"
    return {
        "id": class_id,
        "name": f"Grade {grade}{section.upper()}",
        "grade": int(grade),
        "section": section.upper()
    }

def get_all_classes():
    """Get all classes sorted by grade and section"""
    classes_data = load_data(CLASSES_FILE)
    classes = list(classes_data.values())
    # Sort by grade first, then by section
    classes.sort(key=lambda x: (x.get('grade', 0), x.get('section', '')))
    return classes

def get_class_by_id(class_id):
    """Get a specific class by ID"""
    classes_data = load_data(CLASSES_FILE)
    return classes_data.get(class_id)

def get_students_by_class(class_id):
    """Get all students belonging to a specific class"""
    students = load_data(STUDENTS_FILE)
    students = migrate_legacy_students(students)
    return [student for student in students if student.get('class_id') == class_id]

def get_unassigned_students():
    """Get all students not assigned to any class"""
    students = load_data(STUDENTS_FILE)
    students = migrate_legacy_students(students)
    return [student for student in students if not student.get('class_id')]

# Simulated AI functions
def simulate_ai_grading():
    """Simulate AI-powered grading system"""
    import random
    return random.randint(75, 100)

# Chatbot State Management
class ChatbotState:
    """Finite State Machine for chatbot navigation"""
    
    # State constants
    MAIN_MENU = 'main_menu'
    GRADE_SELECTION = 'grade_selection' 
    SUBJECT_SELECTION = 'subject_selection'
    TOPIC_INPUT = 'topic_input'
    LESSON_PLANNING = 'lesson_planning'
    ACTIVITIES = 'activities'
    DEFINITIONS = 'definitions'
    GENERAL_QUESTIONS = 'general_questions'
    FREE_CHAT = 'free_chat'
    
    # Menu options
    GRADES = ['1', '2', '3', '4', '5']
    SUBJECTS = {
        'English': ['Reading', 'Writing', 'Grammar', 'Vocabulary', 'Literature'],
        'Maths': ['Numbers', 'Addition', 'Subtraction', 'Multiplication', 'Division', 'Fractions', 'Geometry'],
        'Science': ['Plants', 'Animals', 'Weather', 'Earth', 'Space', 'Matter', 'Energy'],
        'Urdu': ['حروف تہجی', 'کلمات', 'جملے', 'کہانیاں', 'نظمیں']
    }
    
    ACTIVITIES_TYPES = {
        'Classroom Games': ['Word games', 'Math puzzles', 'Science experiments', 'Group activities'],
        'Group Work': ['Team projects', 'Discussion circles', 'Peer learning', 'Collaborative tasks'],
        'Individual Tasks': ['Worksheets', 'Reading assignments', 'Practice problems', 'Creative writing'],
        'Creative Projects': ['Art integration', 'Drama activities', 'Music lessons', 'Storytelling']
    }

def get_chatbot_state():
    """Get current chatbot state from session"""
    return session.get('chatbot_state', ChatbotState.MAIN_MENU)

def set_chatbot_state(state):
    """Set chatbot state in session"""
    session['chatbot_state'] = state

def get_session_data(key, default=None):
    """Get data from chatbot session"""
    return session.get(f'chatbot_{key}', default)

def set_session_data(key, value):
    """Set data in chatbot session"""
    session[f'chatbot_{key}'] = value

def clear_session_data():
    """Clear all chatbot session data"""
    keys_to_remove = [key for key in session.keys() if key.startswith('chatbot_')]
    for key in keys_to_remove:
        session.pop(key, None)

def generate_lesson_plan_with_ai(subject, grade, topic):
    """Generate detailed lesson plan using educational templates"""
    # Use structured lesson plan template for consistency
    return generate_simple_lesson_plan(subject, grade, topic)

def get_wikipedia_definition(term):
    """Get definition from Wikipedia API"""
    try:
        # First, search for the term
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + term.replace(' ', '_').replace('(', '%28').replace(')', '%29')
        
        response = requests.get(search_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Extract and clean the summary
            summary = data.get('extract', '')
            if summary:
                # Clean up the summary for elementary teachers
                summary = re.sub(r'\([^)]*\)', '', summary)  # Remove parenthetical notes
                summary = summary.strip()
                
                # Keep it concise for elementary level
                sentences = summary.split('. ')
                if len(sentences) > 3:
                    summary = '. '.join(sentences[:3]) + '.'
                
                return f"**{term.title()}**: {summary}\n\nThis information comes from Wikipedia and is suitable for elementary education."
            
    except Exception as e:
        pass
    
    # Fallback if Wikipedia fails
    return f"I'd be happy to help define '{term}' for you! For detailed definitions, I recommend checking educational resources like dictionaries, encyclopedia, or asking a librarian for age-appropriate explanations."

def get_ai_response(user_message, conversation_type="general"):
    """Get AI-powered response using Google Gemini"""
    if not gemini_model:
        return get_teaching_guidance_fallback(user_message)
    
    try:
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
        
        # Combine system prompt with user message for Gemini
        full_prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        response = gemini_model.generate_content(full_prompt)
        
        return response.text
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return get_teaching_guidance_fallback(user_message)

def get_teaching_guidance_fallback(question):
    """Fallback teaching guidance when OpenAI is not available"""
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
    
    elif any(word in question_lower for word in ['parent', 'communication', 'family']):
        return """**Parent Communication Tips:**
        
• Contact parents with GOOD news first, before any problems arise
• Use clear, jargon-free language
• Be specific about what's happening and what you need
• Offer solutions, not just problems
• Respond to parent concerns promptly and professionally
• Use multiple communication methods (email, phone, notes)
• Include parents as partners in their child's education

Strong parent partnerships make teaching much easier!"""
    
    elif any(word in question_lower for word in ['assessment', 'grade', 'test', 'evaluate']):
        return """**Assessment Ideas:**
        
• Use formative assessment (exit tickets, thumbs up/down) to check understanding daily
• Try alternative assessments: projects, presentations, portfolios
• Give students opportunities to show learning in different ways
• Provide clear rubrics so students know expectations
• Use peer assessment and self-reflection
• Focus on growth over absolute scores
• Give timely, specific feedback

Assessment should help learning, not just measure it!"""
    
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

def get_teaching_guidance(question):
    """Provide AI-powered teaching guidance"""
    return get_ai_response(question, "teaching")

def generate_simple_lesson_plan(subject, grade, topic):
    """Fallback lesson plan generation"""
    responses = {
        'English': f"Here's a lesson plan for Grade {grade} English on {topic}: Start with vocabulary introduction, followed by reading comprehension activities, and end with creative writing exercises.",
        'Maths': f"For Grade {grade} Mathematics on {topic}: Begin with concept introduction using visual aids, practice with guided examples, then independent problem-solving.",
        'Science': f"Grade {grade} Science lesson on {topic}: Start with observation and questioning, conduct simple experiments, and conclude with scientific explanations.",
        'Urdu': f"برائے جماعت {grade} اردو کا سبق {topic} پر: الفاظ کی تعلیم سے شروع کریں، پھر قرات اور آخر میں تحریری مشق۔"
    }
    return responses.get(subject, f"Here's a basic lesson plan for Grade {grade} {subject} on {topic}: Introduction, main activities, and assessment.")

@app.route('/')
def index():
    return render_template('index.html')

# Attendance & Roster Management Routes
# Backward compatibility: redirect old attendance route to integrated classes-attendance
@app.route('/attendance')
def attendance():
    return redirect(url_for('classes_attendance'))

@app.route('/add_student', methods=['POST'])
def add_student():
    student_name = request.form.get('student_name')
    class_id = request.form.get('class_id', '')  # Get selected class
    
    if student_name:
        students = load_data(STUDENTS_FILE)
        students = migrate_legacy_students(students)
        
        # Check if student already exists
        existing_student = get_student_by_name(students, student_name)
        if not existing_student:
            new_student = create_student_profile(student_name, class_id=class_id)
            students.append(new_student)
            save_data(STUDENTS_FILE, students)
            
            if class_id:
                class_info = get_class_by_id(class_id)
                flash(f'Student {student_name} added to {class_info["name"]} successfully!', 'success')
            else:
                flash(f'Student {student_name} added successfully!', 'success')
        else:
            flash(f'Student {student_name} already exists!', 'error')
    
    return redirect(url_for('classes_attendance'))

@app.route('/edit_student', methods=['POST'])
def edit_student():
    old_name = request.form.get('old_name')
    new_name = request.form.get('new_name')
    
    if old_name and new_name and old_name != new_name:
        students = load_data(STUDENTS_FILE)
        students = migrate_legacy_students(students)
        
        old_student = get_student_by_name(students, old_name)
        existing_new_student = get_student_by_name(students, new_name)
        
        if old_student and not existing_new_student:
            # Update student name
            old_student['name'] = new_name.strip()
            save_data(STUDENTS_FILE, students)
            
            # Update all attendance records
            attendance_data = load_data(ATTENDANCE_FILE)
            for date_key in attendance_data:
                if old_name in attendance_data[date_key]:
                    attendance_data[date_key][new_name] = attendance_data[date_key].pop(old_name)
            save_data(ATTENDANCE_FILE, attendance_data)
            
            flash(f'Student name updated from "{old_name}" to "{new_name}"', 'success')
        elif existing_new_student:
            flash(f'Student name "{new_name}" already exists!', 'error')
        else:
            flash('Student not found!', 'error')
    elif old_name == new_name:
        flash('No changes made to student name.', 'info')
    else:
        flash('Please provide both old and new student names.', 'error')
    
    return redirect(url_for('classes_attendance'))

@app.route('/remove_student', methods=['POST'])
def remove_student():
    student_name = request.form.get('student_name')
    if student_name:
        students = load_data(STUDENTS_FILE)
        students = migrate_legacy_students(students)
        
        student_to_remove = get_student_by_name(students, student_name)
        if student_to_remove:
            students.remove(student_to_remove)
            save_data(STUDENTS_FILE, students)
            flash(f'Student {student_name} removed successfully!', 'success')
        else:
            flash(f'Student {student_name} not found!', 'error')
    return redirect(url_for('classes_attendance'))

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    # Get the date from the form submission
    attendance_date = request.form.get('attendance_date', date.today().isoformat())
    
    # Validate the date
    try:
        datetime.fromisoformat(attendance_date).date()
    except (ValueError, TypeError):
        flash('Invalid date format', 'error')
        return redirect(url_for('classes_attendance'))
    
    attendance_data = load_data(ATTENDANCE_FILE)
    
    if attendance_date not in attendance_data:
        attendance_data[attendance_date] = {}
    
    students = load_data(STUDENTS_FILE)
    students = migrate_legacy_students(students)
    
    for student in students:
        student_name = student.get('name', '') if isinstance(student, dict) else str(student)
        status = request.form.get(f'attendance_{student_name}', 'absent')
        attendance_data[attendance_date][student_name] = status
    
    save_data(ATTENDANCE_FILE, attendance_data)
    
    # Provide appropriate success message based on date
    selected_date = datetime.fromisoformat(attendance_date).date()
    today = date.today()
    
    if selected_date == today:
        flash('Today\'s attendance marked successfully!', 'success')
    elif selected_date < today:
        flash(f'Attendance updated for {selected_date.strftime("%B %d, %Y")}', 'success')
    else:
        flash(f'Future attendance set for {selected_date.strftime("%B %d, %Y")}', 'success')
    
    return redirect(url_for('attendance', date=attendance_date))

# Class Management Routes
@app.route('/classes_attendance')
def classes_attendance():
    """Display integrated classes and attendance management"""
    classes = get_all_classes()
    unassigned_students = get_unassigned_students()
    
    # Add student count to each class
    for class_info in classes:
        class_students = get_students_by_class(class_info['id'])
        class_info['student_count'] = len(class_students)
        class_info['students'] = class_students
    
    return render_template('classes_attendance.html', classes=classes, unassigned_students=unassigned_students)

# Keep original classes route for backward compatibility
@app.route('/classes')
def classes():
    """Redirect to integrated classes and attendance"""
    return redirect(url_for('classes_attendance'))

@app.route('/create_class', methods=['POST'])
def create_class_route():
    """Create a new class"""
    grade = request.form.get('grade')
    section = request.form.get('section')
    
    if not grade or not section:
        flash('Grade and section are required!', 'error')
        return redirect(url_for('classes_attendance'))
    
    try:
        grade = int(grade)
        if grade < 1 or grade > 5:
            flash('Grade must be between 1 and 5!', 'error')
            return redirect(url_for('classes_attendance'))
    except ValueError:
        flash('Invalid grade number!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Check if class already exists
    class_id = f"grade_{grade}{section.lower()}"
    classes_data = load_data(CLASSES_FILE)
    
    if class_id in classes_data:
        flash(f'Grade {grade}{section.upper()} already exists!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Create new class
    new_class = create_class(grade, section)
    classes_data[class_id] = new_class
    save_data(CLASSES_FILE, classes_data)
    
    flash(f'Grade {grade}{section.upper()} created successfully!', 'success')
    return redirect(url_for('classes_attendance'))

@app.route('/edit_class', methods=['POST'])
def edit_class():
    """Edit an existing class"""
    class_id = request.form.get('class_id')
    new_grade = request.form.get('grade')
    new_section = request.form.get('section')
    
    if not class_id or not new_grade or not new_section:
        flash('All fields are required for editing!', 'error')
        return redirect(url_for('classes_attendance'))
    
    try:
        new_grade = int(new_grade)
        if new_grade < 1 or new_grade > 5:
            flash('Grade must be between 1 and 5!', 'error')
            return redirect(url_for('classes_attendance'))
    except ValueError:
        flash('Invalid grade number!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Load classes data
    classes_data = load_data(CLASSES_FILE)
    
    if class_id not in classes_data:
        flash('Class not found!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Create new class name
    new_class_name = f"Grade {new_grade}{new_section.upper()}"
    new_class_id = f"grade_{new_grade}{new_section.lower()}"
    
    # Check if new class name/id already exists (and it's not the same class)
    if new_class_id != class_id and new_class_id in classes_data:
        flash(f'Class {new_class_name} already exists!', 'error')
        return redirect(url_for('classes_attendance'))
    
    old_class_name = classes_data[class_id]['name']
    
    # If the ID is changing, we need to update student assignments
    if new_class_id != class_id:
        # Update all students assigned to this class
        students_data = load_data(STUDENTS_FILE)
        students_data = migrate_legacy_students(students_data)
        
        for student in students_data:
            if student.get('class_id') == class_id:
                student['class_id'] = new_class_id
        
        save_data(STUDENTS_FILE, students_data)
        
        # Move the class data to new ID and remove old entry
        classes_data[new_class_id] = classes_data[class_id].copy()
        del classes_data[class_id]
    
    # Update class information
    classes_data[new_class_id].update({
        'id': new_class_id,
        'name': new_class_name,
        'grade': new_grade,
        'section': new_section.upper()
    })
    
    save_data(CLASSES_FILE, classes_data)
    
    flash(f'Class updated from "{old_class_name}" to "{new_class_name}" successfully!', 'success')
    return redirect(url_for('classes_attendance'))

@app.route('/delete_class', methods=['POST'])
def delete_class():
    """Delete a class and unassign all students"""
    class_id = request.form.get('class_id')
    
    if not class_id:
        flash('Invalid class!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Get class info for confirmation message
    class_info = get_class_by_id(class_id)
    if not class_info:
        flash('Class not found!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Unassign all students from this class
    students = load_data(STUDENTS_FILE)
    students = migrate_legacy_students(students)
    
    for student in students:
        if student.get('class_id') == class_id:
            student['class_id'] = ""
    
    save_data(STUDENTS_FILE, students)
    
    # Delete the class
    classes_data = load_data(CLASSES_FILE)
    if class_id in classes_data:
        del classes_data[class_id]
        save_data(CLASSES_FILE, classes_data)
    
    flash(f'{class_info["name"]} deleted successfully! All students have been unassigned.', 'success')
    return redirect(url_for('classes_attendance'))

@app.route('/assign_student', methods=['POST'])
def assign_student():
    """Assign a student to a class"""
    student_id = request.form.get('student_id')
    class_id = request.form.get('class_id')
    
    if not student_id or not class_id:
        flash('Invalid student or class!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Load students and update the specific student
    students = load_data(STUDENTS_FILE)
    students = migrate_legacy_students(students)
    
    student_found = False
    for student in students:
        if student.get('id') == student_id:
            student['class_id'] = class_id
            student_found = True
            break
    
    if student_found:
        save_data(STUDENTS_FILE, students)
        class_info = get_class_by_id(class_id)
        flash(f'Student assigned to {class_info["name"]} successfully!', 'success')
    else:
        flash('Student not found!', 'error')
    
    return redirect(url_for('classes_attendance'))

@app.route('/class_attendance/<class_id>')
def class_attendance(class_id):
    """Display attendance for a specific class"""
    # Get selected date from query parameter, default to today
    selected_date_str = request.args.get('date', date.today().isoformat())
    
    # Validate and parse the date
    try:
        selected_date = datetime.fromisoformat(selected_date_str).date()
    except (ValueError, TypeError):
        selected_date = date.today()
        selected_date_str = selected_date.isoformat()
    
    # Get class information
    class_info = get_class_by_id(class_id)
    if not class_info:
        flash('Class not found!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Get students in this class
    students = get_students_by_class(class_id)
    
    # Get attendance data for this class and date
    attendance_data = load_data(ATTENDANCE_FILE)
    class_attendance_data = attendance_data.get(selected_date_str, {}).get(class_id, {})
    
    # Determine if this is a past, present, or future date
    today = date.today()
    is_past = selected_date < today
    is_today = selected_date == today
    is_future = selected_date > today
    
    return render_template('class_attendance.html', 
                         class_info=class_info,
                         students=students, 
                         selected_attendance=class_attendance_data, 
                         selected_date=selected_date_str,
                         selected_date_formatted=selected_date.strftime('%B %d, %Y'),
                         is_past=is_past,
                         is_today=is_today,
                         is_future=is_future)

@app.route('/mark_class_attendance', methods=['POST'])
def mark_class_attendance():
    """Mark attendance for a specific class"""
    class_id = request.form.get('class_id')
    attendance_date = request.form.get('attendance_date', date.today().isoformat())
    
    if not class_id:
        flash('Invalid class!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Validate the date
    try:
        datetime.fromisoformat(attendance_date).date()
    except (ValueError, TypeError):
        flash('Invalid date format', 'error')
        return redirect(url_for('class_attendance', class_id=class_id))
    
    # Get class info
    class_info = get_class_by_id(class_id)
    if not class_info:
        flash('Class not found!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Load attendance data
    attendance_data = load_data(ATTENDANCE_FILE)
    
    # Initialize date and class structure if needed
    if attendance_date not in attendance_data:
        attendance_data[attendance_date] = {}
    if class_id not in attendance_data[attendance_date]:
        attendance_data[attendance_date][class_id] = {}
    
    # Get students in this class
    students = get_students_by_class(class_id)
    
    # Mark attendance for each student
    for student in students:
        student_name = student.get('name', '')
        status = request.form.get(f'attendance_{student_name}', 'absent')
        attendance_data[attendance_date][class_id][student_name] = status
    
    save_data(ATTENDANCE_FILE, attendance_data)
    
    # Provide appropriate success message based on date
    selected_date = datetime.fromisoformat(attendance_date).date()
    today = date.today()
    
    if selected_date == today:
        flash(f'Today\'s attendance for {class_info["name"]} marked successfully!', 'success')
    elif selected_date < today:
        flash(f'Attendance updated for {class_info["name"]} on {selected_date.strftime("%B %d, %Y")}', 'success')
    else:
        flash(f'Future attendance set for {class_info["name"]} on {selected_date.strftime("%B %d, %Y")}', 'success')
    
    return redirect(url_for('class_attendance', class_id=class_id, date=attendance_date))

@app.route('/student_profile/<student_id>')
def student_profile(student_id):
    """Display and manage student profile"""
    students = load_data(STUDENTS_FILE)
    students = migrate_legacy_students(students)
    
    student = get_student_by_id(students, student_id)
    if not student:
        flash('Student not found!', 'error')
        return redirect(url_for('classes_attendance'))
    
    return render_template('student_profile.html', student=student)

@app.route('/update_student_profile/<student_id>', methods=['POST'])
def update_student_profile(student_id):
    """Update student profile information"""
    students = load_data(STUDENTS_FILE)
    students = migrate_legacy_students(students)
    
    student = get_student_by_id(students, student_id)
    if not student:
        flash('Student not found!', 'error')
        return redirect(url_for('classes_attendance'))
    
    # Update student profile fields
    student['name'] = request.form.get('name', '').strip()
    student['guardian_name'] = request.form.get('guardian_name', '').strip()
    student['guardian_phone'] = request.form.get('guardian_phone', '').strip()
    student['address'] = request.form.get('address', '').strip()
    student['gender'] = request.form.get('gender', '').strip()
    
    # Handle profile picture upload
    if 'profile_picture' in request.files:
        file = request.files['profile_picture']
        if file and file.filename:
            filename = secure_filename(f"{student_id}_{file.filename}")
            file_path = os.path.join('data', 'profile_pictures', filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            student['profile_picture'] = filename
    
    save_data(STUDENTS_FILE, students)
    flash(f'Profile updated for {student["name"]}!', 'success')
    
    return redirect(url_for('student_profile', student_id=student_id))

# Planner feature removed as requested

# Grading feature removed as requested

# Quiz generator feature removed as requested

# Weekly Report Routes
@app.route('/weekly_report')
def weekly_report():
    # Get week range from query params or default to current week
    from datetime import timedelta, datetime
    
    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    end_of_week = start_of_week + timedelta(days=6)  # Sunday
    
    # Allow custom date range
    start_date = request.args.get('start_date', start_of_week.isoformat())
    end_date = request.args.get('end_date', end_of_week.isoformat())
    custom_total_days = request.args.get('custom_total_days', type=int)
    
    # Calculate attendance report
    if custom_total_days:
        report_data = calculate_attendance_summary(start_date, end_date, 'custom', custom_total_days)
    else:
        report_data = calculate_attendance_summary(start_date, end_date, 'recorded_only')
    
    # Handle calculation errors
    if 'error' in report_data:
        flash(report_data['error'], 'danger')
        return redirect(url_for('weekly_report'))
    
    return render_template('attendance_report.html', 
                         report=report_data, 
                         start_date=start_date, 
                         end_date=end_date,
                         report_type='Weekly',
                         custom_total_days=custom_total_days)

@app.route('/monthly_report')
def monthly_report():
    # Get month range from query params or default to current month
    from datetime import datetime
    import calendar
    
    today = datetime.now().date()
    year = int(request.args.get('year', today.year))
    month = int(request.args.get('month', today.month))
    
    # Calculate month start and end dates
    start_date = datetime(year, month, 1).date().isoformat()
    last_day = calendar.monthrange(year, month)[1]
    end_date = datetime(year, month, last_day).date().isoformat()
    
    custom_total_days = request.args.get('custom_total_days', type=int)
    
    # Calculate attendance report
    if custom_total_days:
        report_data = calculate_attendance_summary(start_date, end_date, 'custom', custom_total_days)
    else:
        report_data = calculate_attendance_summary(start_date, end_date, 'recorded_only')
    
    # Handle calculation errors
    if 'error' in report_data:
        flash(report_data['error'], 'danger')
        return redirect(url_for('monthly_report'))
    
    return render_template('attendance_report.html', 
                         report=report_data, 
                         start_date=start_date, 
                         end_date=end_date,
                         report_type='Monthly',
                         year=year,
                         month=month,
                         custom_total_days=custom_total_days)

@app.route('/yearly_report')
def yearly_report():
    # Get year range from query params or default to current year
    from datetime import datetime
    
    today = datetime.now().date()
    year = int(request.args.get('year', today.year))
    
    # Calculate year start and end dates
    start_date = datetime(year, 1, 1).date().isoformat()
    end_date = datetime(year, 12, 31).date().isoformat()
    
    custom_total_days = request.args.get('custom_total_days', type=int)
    
    # Calculate attendance report
    if custom_total_days:
        report_data = calculate_attendance_summary(start_date, end_date, 'custom', custom_total_days)
    else:
        report_data = calculate_attendance_summary(start_date, end_date, 'recorded_only')
    
    # Handle calculation errors
    if 'error' in report_data:
        flash(report_data['error'], 'danger')
        return redirect(url_for('yearly_report'))
    
    return render_template('attendance_report.html', 
                         report=report_data, 
                         start_date=start_date, 
                         end_date=end_date,
                         report_type='Yearly',
                         year=year,
                         custom_total_days=custom_total_days)

@app.route('/daily_report')
def daily_report():
    # Get date from query params or default to today
    from datetime import datetime
    
    today = datetime.now().date()
    selected_date = request.args.get('date', today.isoformat())
    
    try:
        # Validate date format
        datetime.fromisoformat(selected_date).date()
    except ValueError:
        flash('Invalid date format', 'danger')
        return redirect(url_for('daily_report'))
    
    # For daily reports, start and end date are the same
    start_date = selected_date
    end_date = selected_date
    
    # Calculate attendance report for the single day
    report_data = calculate_attendance_summary(start_date, end_date, 'recorded_only')
    
    # Handle calculation errors
    if 'error' in report_data:
        flash(report_data['error'], 'danger')
        return redirect(url_for('daily_report'))
    
    return render_template('attendance_report.html', 
                         report=report_data, 
                         start_date=start_date, 
                         end_date=end_date,
                         report_type='Daily',
                         selected_date=selected_date)

def calculate_attendance_summary(start_date, end_date, denominator_mode='recorded_only', custom_total_days=None):
    """Calculate attendance statistics for any period with flexible denominator"""
    from datetime import datetime, timedelta
    
    students = load_data(STUDENTS_FILE)
    attendance_data = load_data(ATTENDANCE_FILE)
    
    try:
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        
        # Validate date range
        if start > end:
            raise ValueError("Start date must be before or equal to end date")
            
    except ValueError as e:
        return {
            'error': f"Invalid date range: {str(e)}",
            'start_date': start_date,
            'end_date': end_date,
            'total_days': 0,
            'total_students': 0,
            'student_stats': {},
            'overall_percentage': 0,
            'total_present': 0,
            'total_possible': 0
        }
    
    # Get all dates in the range that have attendance records
    recorded_dates = []
    current_date = start
    while current_date <= end:
        date_str = current_date.isoformat()
        if date_str in attendance_data:  # Only include days with actual attendance records
            recorded_dates.append(date_str)
        current_date += timedelta(days=1)
    
    # Calculate statistics for each student
    student_stats = {}
    
    # Determine total days for calculation
    if denominator_mode == 'custom' and custom_total_days is not None:
        # Validate custom total days
        max_present = 0
        for student in students:
            present_count = sum(1 for date_str in recorded_dates 
                              if attendance_data[date_str].get(student, 'absent') == 'present')
            max_present = max(max_present, present_count)
        
        if custom_total_days < max_present:
            return {
                'error': f"Custom total days ({custom_total_days}) cannot be less than maximum present days ({max_present})",
                'start_date': start_date,
                'end_date': end_date,
                'total_days': 0,
                'total_students': 0,
                'student_stats': {},
                'overall_percentage': 0,
                'total_present': 0,
                'total_possible': 0
            }
        
        total_days = custom_total_days
    else:
        total_days = len(recorded_dates)  # Use recorded days only
    
    for student in students:
        present_count = 0
        
        for date_str in recorded_dates:
            status = attendance_data[date_str].get(student, 'absent')
            if status == 'present':
                present_count += 1
        
        absent_count = total_days - present_count
        attendance_percentage = (present_count / total_days * 100) if total_days > 0 else 0
        
        student_stats[student] = {
            'present': present_count,
            'absent': absent_count,
            'total_days': total_days,
            'percentage': round(attendance_percentage, 1)
        }
    
    # Calculate overall statistics
    total_students = len(students)
    total_possible_attendance = total_students * total_days
    total_present = sum(stats['present'] for stats in student_stats.values())
    overall_percentage = (total_present / total_possible_attendance * 100) if total_possible_attendance > 0 else 0
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'total_days': total_days,
        'total_students': total_students,
        'student_stats': student_stats,
        'overall_percentage': round(overall_percentage, 1),
        'total_present': total_present,
        'total_possible': total_possible_attendance,
        'recorded_dates': recorded_dates,
        'denominator_mode': denominator_mode
    }

def calculate_weekly_attendance(start_date, end_date):
    """Calculate weekly attendance statistics (backward compatibility)"""
    return calculate_attendance_summary(start_date, end_date, 'recorded_only')

@app.route('/generate_whatsapp_share')
def generate_whatsapp_share():
    """Generate WhatsApp share link with attendance report"""
    import urllib.parse
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    custom_total_days = request.args.get('custom_total_days', type=int)
    
    if not start_date or not end_date:
        flash('Invalid date range for report generation', 'danger')
        return redirect(url_for('weekly_report'))
    
    # Determine report type
    if start_date == end_date:
        report_type = 'Daily'
        redirect_url = 'daily_report'
    else:
        from datetime import datetime
        try:
            start = datetime.fromisoformat(start_date).date()
            end = datetime.fromisoformat(end_date).date()
            days_diff = (end - start).days + 1
            
            if days_diff <= 7:
                report_type = 'Weekly'
                redirect_url = 'weekly_report'
            elif days_diff <= 31:
                report_type = 'Monthly'
                redirect_url = 'monthly_report'
            else:
                report_type = 'Yearly'
                redirect_url = 'yearly_report'
        except:
            report_type = 'Weekly'
            redirect_url = 'weekly_report'
    
    # Generate report data
    if custom_total_days:
        report_data = calculate_attendance_summary(start_date, end_date, 'custom', custom_total_days)
    else:
        report_data = calculate_attendance_summary(start_date, end_date, 'recorded_only')
    
    # Handle calculation errors
    if 'error' in report_data:
        flash(report_data['error'], 'danger')
        return redirect(url_for(redirect_url))
    
    # Format report for WhatsApp sharing
    message = format_report_for_whatsapp(report_data, report_type)
    
    # Create WhatsApp share link
    whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(message)}"
    
    return redirect(whatsapp_url)

def format_report_for_whatsapp(report_data, report_type='Weekly'):
    """Format attendance report as text for WhatsApp sharing"""
    message = f"📊 *{report_type} Attendance Report*\n"
    
    if report_type == 'Daily':
        message += f"📅 Date: {report_data['start_date']}\n\n"
    else:
        message += f"📅 Period: {report_data['start_date']} to {report_data['end_date']}\n\n"
    
    message += f"📈 *Overall Summary:*\n"
    message += f"• Total Students: {report_data['total_students']}\n"
    
    if report_type == 'Daily':
        message += f"• School Day: {report_data['total_days']}\n"
    else:
        days_label = "Custom Total Days" if report_data.get('denominator_mode') == 'custom' else "Days with Records"
        message += f"• {days_label}: {report_data['total_days']}\n"
    
    message += f"• Overall Attendance: {report_data['overall_percentage']}%\n"
    message += f"• Total Present: {report_data['total_present']}/{report_data['total_possible']}\n\n"
    
    message += f"👥 *Individual Attendance:*\n"
    
    if report_data['total_days'] == 0:
        message += "No attendance records found for this period.\n"
    else:
        for student, stats in report_data['student_stats'].items():
            if report_type == 'Daily':
                status_emoji = "✅" if stats['present'] > 0 else "❌"
                status_text = "Present" if stats['present'] > 0 else "Absent"
                message += f"{status_emoji} {student}: {status_text}\n"
            else:
                status_emoji = "✅" if stats['percentage'] >= 80 else "⚠️" if stats['percentage'] >= 60 else "❌"
                message += f"{status_emoji} {student}: {stats['percentage']}% ({stats['present']}/{stats['total_days']} days)\n"
    
    message += f"\n📱 Generated by Tutor's Assistant"
    
    return message

# Chatbot Routes
@app.route('/chatbot')
def chatbot():
    # Reset chatbot state on page load
    clear_session_data()
    set_chatbot_state(ChatbotState.MAIN_MENU)
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot messages with session-based state management"""
    # Guard against malformed requests
    if not request.json or 'message' not in request.json:
        return jsonify({'message': 'Invalid request format'})
    
    user_message = request.json.get('message', '').strip()
    current_state = get_chatbot_state()
    
    # Handle special commands
    if user_message.lower() in ['menu', 'main menu', 'back', 'home', 'start']:
        clear_session_data()
        set_chatbot_state(ChatbotState.MAIN_MENU)
        return jsonify(get_main_menu_response())
    
    if user_message.lower() in ['hi', 'hello', 'help']:
        clear_session_data()
        set_chatbot_state(ChatbotState.MAIN_MENU)
        return jsonify({
            'message': 'Hello! I\'m your AI Teaching Assistant. How can I help you today?',
            'options': ['Help with Lesson Plan', 'Activities', 'Definitions', 'General Questions', 'Free Chat'],
            'show_menu': True
        })
    
    # State machine logic
    if current_state == ChatbotState.MAIN_MENU:
        return handle_main_menu(user_message)
    elif current_state == ChatbotState.GRADE_SELECTION:
        return handle_grade_selection(user_message)
    elif current_state == ChatbotState.SUBJECT_SELECTION:
        return handle_subject_selection(user_message)
    elif current_state == ChatbotState.TOPIC_INPUT:
        return handle_topic_input(user_message)
    elif current_state == ChatbotState.ACTIVITIES:
        return handle_activities(user_message)
    elif current_state == ChatbotState.DEFINITIONS:
        return handle_definitions(user_message)
    elif current_state == ChatbotState.GENERAL_QUESTIONS:
        return handle_general_questions(user_message)
    elif current_state == ChatbotState.FREE_CHAT:
        return handle_free_chat(user_message)
    else:
        # Fallback to main menu
        set_chatbot_state(ChatbotState.MAIN_MENU)
        return jsonify(get_main_menu_response())

def get_main_menu_response():
    """Get the main menu response"""
    return {
        'message': 'Welcome to your AI Teaching Assistant! Choose what you\'d like help with:',
        'options': ['Help with Lesson Plan', 'Activities', 'Definitions', 'General Questions', 'Free Chat'],
        'show_menu': True
    }

def handle_main_menu(user_message):
    """Handle main menu selections"""
    message_lower = user_message.lower()
    
    if 'lesson plan' in message_lower or message_lower == 'help with lesson plan':
        set_chatbot_state(ChatbotState.GRADE_SELECTION)
        return jsonify({
            'message': 'Great! Let\'s create a lesson plan. First, which grade are you teaching?',
            'options': ChatbotState.GRADES,
            'navigation': 'Grade Selection'
        })
    
    elif 'activities' in message_lower:
        set_chatbot_state(ChatbotState.ACTIVITIES)
        return jsonify({
            'message': 'What type of activities would you like suggestions for?',
            'options': list(ChatbotState.ACTIVITIES_TYPES.keys()),
            'navigation': 'Activity Types'
        })
    
    elif 'definitions' in message_lower:
        set_chatbot_state(ChatbotState.DEFINITIONS)
        return jsonify({
            'message': 'What term or concept would you like me to define? Just type it in the chat below.',
            'navigation': 'Definitions'
        })
    
    elif 'general questions' in message_lower:
        set_chatbot_state(ChatbotState.GENERAL_QUESTIONS)
        return jsonify({
            'message': 'I\'m here to help with any teaching-related questions you have! What would you like to know? Just type your question below.',
            'navigation': 'General Questions'
        })
    
    elif 'free chat' in message_lower or 'chat' in message_lower or 'conversation' in message_lower:
        set_chatbot_state(ChatbotState.FREE_CHAT)
        return jsonify({
            'message': 'Great! I\'m now in free chat mode. You can ask me anything or have a natural conversation. How can I help you today?',
            'navigation': 'Free Chat'
        })
    
    else:
        return jsonify({
            'message': 'Please choose one of the options from the menu above, or type "menu" to see all options again.',
            'show_menu': True
        })

def handle_grade_selection(user_message):
    """Handle grade selection for lesson planning"""
    if user_message in ChatbotState.GRADES:
        set_session_data('grade', user_message)
        set_chatbot_state(ChatbotState.SUBJECT_SELECTION)
        return jsonify({
            'message': f'Perfect! Grade {user_message} selected. Now, which subject?',
            'options': list(ChatbotState.SUBJECTS.keys()),
            'navigation': 'Subject Selection',
            'breadcrumb': f'Grade {user_message}'
        })
    else:
        return jsonify({
            'message': 'Please select a valid grade (1-5) from the options above.',
            'options': ChatbotState.GRADES
        })

def handle_subject_selection(user_message):
    """Handle subject selection for lesson planning"""
    if user_message in ChatbotState.SUBJECTS:
        set_session_data('subject', user_message)
        set_chatbot_state(ChatbotState.TOPIC_INPUT)
        
        grade = get_session_data('grade')
        return jsonify({
            'message': f'Excellent! {user_message} for Grade {grade}. What specific topic would you like the lesson plan to cover? Just type the topic below.',
            'suggestions': ChatbotState.SUBJECTS[user_message],
            'navigation': 'Topic Input',
            'breadcrumb': f'Grade {grade} → {user_message}'
        })
    else:
        return jsonify({
            'message': 'Please select one of the available subjects from the options above.',
            'options': list(ChatbotState.SUBJECTS.keys())
        })

def handle_topic_input(user_message):
    """Handle topic input and generate lesson plan"""
    if user_message.strip():
        grade = get_session_data('grade')
        subject = get_session_data('subject')
        topic = user_message.strip()
        
        # Generate lesson plan using AI
        lesson_plan = generate_lesson_plan_with_ai(subject, grade, topic)
        
        # Clear session and return to main menu
        clear_session_data()
        set_chatbot_state(ChatbotState.MAIN_MENU)
        
        return jsonify({
            'message': lesson_plan,
            'type': 'lesson_plan',
            'title': f'Lesson Plan: Grade {grade} {subject} - {topic}',
            'return_to_menu': True
        })
    else:
        return jsonify({
            'message': 'Please enter a topic for your lesson plan in the chat below.'
        })

def handle_activities(user_message):
    """Handle activity type selection"""
    if user_message in ChatbotState.ACTIVITIES_TYPES:
        activities = ChatbotState.ACTIVITIES_TYPES[user_message]
        
        clear_session_data()
        set_chatbot_state(ChatbotState.MAIN_MENU)
        
        return jsonify({
            'message': f'Here are some {user_message} ideas:\n\n' + '\n'.join(f'• {activity}' for activity in activities),
            'type': 'activities',
            'title': f'{user_message} Suggestions',
            'return_to_menu': True
        })
    else:
        return jsonify({
            'message': 'Please select one of the activity types from the options above.',
            'options': list(ChatbotState.ACTIVITIES_TYPES.keys())
        })

def handle_definitions(user_message):
    """Handle definition requests using Wikipedia API"""
    if user_message.strip():
        term = user_message.strip()
        definition = get_wikipedia_definition(term)
        
        clear_session_data()
        set_chatbot_state(ChatbotState.MAIN_MENU)
        
        return jsonify({
            'message': definition,
            'type': 'definition',
            'title': f'Definition: {term}',
            'return_to_menu': True
        })
    else:
        return jsonify({
            'message': 'Please enter a term or concept you\'d like me to define.',
        })

def handle_general_questions(user_message):
    """Handle general teaching questions using educational resources"""
    if user_message.strip():
        question = user_message.strip()
        answer = get_teaching_guidance(question)
        
        clear_session_data()
        set_chatbot_state(ChatbotState.MAIN_MENU)
        
        return jsonify({
            'message': answer,
            'type': 'general_answer',
            'return_to_menu': True
        })
    else:
        return jsonify({
            'message': 'Please ask me a question about teaching, classroom management, or education in the chat below.'
        })

def handle_free_chat(user_message):
    """Handle free conversation using AI"""
    if user_message.strip():
        # Use AI for natural conversation
        response = get_ai_response(user_message, "general")
        
        # Stay in free chat mode for continued conversation (don't reset state)
        return jsonify({
            'message': response,
            'type': 'free_chat',
            'navigation': 'Free Chat',
            'continue_chat': True
        })
    else:
        return jsonify({
            'message': 'I\'m here for a natural conversation! Ask me anything you\'d like to discuss.',
            'navigation': 'Free Chat'
        })

# File Upload Routes
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Handle audio file upload and transcription"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.webm"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        # Save the audio file
        audio_file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Save metadata
        save_file_metadata(file_id, audio_file.filename, 'audio/webm', file_size, 'audio')
        
        # Try to transcribe with Gemini
        transcript = ""
        if gemini_model:
            try:
                # Upload to Gemini for transcription
                audio_file_gemini = genai.upload_file(file_path)
                
                prompt = "Please transcribe this audio exactly as spoken. Only provide the transcription text, no additional commentary."
                response = gemini_model.generate_content([prompt, audio_file_gemini])
                transcript = response.text.strip()
                
                # Clean up the Gemini uploaded file
                genai.delete_file(audio_file_gemini.name)
                
            except Exception as e:
                print(f"Transcription error: {e}")
                transcript = "[Audio recorded but transcription unavailable]"
        
        return jsonify({
            'audio_id': file_id,
            'transcript': transcript,
            'size': file_size,
            'duration': 'unknown'
        })
        
    except Exception as e:
        print(f"Audio upload error: {e}")
        return jsonify({'error': 'Failed to upload audio'}), 500

@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Handle image and document file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file size (already handled by Flask MAX_CONTENT_LENGTH)
        
        # Generate secure filename
        original_filename = secure_filename(uploaded_file.filename)
        file_extension = os.path.splitext(original_filename)[1].lower()
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{file_extension}"
        
        # Determine file type and save location
        mime_type = get_file_mime_type(original_filename)
        
        if mime_type in ALLOWED_IMAGE_TYPES:
            file_path = os.path.join(IMAGES_DIR, filename)
            file_type = 'image'
        elif mime_type in ALLOWED_DOC_TYPES:
            file_path = os.path.join(DOCS_DIR, filename)
            file_type = 'document'
        else:
            return jsonify({'error': f'File type not allowed: {mime_type}'}), 400
        
        # Save the file
        uploaded_file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Additional processing based on file type
        extracted_text = ""
        if file_type == 'image':
            # Strip EXIF data for privacy
            strip_image_metadata(file_path)
        elif file_type == 'document':
            # Extract text for processing
            if mime_type == 'application/pdf':
                extracted_text = extract_text_from_pdf(file_path)
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                extracted_text = extract_text_from_docx(file_path)
            elif mime_type == 'text/plain':
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
        
        # Save metadata
        metadata = save_file_metadata(file_id, original_filename, mime_type, file_size, file_type)
        if extracted_text:
            metadata['extracted_text'] = extracted_text
            # Update metadata file
            metadata_path = os.path.join(META_DIR, f"{file_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        return jsonify({
            'file_id': file_id,
            'type': file_type,
            'mime_type': mime_type,
            'size': file_size,
            'original_name': original_filename,
            'extracted_text': extracted_text[:500] + '...' if len(extracted_text) > 500 else extracted_text
        })
        
    except Exception as e:
        print(f"File upload error: {e}")
        return jsonify({'error': 'Failed to upload file'}), 500

@app.route('/media/<file_id>')
def serve_media(file_id):
    """Serve uploaded media files with session check"""
    try:
        # Get file metadata
        metadata = get_file_metadata(file_id)
        if not metadata:
            return "File not found", 404
        
        # Find the actual file
        file_type = metadata['type']
        if file_type == 'audio':
            file_path = os.path.join(AUDIO_DIR, f"{file_id}.webm")
        elif file_type == 'image':
            # Find file with any allowed extension
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                potential_path = os.path.join(IMAGES_DIR, f"{file_id}{ext}")
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
            else:
                return "File not found", 404
        elif file_type == 'document':
            # Find file with any allowed extension
            for ext in ['.pdf', '.txt', '.docx']:
                potential_path = os.path.join(DOCS_DIR, f"{file_id}{ext}")
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break
            else:
                return "File not found", 404
        else:
            return "Invalid file type", 400
        
        if not os.path.exists(file_path):
            return "File not found", 404
        
        # Update access time
        metadata['accessed_at'] = datetime.now().isoformat()
        metadata_path = os.path.join(META_DIR, f"{file_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Serve the file
        from flask import send_file
        if file_type == 'audio':
            return send_file(file_path, mimetype=metadata['mime_type'])
        elif file_type == 'image':
            return send_file(file_path, mimetype=metadata['mime_type'])
        else:  # document
            return send_file(file_path, as_attachment=True, download_name=metadata['original_name'])
        
    except Exception as e:
        print(f"Media serve error: {e}")
        return "Error serving file", 500

if __name__ == '__main__':
    init_data_files()
    app.run(host='0.0.0.0', port=5000, debug=True)