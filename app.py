from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import json
import os
import uuid
from datetime import datetime, date
import calendar
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'tutor-assistant-secret-key')

# Data file paths
DATA_DIR = 'data'
STUDENTS_FILE = os.path.join(DATA_DIR, 'students.json')
ATTENDANCE_FILE = os.path.join(DATA_DIR, 'attendance.json')
SCHEDULE_FILE = os.path.join(DATA_DIR, 'schedule.json')
GRADES_FILE = os.path.join(DATA_DIR, 'grades.json')

# Ensure data directory and files exist
def init_data_files():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Initialize empty data files if they don't exist
    default_data = {
        STUDENTS_FILE: [],
        ATTENDANCE_FILE: {},
        SCHEDULE_FILE: {},
        GRADES_FILE: {}
    }
    
    for file_path, default_content in default_data.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(default_content, f)

# Load data from JSON files
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if 'attendance' in file_path or 'schedule' in file_path or 'grades' in file_path else []

# Save data to JSON files
def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# Student profile helper functions
def create_student_profile(name, guardian_name="", guardian_phone="", address="", profile_picture=""):
    """Create a new student profile object"""
    return {
        "id": f"student_{uuid.uuid4().hex[:8]}",
        "name": name.strip(),
        "guardian_name": guardian_name.strip(),
        "guardian_phone": guardian_phone.strip(),
        "address": address.strip(),
        "profile_picture": profile_picture
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
            # Already in new format
            migrated.append(student)
    return migrated

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
    """Generate detailed lesson plan using OpenAI API"""
    try:
        from openai import OpenAI
        client = OpenAI()
        
        prompt = f"""
Create a detailed lesson plan for Grade {grade} {subject} on the topic "{topic}".

Please include:
1. Learning objectives (2-3 clear goals)
2. Materials needed
3. Lesson structure (Introduction, Main Activity, Conclusion)
4. Activities and exercises
5. Assessment methods
6. Time allocation for each section

Make it practical and age-appropriate for Grade {grade} students.
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Fallback to simple response if OpenAI fails
        return generate_simple_lesson_plan(subject, grade, topic)

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
@app.route('/attendance')
def attendance():
    # Get selected date from query parameter, default to today
    selected_date_str = request.args.get('date', date.today().isoformat())
    
    # Validate and parse the date
    try:
        selected_date = datetime.fromisoformat(selected_date_str).date()
    except (ValueError, TypeError):
        selected_date = date.today()
        selected_date_str = selected_date.isoformat()
    
    students = load_data(STUDENTS_FILE)
    # Ensure migration from legacy format if needed
    students = migrate_legacy_students(students)
    save_data(STUDENTS_FILE, students)
    
    attendance_data = load_data(ATTENDANCE_FILE)
    selected_attendance = attendance_data.get(selected_date_str, {})
    
    # Determine if this is a past, present, or future date
    today = date.today()
    is_past = selected_date < today
    is_today = selected_date == today
    is_future = selected_date > today
    
    return render_template('attendance.html', 
                         students=students, 
                         selected_attendance=selected_attendance, 
                         selected_date=selected_date_str,
                         selected_date_formatted=selected_date.strftime('%B %d, %Y'),
                         is_past=is_past,
                         is_today=is_today,
                         is_future=is_future)

@app.route('/add_student', methods=['POST'])
def add_student():
    student_name = request.form.get('student_name')
    if student_name:
        students = load_data(STUDENTS_FILE)
        students = migrate_legacy_students(students)
        
        # Check if student already exists
        existing_student = get_student_by_name(students, student_name)
        if not existing_student:
            new_student = create_student_profile(student_name)
            students.append(new_student)
            save_data(STUDENTS_FILE, students)
            flash(f'Student {student_name} added successfully!', 'success')
        else:
            flash(f'Student {student_name} already exists!', 'error')
    return redirect(url_for('attendance'))

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
    
    return redirect(url_for('attendance'))

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
    return redirect(url_for('attendance'))

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    # Get the date from the form submission
    attendance_date = request.form.get('attendance_date', date.today().isoformat())
    
    # Validate the date
    try:
        datetime.fromisoformat(attendance_date).date()
    except (ValueError, TypeError):
        flash('Invalid date format', 'error')
        return redirect(url_for('attendance'))
    
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
            'options': ['Help with Lesson Plan', 'Activities', 'Definitions', 'General Questions'],
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
    else:
        # Fallback to main menu
        set_chatbot_state(ChatbotState.MAIN_MENU)
        return jsonify(get_main_menu_response())

def get_main_menu_response():
    """Get the main menu response"""
    return {
        'message': 'Welcome to your AI Teaching Assistant! Choose what you\'d like help with:',
        'options': ['Help with Lesson Plan', 'Activities', 'Definitions', 'General Questions'],
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
            'message': 'What term or concept would you like me to define?',
            'input_placeholder': 'Enter a word or concept...',
            'show_input': True,
            'navigation': 'Definitions'
        })
    
    elif 'general questions' in message_lower:
        set_chatbot_state(ChatbotState.GENERAL_QUESTIONS)
        return jsonify({
            'message': 'I\'m here to help with any teaching-related questions you have! What would you like to know?',
            'input_placeholder': 'Ask me anything about teaching...',
            'show_input': True,
            'navigation': 'General Questions'
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
            'message': f'Excellent! {user_message} for Grade {grade}. What specific topic would you like the lesson plan to cover?',
            'input_placeholder': f'Enter a {user_message} topic...',
            'show_input': True,
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
            'message': 'Please enter a topic for your lesson plan.',
            'show_input': True
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
    """Handle definition requests"""
    if user_message.strip():
        term = user_message.strip()
        
        try:
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user", 
                    "content": f"Provide a clear, educational definition of '{term}' suitable for elementary school teachers. Include examples if helpful."
                }],
                max_tokens=300,
                temperature=0.7
            )
            
            definition = response.choices[0].message.content
        except Exception:
            definition = f"I'd be happy to help define '{term}' for you, but I'm having trouble accessing detailed definitions right now. Try asking about it in a more specific way or check educational resources."
        
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
            'show_input': True
        })

def handle_general_questions(user_message):
    """Handle general teaching questions"""
    if user_message.strip():
        question = user_message.strip()
        
        try:
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "You are a helpful AI teaching assistant for elementary school teachers. Provide practical, actionable advice."
                }, {
                    "role": "user", 
                    "content": question
                }],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
        except Exception:
            answer = "I'd love to help with your teaching question! Unfortunately, I'm having trouble accessing my full knowledge base right now. Try rephrasing your question or ask about specific teaching strategies, classroom management, or lesson planning."
        
        clear_session_data()
        set_chatbot_state(ChatbotState.MAIN_MENU)
        
        return jsonify({
            'message': answer,
            'type': 'general_answer',
            'return_to_menu': True
        })
    else:
        return jsonify({
            'message': 'Please ask me a question about teaching, classroom management, or education.',
            'show_input': True
        })

if __name__ == '__main__':
    init_data_files()
    app.run(host='0.0.0.0', port=5000, debug=True)