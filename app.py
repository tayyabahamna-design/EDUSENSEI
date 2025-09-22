from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
import os
from datetime import datetime, date
import calendar

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

# Simulated AI functions
def simulate_ai_grading():
    """Simulate AI-powered grading system"""
    import random
    return random.randint(75, 100)

def generate_lesson_plan_response(subject, grade, topic):
    """Simulate AI lesson plan generation"""
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
    students = load_data(STUDENTS_FILE)
    today = date.today().isoformat()
    attendance_data = load_data(ATTENDANCE_FILE)
    today_attendance = attendance_data.get(today, {})
    
    return render_template('attendance.html', students=students, today_attendance=today_attendance, today=today)

@app.route('/add_student', methods=['POST'])
def add_student():
    student_name = request.form.get('student_name')
    if student_name:
        students = load_data(STUDENTS_FILE)
        if student_name not in students:
            students.append(student_name)
            save_data(STUDENTS_FILE, students)
            flash(f'Student {student_name} added successfully!', 'success')
    return redirect(url_for('attendance'))

@app.route('/remove_student', methods=['POST'])
def remove_student():
    student_name = request.form.get('student_name')
    if student_name:
        students = load_data(STUDENTS_FILE)
        if student_name in students:
            students.remove(student_name)
            save_data(STUDENTS_FILE, students)
            flash(f'Student {student_name} removed successfully!', 'success')
    return redirect(url_for('attendance'))

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    today = date.today().isoformat()
    attendance_data = load_data(ATTENDANCE_FILE)
    
    if today not in attendance_data:
        attendance_data[today] = {}
    
    students = load_data(STUDENTS_FILE)
    for student in students:
        status = request.form.get(f'attendance_{student}', 'absent')
        attendance_data[today][student] = status
    
    save_data(ATTENDANCE_FILE, attendance_data)
    flash('Attendance marked successfully!', 'success')
    return redirect(url_for('attendance'))

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
    
    # Calculate attendance report
    report_data = calculate_weekly_attendance(start_date, end_date)
    
    # Handle calculation errors
    if 'error' in report_data:
        flash(report_data['error'], 'danger')
        return redirect(url_for('weekly_report'))
    
    return render_template('weekly_report.html', 
                         report=report_data, 
                         start_date=start_date, 
                         end_date=end_date)

def calculate_weekly_attendance(start_date, end_date):
    """Calculate weekly attendance statistics"""
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
    date_range = []
    current_date = start
    while current_date <= end:
        date_str = current_date.isoformat()
        if date_str in attendance_data:  # Only include days with actual attendance records
            date_range.append(date_str)
        current_date += timedelta(days=1)
    
    # Calculate statistics for each student
    student_stats = {}
    total_days = len(date_range)  # Only count days with records
    
    for student in students:
        present_count = 0
        absent_count = 0
        
        for date_str in date_range:
            status = attendance_data[date_str].get(student, 'absent')
            if status == 'present':
                present_count += 1
            else:
                absent_count += 1
        
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
        'recorded_dates': date_range
    }

@app.route('/generate_whatsapp_share')
def generate_whatsapp_share():
    """Generate WhatsApp share link with attendance report"""
    import urllib.parse
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        flash('Invalid date range for report generation', 'danger')
        return redirect(url_for('weekly_report'))
    
    # Generate report data
    report_data = calculate_weekly_attendance(start_date, end_date)
    
    # Handle calculation errors
    if 'error' in report_data:
        flash(report_data['error'], 'danger')
        return redirect(url_for('weekly_report'))
    
    # Format report for WhatsApp sharing
    message = format_report_for_whatsapp(report_data)
    
    # Create WhatsApp share link
    whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(message)}"
    
    return redirect(whatsapp_url)

def format_report_for_whatsapp(report_data):
    """Format attendance report as text for WhatsApp sharing"""
    message = f"📊 *Weekly Attendance Report*\n"
    message += f"📅 Period: {report_data['start_date']} to {report_data['end_date']}\n\n"
    
    message += f"📈 *Overall Summary:*\n"
    message += f"• Total Students: {report_data['total_students']}\n"
    message += f"• Days with Records: {report_data['total_days']}\n"
    message += f"• Overall Attendance: {report_data['overall_percentage']}%\n"
    message += f"• Total Present: {report_data['total_present']}/{report_data['total_possible']}\n\n"
    
    message += f"👥 *Individual Attendance:*\n"
    for student, stats in report_data['student_stats'].items():
        status_emoji = "✅" if stats['percentage'] >= 80 else "⚠️" if stats['percentage'] >= 60 else "❌"
        message += f"{status_emoji} {student}: {stats['percentage']}% ({stats['present']}/{stats['total_days']} days)\n"
    
    message += f"\n📱 Generated by Tutor's Assistant"
    
    return message

# Chatbot Routes
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Guard against malformed requests
    if not request.json or 'message' not in request.json:
        return jsonify({'message': 'Invalid request format'})
    
    user_message = request.json.get('message', '').lower().strip()
    
    # Show main menu for greeting or menu requests
    if user_message in ['hi', 'hello', 'menu', 'help', 'start']:
        response = {
            'message': 'Hello! I\'m your AI Teaching Assistant. How can I help you today?',
            'options': ['Help with Lesson Plan', 'Activities', 'Definitions', 'General Questions']
        }
    elif 'lesson plan' in user_message:
        response = {
            'message': 'I\'d be happy to help with lesson planning! Please provide:',
            'questions': ['What subject?', 'What grade level (1-5)?', 'What specific topic?']
        }
    elif 'activities' in user_message:
        response = {
            'message': 'What type of activities would you like suggestions for?',
            'options': ['Classroom Games', 'Group Work', 'Individual Tasks', 'Creative Projects']
        }
    elif 'definitions' in user_message:
        response = {
            'message': 'What term or concept would you like me to define?',
            'placeholder': 'Enter a word or concept...'
        }
    else:
        # Try to extract lesson plan information
        if all(keyword in user_message for keyword in ['subject:', 'grade:', 'topic:']):
            parts = user_message.split()
            subject = grade = topic = ''
            
            for i, part in enumerate(parts):
                if part == 'subject:' and i + 1 < len(parts):
                    subject = parts[i + 1]
                elif part == 'grade:' and i + 1 < len(parts):
                    grade = parts[i + 1]
                elif part == 'topic:' and i + 1 < len(parts):
                    topic = ' '.join(parts[i + 1:])
                    break
            
            if subject and grade and topic:
                lesson_plan = generate_lesson_plan_response(subject.title(), grade, topic)
                response = {
                    'message': lesson_plan,
                    'type': 'lesson_plan'
                }
            else:
                response = {
                    'message': 'Please provide all required information in this format: "subject: [subject] grade: [grade] topic: [topic]"'
                }
        else:
            response = {
                'message': 'I understand you want help! Try saying "hi" to see the main menu or ask me about: lesson planning, activities, or definitions.'
            }
    
    return jsonify(response)

if __name__ == '__main__':
    init_data_files()
    app.run(host='0.0.0.0', port=5000, debug=True)