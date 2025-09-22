# Tutor's Assistant

## Overview

Tutor's Assistant is a comprehensive web application designed to help teachers manage their classrooms efficiently. The application provides tools for attendance tracking, lesson planning, AI-powered grading, quiz generation, and an intelligent chatbot assistant. Built with a clean, colorful interface using pink, yellow, and blue themes, it aims to make teaching tasks more manageable and engaging.

The application serves as a prototype that simulates AI-powered features while providing practical functionality for day-to-day teaching activities including roster management, calendar-based planning, student assessment, and lesson preparation assistance.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Flask for server-side rendering
- **Styling**: Custom CSS with CSS Grid and Flexbox for responsive layouts
- **Color Scheme**: Pink, yellow, and blue theme with CSS custom properties for consistent theming
- **User Interface**: Multi-page application with navigation bar and modular components
- **Client-side Interactivity**: Vanilla JavaScript for dynamic features like chat interface and form interactions

### Backend Architecture
- **Web Framework**: Flask (Python) with route-based architecture
- **Application Structure**: Single-file Flask application with modular function organization
- **Data Flow**: Request-response pattern with form handling and JSON API endpoints
- **Session Management**: Flask sessions with secret key configuration
- **Error Handling**: Flash messages for user feedback and form validation

### Data Storage
- **Storage Method**: File-based JSON storage system
- **Data Organization**: Separate JSON files for different data types (students, attendance, schedule, grades)
- **File Structure**: 
  - `students.json` - Array of student names
  - `attendance.json` - Nested object structure by date and student
  - `schedule.json` - Calendar events with filtering capabilities
  - `grades.json` - Student grades organized by date
- **Data Initialization**: Automatic creation of data directory and files with default values
- **Persistence**: Synchronous file I/O for immediate data consistency

### Core Features Architecture
- **Attendance System**: Manual roster management with daily attendance tracking
- **Calendar Planner**: Month/year navigation with grade and subject filtering
- **Grading System**: Simulated AI grading with placeholder photo upload functionality
- **Quiz Generator**: Form-based assessment creation with templated responses
- **AI Chatbot**: Pattern-matching conversational interface with predefined responses

### Authentication and Authorization
- **Current State**: No authentication system implemented
- **Session Security**: Basic session secret configuration
- **Access Control**: Open access to all features without user authentication

## External Dependencies

### Python Dependencies
- **Flask**: Web framework for routing, templating, and request handling
- **Standard Library**: `json`, `os`, `datetime`, `calendar` for core functionality

### Frontend Dependencies
- **No External Libraries**: Pure HTML, CSS, and JavaScript implementation
- **Font**: System fonts (Segoe UI family) for cross-platform compatibility

### Simulated AI Services
- **Grading AI**: Placeholder implementation that generates random scores
- **Chatbot AI**: Rule-based response system with predefined conversation flows
- **Future Integration Points**: Ready for integration with actual AI services for photo analysis and natural language processing

### File System Dependencies
- **Data Directory**: Local file system for JSON data storage
- **Static Assets**: CSS files served through Flask's static file handling
- **Templates**: Jinja2 template files for dynamic content rendering

### Environment Configuration
- **Session Secret**: Environment variable support with fallback default
- **Development Setup**: Local development server through Flask's built-in server