# AI Teaching Assistant

## Overview

AI Teaching Assistant is a pure AI-powered conversational application designed to help primary school teachers with their daily teaching needs. The application provides an intelligent chatbot interface that supports multimodal interactions including text, voice messages, image analysis, and document processing. Built with a beautiful, colorful interface using pink, yellow, and blue themes, it offers a friendly and intuitive experience for educators.

The application serves as an intelligent companion for teachers, providing instant access to AI-powered assistance for lesson planning, classroom management advice, educational resources, and general teaching support through natural conversation.

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
- **Data Organization**: Separate JSON files for different data types (students, attendance)
- **File Structure**: 
  - `students.json` - Array of specific student names for the classroom roster
  - `attendance.json` - Nested object structure by date and student for daily attendance tracking
- **Data Initialization**: Automatic creation of data directory and files with default values
- **Persistence**: Synchronous file I/O for immediate data consistency

### Core Features Architecture
- **Attendance System**: Manual roster management with daily attendance tracking for specific students (Syeda Sughra Fatima, Mehmoona, Aira, Aliza)
- **Weekly Reporting**: Generate attendance reports with date range selection, accurate statistics calculation, and WhatsApp sharing functionality
- **AI Chatbot**: Pattern-matching conversational interface with predefined responses for lesson planning assistance

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