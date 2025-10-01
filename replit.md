# USTAAD DOST - Pakistani Teacher Assistant App

## Overview

USTAAD DOST is a comprehensive teacher assistant application designed specifically for Pakistani educators teaching grades 1-5. The platform combines traditional teaching management with AI-powered educational tools, featuring a soothing eye-friendly design with soft blues, greens, and warm neutrals.

The application serves Pakistani teachers with three core modules: Yearly Planner for curriculum scheduling, Grading Buddy for class and assessment management, and AI Chatbox for curriculum-specific educational content generation. Built as a progressive web app with offline capabilities, it supports the complete Pakistani curriculum (Math, English, Urdu, Islamiyat, General Knowledge, Social Studies, Science) with multilingual support.

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
- **Database**: PostgreSQL with comprehensive schema for Pakistani education system
- **Tables**: Users (teachers), classes (grades 1-5), students, subjects, yearly_plans, scoresheets, student_scores, screener_tasks, ai_chat_history, file_uploads
- **File Storage**: Organized uploads for textbooks, profile photos, and educational resources
- **File Structure**: 
  - `uploads/profiles/` - Teacher profile photos
  - `uploads/textbooks/` - Curriculum textbooks and resources
  - `uploads/audio/` - Voice recordings (Urdu/English)
  - `uploads/images/` - Educational images and captures
- **Security**: Encrypted storage with user access controls
- **Backup**: Automatic data backup with recovery capabilities

### Core Features Architecture
**1. Yearly Planner Module:**
- Weekly schedule templates with auto-population for entire academic year
- Grade-wise (1-5) and subject-specific planning for Pakistani curriculum
- Individual entry editing with progress tracking
- Monthly/yearly class count displays

**2. Grading Buddy Module (Simplified v2.0):**
- Simple class creation without subject requirements
- Bulk student addition (paste multiple names at once)
- Individual student addition with quick input
- Subject-based scoresheet creation (English, Math, Urdu, Islamiyat, Science, Social Studies, General Knowledge)
- Dual assessment types:
  - **Marks-based scoresheets**: Total marks, obtained marks, percentage, and grade calculation
  - **Tick/Cross scoresheets**: Simple completion tracking
- Print functionality for all scoresheets
- Data separation by teacher phone number using localStorage
- Student and scoresheet management per class

**3. AI Chatbox Module:**
- Multimodal inputs: text, camera capture, image upload, voice notes (Urdu/English)
- Pakistani curriculum navigation: Grade → Subject → Chapter → Topic
- Content generation: Lesson Plans, Teaching Strategies, Activities, Assessments
- Specialized tools: Hooks/Games, Definitions, Examples with answer keys
- OpenAI integration with curriculum-specific knowledge base

### Authentication and Authorization
- **Phone Authentication**: Login system using phone number and password with bcrypt hashing
- **Password Security**: All passwords encrypted with bcrypt salt and hash (minimum 6 characters)
- **Password Verification**: Login validates password hash before granting access
- **Profile Management**: Complete teacher profiles with CNIC, address, email, photo upload
- **Session Security**: Secure session cookies with HTTPONLY and SECURE flags enabled
- **Data Privacy**: Individual teacher data isolation with secure access patterns
- **Progressive Features**: Editable profiles with persistent data storage

### Recent Updates (October 2025)
- **Security Enhancements**: Added password hashing and verification to login/registration
- **Grading Buddy Rebuild**: Simplified interface with localStorage-based data management
- **Bug Fixes**: Resolved 67+ code errors including undefined variables and database connection issues

## External Dependencies

### Python Dependencies
- **Flask**: Web framework for routing, templating, and request handling
- **Google Generative AI**: Gemini API for natural language processing and multimodal AI
- **Pillow (PIL)**: Image processing and metadata removal
- **pypdf**: PDF text extraction
- **python-docx**: DOCX document text extraction
- **Standard Library**: `json`, `os`, `datetime`, `uuid`, `mimetypes` for core functionality

### Frontend Dependencies
- **No External Libraries**: Pure HTML, CSS, and JavaScript implementation
- **MediaRecorder API**: Native browser voice recording capabilities
- **Font**: System fonts (Segoe UI family) for cross-platform compatibility

### AI Services Integration
- **Google Gemini**: Production AI service for natural conversation and multimodal analysis
- **Multimodal Processing**: Combined text, image, and document understanding
- **Fallback System**: Educational guidance system when AI is unavailable

### File System Dependencies
- **Upload Directory**: Local file system for temporary file storage with automatic cleanup
- **Static Assets**: CSS files served through Flask's static file handling
- **Templates**: Jinja2 template files for dynamic content rendering

### Environment Configuration
- **Session Secret**: Environment variable support with fallback default
- **Development Setup**: Local development server through Flask's built-in server

### Deployment Configuration
- **Deployment Type**: VM (Virtual Machine) deployment for persistent file storage
- **Production Server**: Gunicorn WSGI server with multiple workers
- **File Persistence**: All uploaded files (photos, textbooks, audio) persist across restarts
- **Configuration**: Gunicorn binds to 0.0.0.0:5000 with port reuse enabled