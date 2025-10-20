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
- **Color Scheme**: Regal & Minimal design system with CSS custom properties for consistent theming
  - Primary Accent: Deep Emerald Green (#004D40) - for prominent text, buttons, and navigation
  - Highlight Accent: Soft Gold (#C8A35C) - for accents, borders, and hover effects
  - Background: Light Off-White Gray (#F9FAFB) - clean, professional canvas
  - Cards: Pure White (#FFFFFF) - sharp, clean card backgrounds
  - Text: Soft Dark Gray (#374151) for body text, Deep Emerald Green for headings
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
- **Single-Day Navigation**: Simplified template creation showing one day at a time with Previous/Next buttons
- **Saturday Toggle**: Optional Saturday inclusion in weekly schedule (5-day or 6-day week)
- Individual entry editing with progress tracking
- Monthly/yearly class count displays
- Database Tables: `weekly_templates` (includes `include_saturday` field), `weekly_template_periods`, `yearly_schedule_entries`

**2. Grading Buddy Module:**
- Class Management: Create and manage multiple classes with student rosters
- Student Management: Bulk add students, edit profiles (name, father/mother name, contact, address, gender)
- Test Creation: Marks-based (e.g., 85/100) or Tick/Cross assessment methods
- Grade Calculation: Automatic percentage and grade computation (A+, A, B, C, D, E, F)
- Test Types: Written Test, Oral Test, Quiz, Assignment, Homework
- Reports: Full progress reports and subject-wise reports for students
- Print Functionality: Print test sheets and results for physical records
- **Data Persistence**: PostgreSQL database with cross-device synchronization
  - Database Table: `grading_data` (stores entire class structure as JSON)
  - Auto-sync: Data automatically saves to database on every change
  - Cross-device: Access same data from any device with same phone number
  - Periodic backup: Auto-save every 2 minutes
  - Logout sync: Data saved to database before logout
  - Simple & Reliable: JSON storage for maximum compatibility

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
- **Yearly Planner UI Redesign (Oct 20)**: Improved weekly template creation interface
  - Single-day view with Previous/Next navigation instead of showing all days at once
  - Saturday toggle option to include/exclude Saturday from weekly schedule
  - Dynamic day counter showing "Day X of Y" based on Saturday preference
  - Backend automatically generates 5-day or 6-day weekly schedules based on preference
  - Improved mobile responsiveness with cleaner, less overwhelming interface
- **Database Persistence (Oct 1)**: Full PostgreSQL integration for Grading Buddy with cross-device sync
  - Automatic save to database on every change
  - Load data from database on login (works across any device)
  - Periodic auto-save every 2 minutes
  - Data synced before logout
  - Backend API endpoints: `/api/grading/save-classes`, `/api/grading/load-classes`
- **Grading Buddy Rebuilt**: Complete new implementation with database-based data management
- **Simplified Authentication**: Phone number only login system
- **Enhanced Features**: Added marks-based and tick/cross assessment methods
- **Print Support**: Full printing capability for tests and progress reports
- **Student Management**: Comprehensive student profile editing with bulk add support

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
- **Deployment Type**: Autoscale deployment for efficient resource usage
- **Production Server**: Gunicorn WSGI server with multiple workers
- **File Persistence**: All uploaded files (photos, textbooks, audio) persist across restarts
- **Configuration**: Gunicorn binds to 0.0.0.0:5000 with port reuse enabled

### Production Database Setup
When you publish the app, Replit automatically provisions a production database:
- **Automatic Setup**: Production database is created automatically when you publish
- **Same Code**: Your existing code continues to work - the DATABASE_URL environment variable automatically points to the production database
- **Data Separation**: Development and production databases are completely separate
- **No Migration Needed**: Your grading_data table structure transfers automatically
- **Cross-Device Sync**: Teachers can login from any device and access their data
- **Security**: Production database has enhanced security and backup features
- **How to Publish**: Click the "Publish" button in Replit - database setup happens automatically