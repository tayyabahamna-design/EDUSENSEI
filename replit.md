# AI Assistant

## Overview

AI Assistant is a pure AI-powered conversational application designed as a helpful, knowledgeable, and conversational AI companion. The application provides an intelligent chatbot interface that supports multimodal interactions including text, voice messages, image analysis, and document processing. Built with a beautiful, colorful interface using pink, yellow, and blue themes, it offers a friendly and intuitive experience for users.

The application serves as a versatile AI assistant, providing instant access to AI-powered help with coding, writing, analysis, creative tasks, general questions, and much more through natural conversation. It maintains assessment capabilities and curriculum navigation features while expanding to serve as a general-purpose AI assistant that adapts to user needs and communication styles.

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
- **Storage Method**: File-based storage for uploaded content (images, documents, audio)
- **File Organization**: Upload directories organized by content type (images, documents, audio, metadata)
- **File Structure**: 
  - `uploads/images/` - User-uploaded images for AI analysis
  - `uploads/docs/` - User-uploaded documents (PDF, DOCX, TXT)
  - `uploads/audio/` - Voice message recordings
  - `uploads/meta/` - Metadata files for tracking uploads
- **Temporary Storage**: Automatic cleanup of files older than 7 days
- **Privacy**: Image metadata stripping for user privacy

### Core Features Architecture
- **AI Chatbot**: Natural language conversation using Google Gemini AI with multimodal support and adaptive personality
- **Voice Messaging**: Audio recording, upload, and transcription capabilities
- **File Upload**: Support for images, documents (PDF, DOCX), and audio files
- **Multimodal AI**: Combined text, image, and document analysis in conversations
- **Content Processing**: Automatic text extraction from PDFs and DOCX files
- **General Purpose**: Helps with coding, writing, analysis, creative tasks, and general questions

### Authentication and Authorization
- **Current State**: No authentication system implemented
- **Session Security**: Basic session secret configuration
- **Access Control**: Open access to all features without user authentication

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