import streamlit as st
import pandas as pd
import os
import base64
import json
import time
import re
import random
import string
from datetime import datetime, timedelta, date
import io
import tempfile
import requests
from PIL import Image
import matplotlib
matplotlib.use('Agg')
try:
    import pymupdf  # Use pymupdf instead of fitz
except ImportError:
    st.error("Please install PyMuPDF: pip install PyMuPDF")
    pymupdf = None
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

# Set matplotlib style for better looking charts
plt.style.use('seaborn-v0_8')

# Import utilities
from edumate.utils.logger import log_system_event, log_access, log_error, log_audit
from edumate.utils.encryption import Encryptor
from edumate.utils.analytics import Analytics
from edumate.utils.audit import AuditTrail
from edumate.utils.career_planner import CareerPlanner
from edumate.utils.indian_education import IndianEducationSystem
from edumate.utils.exam_manager import ExamManager
from edumate.utils.classroom_manager import ClassroomManager
from edumate.utils.teacher_tools import TeacherTools

# Import services
from edumate.services import (
    GeminiService, 
    GradingService, 
    FeedbackService, 
    PlagiarismService,
    AIService,
    AIGradingService,
    StudyRecommendationsService,
    GroupFormationService,
    LearningPathService,
    MultilingualFeedbackService,
    TeacherAnalyticsService
)

# Load environment variables from .env file
load_dotenv()

# Initialize utilities
encryptor = Encryptor()
analytics = Analytics('data')
audit_trail = AuditTrail('data')
career_planner = CareerPlanner('data')
indian_education = IndianEducationSystem()
exam_manager = ExamManager('data')
classroom_manager = ClassroomManager('data')
teacher_tools = TeacherTools('data')

# Initialize services
gemini_service = GeminiService()  # Initialize with your API key
grading_service = GradingService()
feedback_service = FeedbackService()
plagiarism_service = PlagiarismService()
ai_service = AIService()
ai_grading_service = AIGradingService()
study_recommendations_service = StudyRecommendationsService(gemini_service=gemini_service, data_dir='data')
group_formation_service = GroupFormationService()
learning_path_service = LearningPathService()
multilingual_feedback_service = MultilingualFeedbackService(gemini_service=gemini_service, data_dir='data')
teacher_analytics_service = TeacherAnalyticsService()

# Set page configuration
st.set_page_config(
    page_title="EduMate - AI-Powered Education Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Create teacher tools data directory if it doesn't exist
if not os.path.exists('data/teacher_tools'):
    os.makedirs('data/teacher_tools')

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Create career data directory if it doesn't exist
if not os.path.exists('data/career'):
    os.makedirs('data/career')

# Initialize data files if they don't exist
if not os.path.exists('data/users.json'):
    with open('data/users.json', 'w') as f:
        json.dump([], f)

if not os.path.exists('data/courses.json'):
    with open('data/courses.json', 'w') as f:
        json.dump([], f)

if not os.path.exists('data/assignments.json'):
    with open('data/assignments.json', 'w') as f:
        json.dump([], f)

if not os.path.exists('data/submissions.json'):
    with open('data/submissions.json', 'w') as f:
        json.dump([], f)

# Initialize career planning data files
if not os.path.exists('data/career/skill_matrices.json'):
    with open('data/career/skill_matrices.json', 'w') as f:
        json.dump({
            "software_engineering": {
                "programming": 9,
                "problem_solving": 8,
                "system_design": 7,
                "teamwork": 6,
                "communication": 6
            },
            "data_science": {
                "statistics": 9,
                "programming": 7,
                "data_visualization": 8,
                "machine_learning": 8,
                "communication": 7
            },
            "product_management": {
                "business_acumen": 8,
                "communication": 9,
                "leadership": 8,
                "technical_understanding": 7,
                "problem_solving": 8
            }
        }, f, indent=4)

if not os.path.exists('data/career/course_recommendations.json'):
    with open('data/career/course_recommendations.json', 'w') as f:
        json.dump({
            "programming": [
                {"title": "Python for Everybody", "platform": "Coursera", "url": "https://www.coursera.org/specializations/python"},
                {"title": "The Complete Web Developer Course", "platform": "Udemy", "url": "https://www.udemy.com/course/the-complete-web-developer-course-2"}
            ],
            "data_analysis": [
                {"title": "Data Science Specialization", "platform": "Coursera", "url": "https://www.coursera.org/specializations/jhu-data-science"},
                {"title": "Data Analyst Nanodegree", "platform": "Udacity", "url": "https://www.udacity.com/course/data-analyst-nanodegree--nd002"}
            ],
            "communication": [
                {"title": "Effective Communication", "platform": "LinkedIn Learning", "url": "https://www.linkedin.com/learning/topics/communication"},
                {"title": "Public Speaking", "platform": "Coursera", "url": "https://www.coursera.org/learn/public-speaking"}
            ]
        }, f, indent=4)

# Load data
def load_data(file_name):
    with open(f'data/{file_name}.json', 'r') as f:
        return json.load(f)

def save_data(data, file_name):
    with open(f'data/{file_name}.json', 'w') as f:
        json.dump(data, f, indent=4)

# User authentication functions
def register_user(email, password, name, role):
    users = load_data('users')
    
    # Check if user already exists
    if any(user['email'] == email for user in users):
        return False, "Email already registered"
    
    # Create new user
    new_user = {
        'id': len(users) + 1,
        'email': email,
        'password': password,  # In a real app, you would hash this
        'name': name,
        'role': role,
        'created_at': datetime.now().isoformat()
    }
    
    users.append(new_user)
    save_data(users, 'users')
    return True, "Registration successful"

def login_user(email, password):
    """Login user with logging"""
    users = load_data('users')
    
    for user in users:
        if user['email'] == email and user['password'] == password:
            log_access(user['id'], "User logged in")
            return True, user
    
    log_error("Failed login attempt", {"email": email})
    return False, "Invalid email or password"

# Course management functions
def create_course(name, code, description, teacher_id, start_date, end_date):
    courses = load_data('courses')
    
    # Check if course code already exists
    if any(course['code'] == code for course in courses):
        return False, "Course code already exists"
    
    # Create new course
    new_course = {
        'id': len(courses) + 1,
        'name': name,
        'code': code,
        'description': description,
        'teacher_id': teacher_id,
        'start_date': start_date,
        'end_date': end_date,
        'created_at': datetime.now().isoformat(),
        'students': []
    }
    
    courses.append(new_course)
    save_data(courses, 'courses')
    return True, "Course created successfully"

def get_teacher_courses(teacher_id):
    courses = load_data('courses')
    return [course for course in courses if course['teacher_id'] == teacher_id]

def get_student_courses(student_id):
    courses = load_data('courses')
    return [course for course in courses if student_id in course['students']]

def enroll_student(course_id, student_id):
    courses = load_data('courses')
    
    for i, course in enumerate(courses):
        if course['id'] == course_id:
            if student_id not in course['students']:
                courses[i]['students'].append(student_id)
                save_data(courses, 'courses')
                return True, "Enrolled successfully"
            else:
                return False, "Already enrolled"
    
    return False, "Course not found"

# Assignment management functions
def create_assignment(title, description, course_id, teacher_id, due_date, points=100):
    assignments = load_data('assignments')
    
    # Create new assignment
    new_assignment = {
        'id': len(assignments) + 1,
        'title': title,
        'description': description,
        'course_id': course_id,
        'teacher_id': teacher_id,
        'due_date': due_date,
        'points': points,
        'created_at': datetime.now().isoformat()
    }
    
    assignments.append(new_assignment)
    save_data(assignments, 'assignments')
    return True, "Assignment created successfully"

def get_course_assignments(course_id):
    assignments = load_data('assignments')
    return [assignment for assignment in assignments if assignment['course_id'] == course_id]

def delete_assignment(assignment_id, teacher_id):
    """Delete assignment with audit trail"""
    assignments = load_data('assignments')
    
    # Find the assignment
    assignment_index = None
    for i, assignment in enumerate(assignments):
        if assignment['id'] == assignment_id:
            if assignment['teacher_id'] != teacher_id:
                return False, "You don't have permission to delete this assignment"
            assignment_index = i
            break
    
    if assignment_index is None:
        return False, "Assignment not found"
    
    # Check if there are any submissions for this assignment
    submissions = load_data('submissions')
    assignment_submissions = [sub for sub in submissions if sub['assignment_id'] == assignment_id]
    
    if assignment_submissions:
        # If there are submissions, we should handle them
        # Option 1: Delete all submissions (implemented here)
        # Option 2: Prevent deletion if submissions exist (alternative approach)
        
        # Delete all related submissions and their files
        for submission in assignment_submissions:
            # Delete any attached files
            if submission.get('file_info') and submission['file_info'].get('file_path'):
                file_path = submission['file_info']['file_path']
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting file: {e}")
        
        # Remove all submissions for this assignment
        submissions = [sub for sub in submissions if sub['assignment_id'] != assignment_id]
        save_data(submissions, 'submissions')
    
    # Remove the assignment
    deleted_assignment = assignments.pop(assignment_index)
    save_data(assignments, 'assignments')
    
    audit_trail.add_entry(
        teacher_id,
        "delete_assignment",
        {"assignment_id": assignment_id}
    )
    
    return True, "Assignment deleted successfully"

def submit_assignment(assignment_id, student_id, content, uploaded_file=None):
    submissions = load_data('submissions')
    
    # Check if already submitted
    if any(sub['assignment_id'] == assignment_id and sub['student_id'] == student_id for sub in submissions):
        return False, "You have already submitted this assignment"
    
    # Handle file upload if provided
    file_info = None
    if uploaded_file is not None:
        # Create directory for this assignment if it doesn't exist
        assignment_dir = f"uploads/assignment_{assignment_id}"
        if not os.path.exists(assignment_dir):
            os.makedirs(assignment_dir)
        
        # Save the file
        file_path = f"{assignment_dir}/{student_id}_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Store file information
        file_info = {
            'filename': uploaded_file.name,
            'file_path': file_path,
            'file_type': uploaded_file.type,
            'file_size': uploaded_file.size
        }
    
    # Create new submission
    new_submission = {
        'id': len(submissions) + 1,
        'assignment_id': assignment_id,
        'student_id': student_id,
        'content': content,
        'file_info': file_info,
        'submitted_at': datetime.now().isoformat(),
        'status': 'submitted',
        'score': None,
        'feedback': None,
        'ai_feedback': None
    }
    
    submissions.append(new_submission)
    save_data(submissions, 'submissions')
    return True, "Assignment submitted successfully"

def grade_submission(submission_id, score, feedback, use_ai_grading=False):
    submissions = load_data('submissions')
    
    for submission in submissions:
        if submission['id'] == submission_id:
            submission['score'] = score
            submission['feedback'] = feedback
            if use_ai_grading:
                submission['ai_feedback'] = generate_ai_feedback(submission)
            submission['status'] = 'graded'
            submission['graded_at'] = datetime.now().isoformat()
            save_data(submissions, 'submissions')
            return True, "Submission graded successfully"
    
    return False, "Submission not found"

def delete_submission(submission_id, student_id):
    """Delete a student's submission if it hasn't been graded yet."""
    submissions = load_data('submissions')
    
    # Find the submission
    submission_index = None
    for i, submission in enumerate(submissions):
        if submission['id'] == submission_id and submission['student_id'] == student_id:
            submission_index = i
            break
    
    if submission_index is None:
        return False, "Submission not found"
    
    # Check if the submission has been graded
    if submissions[submission_index]['status'] in ['graded', 'auto-graded']:
        return False, "Cannot delete a submission that has already been graded"
    
    # Delete the file if it exists
    if submissions[submission_index].get('file_info'):
        file_path = submissions[submission_index]['file_info'].get('file_path')
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                # Continue even if file deletion fails
                print(f"Error deleting file: {e}")
    
    # Remove the submission from the list
    deleted_submission = submissions.pop(submission_index)
    save_data(submissions, 'submissions')
    
    return True, "Submission deleted successfully"

# Add Gemini API configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
GEMINI_MODELS = {
    "gemini-1.5-pro": "Improved model with better accuracy",
    "gemini-2.0-pro": "Latest model with best performance",
    "gemini-2.0-flash": "Advanced model with enhanced comprehension",
    "gemini-2.0-vision": "Vision-capable model for image analysis"
}
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1/models"

def analyze_with_gemini(content_type, file_path, prompt, mime_type, model="gemini-2.0-pro"):
    """Analyze content using specific Gemini model"""
    if not GEMINI_API_KEY:
        st.error("Gemini API key not found. Please add GEMINI_API_KEY to your .env file.")
        return "Error: Valid Gemini API key not configured."
    
    try:
        # Read the file and encode it as base64
        with open(file_path, "rb") as file:
            file_data = base64.b64encode(file.read()).decode('utf-8')
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": file_data
                            }
                        }
                    ]
                }
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        
        api_url = f"{GEMINI_API_BASE_URL}/{model}:generateContent"
        
        st.info(f"Using model: {model}...")
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                if 'content' in result['candidates'][0] and 'parts' in result['candidates'][0]['content']:
                    for part in result['candidates'][0]['content']['parts']:
                        if 'text' in part:
                            st.success(f"Successfully used model: {model}")
                            return part['text']
        
        return f"Model {model} failed: {response.text}"
        
    except Exception as e:
        return f"Error analyzing with {model}: {str(e)}"

def analyze_image_with_gemini(image_path, prompt):
    """
    Analyze an image using Google Gemini API
    """
    return analyze_with_gemini('image', image_path, prompt, 'image/jpeg')

def analyze_pdf_with_gemini(pdf_path, prompt):
    """
    Analyze a PDF file directly using Google Gemini API
    This function sends the PDF file directly to Gemini without extracting images
    """
    return analyze_with_gemini('pdf', pdf_path, prompt, 'application/pdf')

def extract_images_from_pdf(pdf_path, output_dir):
    """Extract images from a PDF file using PyMuPDF"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_paths = []
    
    try:
        if pymupdf is None:
            raise ImportError("PyMuPDF is not installed")

        # Open the PDF using PyMuPDF
        pdf_document = pymupdf.open(pdf_path)
        
        # Rest of the function remains the same
        for page_num, page in enumerate(pdf_document):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)
                
                image_paths.append(image_path)
        
        # If no images found, try extracting as whole page images
        if not image_paths:
            for page_num, page in enumerate(pdf_document):
                pix = page.get_pixmap()
                image_filename = f"page{page_num+1}.png"
                image_path = os.path.join(output_dir, image_filename)
                pix.save(image_path)
                image_paths.append(image_path)
        
        return image_paths

    except ImportError as e:
        st.error(f"Error: {str(e)}. Please install PyMuPDF using: pip install PyMuPDF")
        return []
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
        return []

# Add imports for our enhanced grading services at the top of the file
from edumate.services.grading_service import GradingService
from edumate.services.plagiarism_service import PlagiarismService
from edumate.services.feedback_service import FeedbackService
from edumate.services.gemini_service import GeminiService

def auto_grade_submission(submission_id):
    """
    Automatically grade a submission using AI.
    Returns: (success, message)
    """
    try:
        # Load submissions and get the specific submission
    submissions = load_data('submissions')
        submission = next((s for s in submissions if s['id'] == submission_id), None)
    
    if not submission:
            return False, "Submission not found."
    
        # Get the assignment details
    assignment = get_assignment_by_id(submission['assignment_id'])
        if not assignment:
            return False, "Assignment not found."
    
        # Use the GradingService to grade the submission
    file_content = ""
    file_analysis = ""
    
        # Get file content if available
    if submission.get('file_info'):
            file_path = submission['file_info']['file_path']
            file_type = submission['file_info']['file_type']
        
            try:
                # Extract text based on file type
                if file_type == 'application/pdf':
                    file_content = extract_text_from_pdf(file_path)
                elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    file_content = extract_text_from_docx(file_path)
                else:
                    # For other file types, try to read as text
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()
                
                # Analyze file content
                file_analysis = analyze_file_content(file_content, submission['file_info']['filename'])
            except Exception as e:
                log_error(f"Error processing file for auto-grading: {str(e)}")
                file_analysis = f"Error analyzing file: {str(e)}"
    
        # Get combined content
        combined_content = submission['content']
        if file_content:
            combined_content += "\n\n" + file_content
            
        # Get assignment rubric
        rubric = assignment.get('rubric', {})
        if not rubric:
            # Generate a simple default rubric
            rubric = {
                "criteria": [
                    {"name": "Content", "weight": 40, "description": "Completeness and quality of content"},
                    {"name": "Understanding", "weight": 30, "description": "Demonstrates understanding of concepts"},
                    {"name": "Presentation", "weight": 20, "description": "Organization and clarity"},
                    {"name": "Technical", "weight": 10, "description": "Technical correctness"}
                ],
                "total_points": assignment.get("points", 100)
            }
            
        # Get grading result using our service
        grading_result = grading_service.grade_submission(
            content=combined_content,
            assignment_details={
                "title": assignment.get("title", ""),
                "description": assignment.get("description", ""),
                "points": assignment.get("points", 100),
                "rubric": rubric
            },
            file_analysis=file_analysis,
            student_id=submission.get("student_id", ""),
            submission_type=assignment.get("type", "assignment")
        )
        
        # Check for plagiarism if the service is available
        plagiarism_result = {"score": 0, "matches": []}
        try:
            plagiarism_result = plagiarism_service.check_plagiarism(
                content=combined_content,
                student_id=submission.get("student_id", ""),
                assignment_id=assignment.get("id", "")
            )
            
            # Adjust score based on plagiarism if detected
            if plagiarism_result["score"] > 30:  # Over 30% plagiarism
                reduction = min(grading_result["score"] * (plagiarism_result["score"]/200), 25)  # Max 25% reduction
                grading_result["score"] -= reduction
                grading_result["feedback"] += f"\n\n⚠️ **Plagiarism Warning**: This submission shows {plagiarism_result['score']:.1f}% similarity with existing sources. Points have been deducted accordingly."
                
        except Exception as e:
            log_error(f"Error checking plagiarism: {str(e)}")
            
        # Generate AI feedback (comprehensive version from service)
        ai_feedback = grading_result.get("feedback", "")
        if not ai_feedback:
            # Fallback to legacy feedback generator if service doesn't provide feedback
            ai_feedback = generate_ai_feedback(submission, file_content, file_analysis)
        
        # Update the submission with AI-graded score and feedback
        submission['score'] = round(grading_result["score"])
        submission['feedback'] = grading_result.get("comments", "")
        submission['ai_feedback'] = ai_feedback
        submission['ai_suggestions'] = grading_result.get("suggestions", {})
        submission['rubric_scores'] = grading_result.get("rubric_scores", {})
        submission['graded_by'] = 'ai'
        submission['graded_at'] = datetime.now().isoformat()
        submission['plagiarism_data'] = plagiarism_result
        
        # Save the updated submissions
            save_data(submissions, 'submissions')
        
        # Log the action
        log_audit(
            user_id=0,  # 0 indicates system/AI
            action='grade',
            resource_type='submission',
            resource_id=submission_id,
            success=True,
            details=f"Auto-graded submission with score: {submission['score']}"
        )
        
        return True, f"Submission auto-graded successfully with score: {submission['score']}"
        
    except Exception as e:
        error_message = f"Error auto-grading submission: {str(e)}"
        log_error(error_message)
        return False, error_message

def extract_feedback_points(feedback):
    """Extract key feedback points from grading feedback."""
    points = []
    
    # Extract strengths section
    strengths_match = re.search(r'(?:STRENGTHS|Strengths):(.*?)(?:\n\n|\n[A-Z]|$)', feedback, re.DOTALL | re.IGNORECASE)
    if strengths_match:
        strengths_text = strengths_match.group(1)
        strength_points = re.findall(r'[-*•]\s*(.*?)(?:\n[-*•]|\n\n|$)', strengths_text, re.DOTALL)
        for point in strength_points:
            if point.strip():
                points.append({"type": "strength", "text": point.strip()})
                
    # Extract weaknesses/areas for improvement section
    weaknesses_match = re.search(r'(?:WEAKNESSES|AREAS FOR IMPROVEMENT|Areas for improvement|Weaknesses):(.*?)(?:\n\n|\n[A-Z]|$)', 
                                feedback, re.DOTALL | re.IGNORECASE)
    if weaknesses_match:
        weaknesses_text = weaknesses_match.group(1)
        weakness_points = re.findall(r'[-*•]\s*(.*?)(?:\n[-*•]|\n\n|$)', weaknesses_text, re.DOTALL)
        for point in weakness_points:
            if point.strip():
                points.append({"type": "weakness", "text": point.strip()})
    
    return points

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    # In a real app, you would use a library like PyPDF2 or pdfplumber
    # For this simplified version, we'll return a placeholder
    return f"[PDF content extracted from {os.path.basename(file_path)}]"

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file"""
    # In a real app, you would use a library like python-docx
    # For this simplified version, we'll return a placeholder
    return f"[DOCX content extracted from {os.path.basename(file_path)}]"

def analyze_file_content(content, filename):
    """Analyze the content of a file"""
    # This is a simplified analysis
    file_ext = os.path.splitext(filename)[1].lower()
    
    analysis = f"## File Analysis: {filename}\n\n"
    
    # Word count
    words = content.split()
    word_count = len(words)
    analysis += f"- Word count: {word_count}\n"
    
    # Sentence count
    sentences = re.split(r'[.!?]+', content)
    sentence_count = len([s for s in sentences if s.strip()])
    analysis += f"- Sentence count: {sentence_count}\n"
    
    # Average sentence length
    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
        analysis += f"- Average sentence length: {avg_sentence_length:.1f} words\n"
    
    # File type specific analysis
    if file_ext == '.pdf':
        analysis += "- Document appears to be a PDF, which typically indicates a formal submission.\n"
    elif file_ext == '.docx':
        analysis += "- Document is in Word format, which is appropriate for academic submissions.\n"
    elif file_ext == '.txt':
        analysis += "- Document is in plain text format. Consider using a more formatted document type for future submissions.\n"
    
    return analysis

def generate_ai_feedback(submission, file_content="", file_analysis="", gemini_analysis=""):
    """Generate AI feedback for a submission"""
    # Get the assignment and student details
    assignment = get_assignment_by_id(submission['assignment_id'])
    student = get_user_by_id(submission['student_id'])
    
    # Analyze the submission content
    content = submission['content']
    word_count = len(content.split())
    
    # Generate feedback based on content length and quality
    strengths = []
    improvements = []
    
    # Text content analysis
    if word_count < 50:
        improvements.append("Your text submission is quite brief. Consider expanding your answer with more details.")
    else:
        strengths.append("You provided a detailed text response with good length.")
    
    if "because" in content.lower() or "therefore" in content.lower():
        strengths.append("Good use of reasoning and logical connections in your answer.")
    else:
        improvements.append("Try to include more reasoning and logical connections in your answer.")
    
    if len(content.split('.')) > 5:
        strengths.append("Good structure with multiple sentences in your text submission.")
    else:
        improvements.append("Consider structuring your text answer with more complete sentences.")
    
    # File submission analysis
    if submission.get('file_info'):
        file_info = submission.get('file_info')
        strengths.append(f"You submitted a file ({file_info['filename']}) which demonstrates thoroughness.")
        
        # Add file-specific feedback
        if file_info['file_type'].endswith('pdf'):
            strengths.append("Your PDF submission is in a professional format suitable for academic work.")
            
            # If we have Gemini analysis, it means it was a handwritten or complex PDF
            if gemini_analysis:
                strengths.append("Your PDF was analyzed directly by our AI system, including any handwritten content.")
                
                # Check if the analysis mentions handwritten content
                if "handwritten" in gemini_analysis.lower() or "handwriting" in gemini_analysis.lower():
                    strengths.append("Your handwritten work shows dedication and personal effort in your submission.")
        elif file_info['file_type'].endswith('docx'):
            strengths.append("Your Word document submission follows standard academic formatting.")
        elif file_info['file_type'].endswith('txt'):
            improvements.append("Consider using a more formatted document type (like PDF or DOCX) for future submissions.")
    else:
        improvements.append("Consider attaching a file with your submission for more comprehensive work.")
    
    # Generate personalized feedback
    feedback = f"# AI-Generated Feedback for {student['name']}\n\n"
    feedback += f"## Assignment: {assignment['title']}\n\n"
    
    feedback += "### Strengths:\n"
    for strength in strengths:
        feedback += f"- {strength}\n"
    
    feedback += "\n### Areas for Improvement:\n"
    for improvement in improvements:
        feedback += f"- {improvement}\n"
    
    # Add file analysis if available
    if file_analysis:
        feedback += f"\n{file_analysis}\n"
    
    # Add Gemini analysis of PDF content if available
    if gemini_analysis:
        feedback += f"\n{gemini_analysis}\n"
    
    feedback += "\n### Summary:\n"
    feedback += "This automated feedback is based on an analysis of your submission. "
    
    if submission.get('file_info'):
        if gemini_analysis:
            feedback += "Your PDF was analyzed directly using Google Gemini AI, which can process both text and handwritten content. "
        else:
            feedback += "Both your text response and uploaded file were evaluated. "
    else:
        feedback += "Your text response was evaluated. "
    
    feedback += "It's designed to help you improve your work. "
    feedback += "Please review the teacher's manual feedback as well for more personalized guidance."
    
    return feedback

def get_file_download_link(file_path, filename):
    """Generate a download link for a file"""
    if not os.path.exists(file_path):
        return "File not found"
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def check_api_key_status():
    """Check if the Gemini API key is configured"""
    if not GEMINI_API_KEY:
        return "❌ Not configured", "Please add GEMINI_API_KEY to your .env file."
    else:
        # Only show that the key is configured, not the actual key
        return "✅ Configured", "API key is set and ready to use."

def get_assignment_submissions(assignment_id):
    submissions = load_data('submissions')
    return [sub for sub in submissions if sub['assignment_id'] == assignment_id]

def get_student_submissions(student_id):
    submissions = load_data('submissions')
    return [sub for sub in submissions if sub['student_id'] == student_id]

# Helper functions
def get_user_by_id(user_id):
    users = load_data('users')
    for user in users:
        if user['id'] == user_id:
            return user
    return None

def get_course_by_id(course_id):
    courses = load_data('courses')
    for course in courses:
        if course['id'] == course_id:
            return course
    return None

def get_assignment_by_id(assignment_id):
    assignments = load_data('assignments')
    for assignment in assignments:
        if assignment['id'] == assignment_id:
            return assignment
    return None

def get_submission_by_id(submission_id):
    submissions = load_data('submissions')
    for submission in submissions:
        if submission['id'] == submission_id:
            return submission
    return None

# Navigation functions
def set_page(page):
    st.session_state.current_page = page

# UI Components
def show_login_page():
    st.title("Login to EduMate")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            success, result = login_user(email, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.current_user = result
                st.session_state.current_page = 'dashboard'
                st.rerun()
            else:
                st.error(result)
    
    st.markdown("---")
    st.write("Don't have an account?")
    if st.button("Register", key="login_register_btn"):
        st.session_state.current_page = 'register'
        st.rerun()

def show_register_page():
    st.title("Register for EduMate")
    
    with st.form("register_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["teacher", "student"])
        submit = st.form_submit_button("Register")
        
        if submit:
            if not name or not email or not password:
                st.error("Please fill in all fields")
            else:
                success, message = register_user(email, password, name, role)
                if success:
                    st.success(message)
                    st.session_state.current_page = 'login'
                    st.rerun()
                else:
                    st.error(message)
    
    st.markdown("---")
    st.write("Already have an account?")
    if st.button("Login", key="register_login_btn"):
        st.session_state.current_page = 'login'
        st.rerun()

def show_dashboard():
    st.title(f"Welcome, {st.session_state.current_user['name']}!")
    
    # Display different content based on user role
    if st.session_state.current_user['role'] == 'teacher':
        show_teacher_dashboard()
    else:
        show_student_dashboard()

def show_teacher_dashboard():
    # Get teacher's courses
    teacher_id = st.session_state.current_user['id']
    courses = get_teacher_courses(teacher_id)
    
    st.title("Teacher Dashboard")
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["My Courses", "Create Course", "Create Test", "Analytics", "Teacher Tools"])
    
    with tab1:
        st.header("My Courses")
        if not courses:
            st.info("You haven't created any courses yet.")
        else:
            for course in courses:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{course['name']} ({course['code']})")
                    st.write(course['description'])
                    st.write(f"**Period:** {course['start_date']} to {course['end_date']}")
                    
                    # Get enrollment count
                    enrolled_students = [s for s in course.get('students_enrolled', []) 
                                        if isinstance(s, dict) and s.get('status') == 'active']
                    st.write(f"**Students enrolled:** {len(enrolled_students)}")
                    
                    # Get assignment count
                    assignments = get_course_assignments(course['id'])
                    st.write(f"**Assignments:** {len(assignments)}")
                
                with col2:
                    if st.button("View Course", key=f"view_course_{course['id']}"):
                        st.session_state.current_course = course
                        st.session_state.current_page = 'course_detail'
                        st.rerun()
                    
                    if st.button("Add Assignment", key=f"add_assignment_{course['id']}"):
                        st.session_state.current_course = course
                        st.session_state.current_page = 'create_assignment'
                        st.rerun()
                
                st.divider()
    
    with tab2:
        st.header("Create New Course")
        with st.form(key="create_course_form"):
            course_name = st.text_input("Course Name")
            course_code = st.text_input("Course Code")
            course_description = st.text_area("Description")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
            
            submit_button = st.form_submit_button("Create Course")
            
            if submit_button:
                if not course_name or not course_code:
                    st.error("Course name and code are required.")
                elif start_date >= end_date:
                    st.error("End date must be after start date.")
                else:
                    # Create the course
                    new_course = create_course(
                        name=course_name,
                        code=course_code,
                        description=course_description,
                        teacher_id=teacher_id,
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat()
                    )
                    
                    if new_course:
                        st.success(f"Course '{course_name}' created successfully!")
                        # Log the action
                        log_audit(
                            teacher_id,
                            'create',
                            'course',
                            new_course['id'],
                            True,
                            f"Created course: {course_name}"
                        )
                        # Refresh the page
                        st.rerun()
                    else:
                        st.error("Failed to create course. Please try again.")
    
    with tab3:
        show_test_creator()
    
    with tab4:
        st.header("Analytics")
        
        # Course selection for analytics
        if not courses:
            st.info("You need to create courses to view analytics.")
        else:
            selected_course = st.selectbox(
                "Select Course",
                options=courses,
                format_func=lambda x: f"{x['name']} ({x['code']})",
                key="analytics_course_select"  # Add unique key here
            )
            
            if selected_course:
                # Get assignments for the selected course
                course_assignments = get_course_assignments(selected_course['id'])
                
                if not course_assignments:
                    st.info("No assignments found for this course.")
                else:
                    # Display assignment completion stats
                    st.subheader("Assignment Completion")
                    
                    # Calculate completion rates
                    completion_data = []
                    for assignment in course_assignments:
                        submissions = get_assignment_submissions(assignment['id'])
                        enrolled_count = len([s for s in selected_course.get('students_enrolled', []) 
                                            if isinstance(s, dict) and s.get('status') == 'active'])
                        
                        if enrolled_count > 0:
                            completion_rate = (len(submissions) / enrolled_count) * 100
                        else:
                            completion_rate = 0
                        
                        completion_data.append({
                            'Assignment': assignment['title'],
                            'Completion Rate (%)': completion_rate,
                            'Submissions': len(submissions),
                            'Total Students': enrolled_count
                        })
                    
                    if completion_data:
                        completion_df = pd.DataFrame(completion_data)
                        st.bar_chart(completion_df.set_index('Assignment')['Completion Rate (%)'])
                        st.dataframe(completion_df)
                    
                    # Display grade distribution
                    st.subheader("Grade Distribution")
                    
                    # Calculate grade distribution
                    all_grades = []
                    for assignment in course_assignments:
                        submissions = get_assignment_submissions(assignment['id'])
                        for submission in submissions:
                            if submission.get('score') is not None:
                                all_grades.append({
                                    'Assignment': assignment['title'],
                                    'Score': submission['score'],
                                    'Student': get_user_by_id(submission['student_id'])['name']
                                })
                    
                    if all_grades:
                        grades_df = pd.DataFrame(all_grades)
                        
                        # Histogram of grades
                        fig, ax = plt.subplots()
                        ax.hist(grades_df['Score'], bins=10, range=(0, 100))
                        ax.set_xlabel('Score')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Grade Distribution')
                        st.pyplot(fig)
                        
                        # Average scores by assignment
                        st.subheader("Average Scores by Assignment")
                        avg_scores = grades_df.groupby('Assignment')['Score'].mean().reset_index()
                        st.bar_chart(avg_scores.set_index('Assignment')['Score'])
    
    with tab5:
        show_teacher_tools()

def show_teacher_tools():
    """Display teacher tools and resources."""
    st.header("Teacher Tools")
    
    # Create tabs for different teacher tools
    tools_tab1, tools_tab2, tools_tab3, tools_tab4 = st.tabs([
        "Lesson Planning", 
        "Assessment Tools", 
        "Resource Library",
        "Professional Development"
    ])
    
    with tools_tab1:
        st.subheader("Lesson Planning")
        
        # Get lesson plan templates from teacher tools
        lesson_templates = teacher_tools.get_lesson_templates()
        
        if lesson_templates:
            st.write("Select a lesson plan template to get started:")
            
            for template in lesson_templates:
                with st.expander(template.get('name', 'Unnamed Template')):
                    st.write(f"**Description:** {template.get('description', 'No description available.')}")
                    st.write(f"**Subject:** {template.get('subject', 'General')}")
                    st.write(f"**Grade Level:** {template.get('grade_level', 'All levels')}")
                    
                    # Display template structure
                    if template.get('structure'):
                        st.write("**Template Structure:**")
                        for section in template.get('structure'):
                            st.write(f"- {section}")
                    
                    # Download template button
                    if st.button("Use This Template", key=f"use_template_{template.get('id', '0')}"):
                        # In a real app, this would download or open the template
                        st.success(f"Template '{template.get('name')}' selected. You can now customize it.")
                        
                        # Show template form
                        with st.form(f"lesson_plan_{template.get('id', '0')}"):
                            st.write("### Customize Your Lesson Plan")
                            
                            lesson_title = st.text_input("Lesson Title")
                            lesson_objectives = st.text_area("Learning Objectives")
                            lesson_duration = st.number_input("Duration (minutes)", min_value=15, max_value=180, value=45, step=5)
                            
                            # Materials needed
                            materials = st.text_area("Materials Needed")
                            
                            # Lesson structure
                            st.write("### Lesson Structure")
                            
                            intro = st.text_area("Introduction/Warm-up (5-10 minutes)")
                            main_activity = st.text_area("Main Activity (20-30 minutes)")
                            conclusion = st.text_area("Conclusion/Wrap-up (5-10 minutes)")
                            assessment = st.text_area("Assessment/Evaluation")
                            
                            # Additional notes
                            notes = st.text_area("Additional Notes")
                            
                            # Submit button
                            submit_lesson = st.form_submit_button("Save Lesson Plan")
                            
                            if submit_lesson:
                                if not lesson_title:
                                    st.error("Please enter a lesson title.")
                                else:
                                    # In a real app, this would save the lesson plan
                                    st.success(f"Lesson plan '{lesson_title}' saved successfully!")
        else:
            st.info("No lesson plan templates available.")
        
        # AI Lesson Plan Generator
        st.subheader("AI Lesson Plan Generator")
        
        with st.form("ai_lesson_generator"):
            st.write("Let AI help you create a lesson plan based on your requirements.")
            
            # Required fields with red asterisks
            st.markdown("**Required Fields**")
            col1, col2 = st.columns([10, 1])
            with col1:
                ai_subject = st.text_input("Subject")
            with col2:
                st.markdown('<p style="color:red; font-size:20px; margin-top:15px">*</p>', unsafe_allow_html=True)
                
            col1, col2 = st.columns([10, 1])
            with col1:
                ai_topic = st.text_input("Specific Topic")
            with col2:
                st.markdown('<p style="color:red; font-size:20px; margin-top:15px">*</p>', unsafe_allow_html=True)
            
            # Optional fields
            st.markdown("**Optional Fields**")
            ai_grade = st.selectbox("Grade Level", ["Elementary", "Middle School", "High School", "College"])
            ai_duration = st.number_input("Lesson Duration (minutes)", min_value=15, max_value=180, value=45, step=5)
            
            # Special requirements
            ai_requirements = st.text_area("Special Requirements or Notes")
            
            # Generate button
            generate_button = st.form_submit_button("Generate Lesson Plan")
            
            if generate_button:
                if not ai_subject or not ai_topic:
                    st.error("Please fill in all required fields (marked with red asterisk).")
                else:
                    with st.spinner("Generating lesson plan..."):
                        # Simulate AI processing
                        import time
                        time.sleep(3)
                        
                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
                        
                        st.write(f"## {ai_topic} - {ai_grade} Lesson Plan")
                        st.write(f"**Subject:** {ai_subject}")
                        st.write(f"**Duration:** {ai_duration} minutes")
                        
                        # Generate more relevant content based on subject and topic
                        if ai_subject.lower() in ["math", "mathematics"]:
                            objectives = [
                                f"Explain the key concepts and principles of {ai_topic} in their own words",
                                f"Solve various problems involving {ai_topic} using appropriate techniques",
                                f"Apply {ai_topic} concepts to real-world situations and explain their relevance"
                            ]
                            materials = [
                                "Graphing calculators or appropriate math tools",
                                f"Worksheets with {ai_topic} problems of varying difficulty",
                                f"Visual aids showing {ai_topic} concepts and applications"
                            ]
                            activities = [
                                f"Begin by asking students what they already know about {ai_topic}",
                                f"Demonstrate how {ai_topic} works using interactive examples",
                                f"Guide students through increasingly complex {ai_topic} problems",
                                f"Have students work in pairs to explain {ai_topic} to each other"
                            ]
                            assessment = [
                                f"Ask students to explain a {ai_topic} concept in their own words",
                                f"Have students solve 3-5 problems involving {ai_topic}",
                                f"Assign homework that connects {ai_topic} to real-world applications"
                            ]
                            topic_intro = f"Begin the lesson by connecting {ai_topic} to something students are familiar with. For example, you might discuss how {ai_topic} appears in everyday situations or how it builds on concepts they already know. Ask open-ended questions to gauge their current understanding."
                        elif ai_subject.lower() in ["science", "biology", "chemistry", "physics"]:
                            objectives = [
                                f"Describe the fundamental principles of {ai_topic} using scientific terminology",
                                f"Design and conduct investigations to explore {ai_topic}",
                                f"Analyze data related to {ai_topic} and draw evidence-based conclusions"
                            ]
                            materials = [
                                f"Laboratory equipment for {ai_topic} demonstrations or experiments",
                                f"Scientific models or diagrams illustrating {ai_topic}",
                                "Data collection sheets and analysis tools"
                            ]
                            activities = [
                                f"Start with an intriguing question or demonstration about {ai_topic}",
                                f"Guide students through a hands-on investigation of {ai_topic}",
                                f"Have students collect and analyze data related to {ai_topic}",
                                f"Facilitate a discussion about the implications of {ai_topic}"
                            ]
                            assessment = [
                                f"Have students create a concept map showing relationships within {ai_topic}",
                                f"Ask students to design their own investigation related to {ai_topic}",
                                f"Assign a short writing task explaining how {ai_topic} relates to other scientific concepts"
                            ]
                            topic_intro = f"Begin with a demonstration or thought-provoking question about {ai_topic} that challenges students' preconceptions. For example, you might show a surprising phenomenon related to {ai_topic} and ask students to explain what they observe. This creates curiosity and sets the stage for deeper exploration."
                        elif ai_subject.lower() in ["english", "language arts", "literature"]:
                            objectives = [
                                f"Analyze key elements and themes of {ai_topic} using textual evidence",
                                f"Evaluate different perspectives and interpretations of {ai_topic}",
                                f"Express thoughtful ideas about {ai_topic} through discussion and writing"
                            ]
                            materials = [
                                f"Text excerpts or complete works related to {ai_topic}",
                                f"Discussion prompts and guiding questions about {ai_topic}",
                                "Writing materials or digital tools for response"
                            ]
                            activities = [
                                f"Begin with a compelling quote or passage about {ai_topic}",
                                f"Guide students through close reading of text related to {ai_topic}",
                                f"Facilitate a structured discussion exploring different aspects of {ai_topic}",
                                f"Have students write reflectively about {ai_topic}"
                            ]
                            assessment = [
                                f"Ask students to analyze a passage related to {ai_topic} using specific textual evidence",
                                f"Evaluate student participation in discussion about {ai_topic}",
                                f"Collect written responses that demonstrate understanding of {ai_topic}"
                            ]
                            topic_intro = f"Begin by connecting {ai_topic} to students' experiences or current events. You might ask students to reflect on their own encounters with themes related to {ai_topic} or show how {ai_topic} remains relevant today. This personal connection helps students see the value in studying this topic."
                        elif ai_subject.lower() in ["history", "social studies"]:
                            objectives = [
                                f"Explain the key events, figures, and developments in {ai_topic}",
                                f"Analyze primary and secondary sources related to {ai_topic}",
                                f"Evaluate the historical significance and modern relevance of {ai_topic}"
                            ]
                            materials = [
                                f"Historical maps, timelines, or artifacts related to {ai_topic}",
                                f"Primary source documents from the {ai_topic} period or event",
                                f"Video clips or multimedia resources about {ai_topic}"
                            ]
                            activities = [
                                f"Begin with an engaging historical question about {ai_topic}",
                                f"Guide students through analysis of sources related to {ai_topic}",
                                f"Have students create timelines or maps illustrating aspects of {ai_topic}",
                                f"Facilitate a structured discussion about different perspectives on {ai_topic}"
                            ]
                            assessment = [
                                f"Ask students to analyze a primary source related to {ai_topic}",
                                f"Have students write a brief argument about the significance of {ai_topic}",
                                f"Evaluate students' ability to connect {ai_topic} to current events or issues"
                            ]
                            topic_intro = f"Begin by posing a thought-provoking question about {ai_topic} that challenges students to consider its significance. You might use a compelling image, quote, or artifact to spark curiosity. Consider starting with a brief anecdote that humanizes the historical figures or events involved in {ai_topic}."
                        else:
                            objectives = [
                                f"Explain the fundamental concepts and principles of {ai_topic}",
                                f"Apply knowledge of {ai_topic} to relevant problems or situations",
                                f"Analyze and evaluate information related to {ai_topic}"
                            ]
                            materials = [
                                f"Textbooks or digital resources covering {ai_topic}",
                                f"Worksheets or handouts with {ai_topic} exercises",
                                f"Visual aids or presentation slides illustrating {ai_topic}"
                            ]
                            activities = [
                                f"Begin with an engaging hook related to {ai_topic}",
                                f"Present key concepts of {ai_topic} with relevant examples",
                                f"Guide students through practice activities related to {ai_topic}",
                                f"Have students apply their understanding of {ai_topic} independently"
                            ]
                            assessment = [
                                f"Ask questions throughout the lesson to check understanding of {ai_topic}",
                                f"Have students complete a brief quiz or exit ticket about {ai_topic}",
                                f"Assign homework that reinforces learning about {ai_topic}"
                            ]
                            topic_intro = f"Begin by connecting {ai_topic} to students' prior knowledge or experiences. Ask what they already know about {ai_topic} or related concepts. You might use a brief demonstration, video clip, or real-world example to show why {ai_topic} is important and relevant to their lives."
                        
                        # Display learning objectives
                        st.write("### Learning Objectives")
                        st.write("By the end of this lesson, students will be able to:")
                        for i, objective in enumerate(objectives, 1):
                            st.write(f"{i}. {objective}")
                        
                        # Display materials needed
                        st.write("### Materials Needed")
                        for material in materials:
                            st.write(f"- {material}")
                        
                        # Display lesson structure
                        st.write("### Lesson Structure")
                        
                        st.write("**Introduction (10 minutes)**")
                        st.write(topic_intro)
                        
                        st.write("**Main Activity (25 minutes)**")
                        for activity in activities:
                            st.write(f"- {activity}")
                        
                        st.write("**Conclusion (10 minutes)**")
                        st.write(f"- Have students summarize the key points about {ai_topic} in their own words")
                        st.write("- Address any questions or misconceptions that arose during the lesson")
                        st.write(f"- Preview how {ai_topic} connects to upcoming content")
                        
                        # Display assessment
                        st.write("### Assessment")
                        for assess in assessment:
                            st.write(f"- {assess}")
                        
                        # Add suggested next steps and resources
                        st.write("### Next Steps and Resources")
                        
                        # Generate relevant resources based on subject and topic
                        if ai_subject.lower() in ["math", "mathematics"]:
                            resources = [
                                {"name": f"Khan Academy: {ai_topic}", "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={ai_topic.replace(' ', '+')}"},
                                {"name": f"Desmos Activities for {ai_topic}", "url": f"https://teacher.desmos.com/search?q={ai_topic.replace(' ', '+')}"},
                                {"name": "NCTM Illuminations", "url": "https://illuminations.nctm.org/"},
                                {"name": f"GeoGebra Materials for {ai_topic}", "url": f"https://www.geogebra.org/search/{ai_topic.replace(' ', '%20')}"}
                            ]
                            online_courses = [
                                {"name": f"Coursera: Mathematics for {ai_grade} Level", "url": f"https://www.coursera.org/search?query={ai_topic}%20{ai_subject}&index=prod_all_launched_products_term_optimization"},
                                {"name": f"edX: {ai_topic} Courses", "url": f"https://www.edx.org/search?q={ai_topic}+{ai_subject}"},
                                {"name": f"Udemy: {ai_topic} in Mathematics", "url": f"https://www.udemy.com/courses/search/?src=ukw&q={ai_topic}+{ai_subject}"}
                            ]
                            books_articles = [
                                {"name": f"OpenStax: Free Mathematics Textbooks", "url": "https://openstax.org/subjects/math"},
                                {"name": f"JSTOR Articles on {ai_topic}", "url": f"https://www.jstor.org/action/doBasicSearch?Query={ai_topic}+{ai_subject}"},
                                {"name": f"arXiv Mathematics Papers on {ai_topic}", "url": f"https://arxiv.org/search/?query={ai_topic}+{ai_subject}&searchtype=all"}
                            ]
                            videos_podcasts = [
                                {"name": f"3Blue1Brown: Visual Mathematics", "url": "https://www.3blue1brown.com/"},
                                {"name": f"Numberphile Videos on {ai_topic}", "url": f"https://www.youtube.com/user/numberphile/search?query={ai_topic}"},
                                {"name": f"Math Ed Podcast", "url": "https://www.podomatic.com/podcasts/mathed"}
                            ]
                            next_topics = [
                                f"Advanced applications of {ai_topic} in real-world contexts",
                                f"Connecting {ai_topic} to related mathematical concepts",
                                f"Using technology tools to explore {ai_topic} more deeply"
                            ]
                        elif ai_subject.lower() in ["science", "biology", "chemistry", "physics"]:
                            resources = [
                                {"name": f"PhET Interactive Simulations: {ai_topic}", "url": f"https://phet.colorado.edu/en/simulations/filter?subjects=biology,chemistry,earth-science,physics&type=html,prototype&sort=alpha&view=grid&search={ai_topic.replace(' ', '+')}"},
                                {"name": f"NASA Education Resources on {ai_topic}", "url": f"https://www.nasa.gov/education/resources/?search={ai_topic.replace(' ', '+')}"},
                                {"name": f"National Geographic: {ai_topic}", "url": f"https://www.nationalgeographic.org/education/search/?q={ai_topic.replace(' ', '+')}"},
                                {"name": f"HHMI BioInteractive: {ai_topic}", "url": f"https://www.biointeractive.org/search?keywords={ai_topic.replace(' ', '+')}&sort_by=search_api_relevance"}
                            ]
                            online_courses = [
                                {"name": f"Coursera: {ai_subject} Courses on {ai_topic}", "url": f"https://www.coursera.org/search?query={ai_topic}%20{ai_subject}&index=prod_all_launched_products_term_optimization"},
                                {"name": f"edX: {ai_topic} in {ai_subject}", "url": f"https://www.edx.org/search?q={ai_topic}+{ai_subject}"},
                                {"name": f"FutureLearn: {ai_subject} Courses", "url": f"https://www.futurelearn.com/search?q={ai_topic}+{ai_subject}"}
                            ]
                            books_articles = [
                                {"name": f"OpenStax: Free {ai_subject} Textbooks", "url": f"https://openstax.org/subjects/{ai_subject.lower()}"},
                                {"name": f"Science Direct Articles on {ai_topic}", "url": f"https://www.sciencedirect.com/search?qs={ai_topic}"},
                                {"name": f"Nature: Research on {ai_topic}", "url": f"https://www.nature.com/search?q={ai_topic}&journal="}
                            ]
                            videos_podcasts = [
                                {"name": f"Crash Course {ai_subject}", "url": f"https://www.youtube.com/c/crashcourse/search?query={ai_topic}"},
                                {"name": f"Science Friday Podcasts on {ai_topic}", "url": f"https://www.sciencefriday.com/search/?s={ai_topic}"},
                                {"name": f"TED-Ed Science Videos", "url": f"https://ed.ted.com/search?qs={ai_topic}"}
                            ]
                            next_topics = [
                                f"Designing more complex investigations of {ai_topic}",
                                f"Exploring current scientific research related to {ai_topic}",
                                f"Examining real-world applications and technologies based on {ai_topic}"
                            ]
                        elif ai_subject.lower() in ["english", "language arts", "literature"]:
                            resources = [
                                {"name": f"ReadWriteThink: {ai_topic} Resources", "url": f"http://www.readwritethink.org/search/?resource_type=6-8&q={ai_topic.replace(' ', '+')}"},
                                {"name": f"CommonLit: {ai_topic} Texts", "url": f"https://www.commonlit.org/en/texts?search_term={ai_topic.replace(' ', '+')}"},
                                {"name": f"Poetry Foundation: {ai_topic}", "url": f"https://www.poetryfoundation.org/search?query={ai_topic.replace(' ', '+')}"},
                                {"name": f"Purdue OWL: Writing about {ai_topic}", "url": "https://owl.purdue.edu/owl/general_writing/index.html"}
                            ]
                            online_courses = [
                                {"name": f"Coursera: {ai_topic} in Literature", "url": f"https://www.coursera.org/search?query={ai_topic}%20literature&index=prod_all_launched_products_term_optimization"},
                                {"name": f"edX: Courses on {ai_topic}", "url": f"https://www.edx.org/search?q={ai_topic}+literature"},
                                {"name": f"Udemy: {ai_topic} Analysis", "url": f"https://www.udemy.com/courses/search/?src=ukw&q={ai_topic}+literature"}
                            ]
                            books_articles = [
                                {"name": f"Project Gutenberg: Free Classic Texts", "url": f"https://www.gutenberg.org/ebooks/search/?query={ai_topic}"},
                                {"name": f"JSTOR Articles on {ai_topic}", "url": f"https://www.jstor.org/action/doBasicSearch?Query={ai_topic}+literature"},
                                {"name": f"Google Scholar: Research on {ai_topic}", "url": f"https://scholar.google.com/scholar?q={ai_topic}+literature"}
                            ]
                            videos_podcasts = [
                                {"name": f"Crash Course Literature", "url": f"https://www.youtube.com/playlist?list=PL8dPuuaLjXtOeEc9ME62zTfqc0h6Pe8vb"},
                                {"name": f"The Literary Life Podcast", "url": "https://www.literarylife.org/podcast"},
                                {"name": f"TED Talks on Literature", "url": f"https://www.ted.com/search?q={ai_topic}+literature"}
                            ]
                            next_topics = [
                                f"Comparative analysis of different works addressing {ai_topic}",
                                f"Creative writing projects inspired by {ai_topic}",
                                f"Multimedia exploration of {ai_topic} through film, art, or music"
                            ]
                        elif ai_subject.lower() in ["history", "social studies"]:
                            resources = [
                                {"name": f"Library of Congress: {ai_topic}", "url": f"https://www.loc.gov/search/?q={ai_topic.replace(' ', '+')}"},
                                {"name": f"National Archives: {ai_topic} Documents", "url": f"https://www.archives.gov/research/search?q={ai_topic.replace(' ', '+')}"},
                                {"name": f"Stanford History Education Group: {ai_topic}", "url": f"https://sheg.stanford.edu/search?search={ai_topic.replace(' ', '+')}"},
                                {"name": f"Facing History: {ai_topic}", "url": f"https://www.facinghistory.org/search?keys={ai_topic.replace(' ', '+')}&type=All"}
                            ]
                            online_courses = [
                                {"name": f"Coursera: {ai_topic} in History", "url": f"https://www.coursera.org/search?query={ai_topic}%20history&index=prod_all_launched_products_term_optimization"},
                                {"name": f"edX: Historical Analysis of {ai_topic}", "url": f"https://www.edx.org/search?q={ai_topic}+history"},
                                {"name": f"FutureLearn: {ai_topic} Courses", "url": f"https://www.futurelearn.com/search?q={ai_topic}+history"}
                            ]
                            books_articles = [
                                {"name": f"JSTOR Articles on {ai_topic}", "url": f"https://www.jstor.org/action/doBasicSearch?Query={ai_topic}+history"},
                                {"name": f"Google Books on {ai_topic}", "url": f"https://www.google.com/search?tbm=bks&q={ai_topic}+history"},
                                {"name": f"Internet History Sourcebooks", "url": "https://sourcebooks.fordham.edu/"}
                            ]
                            videos_podcasts = [
                                {"name": f"Crash Course History", "url": f"https://www.youtube.com/c/crashcourse/search?query={ai_topic}+history"},
                                {"name": f"Dan Carlin's Hardcore History", "url": "https://www.dancarlin.com/hardcore-history-series/"},
                                {"name": f"BBC History Podcasts", "url": "https://www.bbc.co.uk/sounds/category/factual-history"}
                            ]
                            next_topics = [
                                f"Examining different historical perspectives on {ai_topic}",
                                f"Investigating the long-term impacts and legacy of {ai_topic}",
                                f"Connecting {ai_topic} to current events and contemporary issues"
                            ]
                        else:
                            resources = [
                                {"name": f"TED-Ed: {ai_topic}", "url": f"https://ed.ted.com/search?qs={ai_topic.replace(' ', '+')}"},
                                {"name": f"PBS Learning Media: {ai_topic}", "url": f"https://www.pbslearningmedia.org/search/?q={ai_topic.replace(' ', '+')}"},
                                {"name": f"Smithsonian Education: {ai_topic}", "url": f"https://www.si.edu/search/education-resources?edan_q={ai_topic.replace(' ', '+')}"},
                                {"name": f"OER Commons: {ai_topic}", "url": f"https://www.oercommons.org/search?f.search={ai_topic.replace(' ', '+')}&f.general_subject=arts"}
                            ]
                            online_courses = [
                                {"name": f"Coursera: {ai_topic} Courses", "url": f"https://www.coursera.org/search?query={ai_topic}&index=prod_all_launched_products_term_optimization"},
                                {"name": f"edX: Learn about {ai_topic}", "url": f"https://www.edx.org/search?q={ai_topic}"},
                                {"name": f"Udemy: {ai_topic} Classes", "url": f"https://www.udemy.com/courses/search/?src=ukw&q={ai_topic}"}
                            ]
                            books_articles = [
                                {"name": f"Google Scholar: Research on {ai_topic}", "url": f"https://scholar.google.com/scholar?q={ai_topic}"},
                                {"name": f"Open Textbook Library", "url": "https://open.umn.edu/opentextbooks/"},
                                {"name": f"JSTOR Articles on {ai_topic}", "url": f"https://www.jstor.org/action/doBasicSearch?Query={ai_topic}"}
                            ]
                            videos_podcasts = [
                                {"name": f"YouTube Educational Videos on {ai_topic}", "url": f"https://www.youtube.com/results?search_query={ai_topic}+education"},
                                {"name": f"TED Talks on {ai_topic}", "url": f"https://www.ted.com/search?q={ai_topic}"},
                                {"name": f"Educational Podcasts on {ai_topic}", "url": f"https://player.fm/search/{ai_topic}"}
                            ]
                            next_topics = [
                                f"Deeper exploration of advanced concepts in {ai_topic}",
                                f"Interdisciplinary connections between {ai_topic} and other subjects",
                                f"Project-based learning activities centered on {ai_topic}"
                            ]
                        
                        # Display suggested next topics
                        st.write("**Suggested Next Topics:**")
                        for topic in next_topics:
                            st.write(f"- {topic}")
                        
                        # Display comprehensive resource sections
                        st.write("### Comprehensive Learning Resources")
                        
                        # Interactive Teaching Tools
                        st.write("**Interactive Teaching Tools:**")
                        for resource in resources:
                            st.markdown(f"- [{resource['name']}]({resource['url']}) - Interactive resources for teaching {ai_topic}")
                        
                        # Online Courses
                        st.write("**Online Courses:**")
                        for course in online_courses:
                            st.markdown(f"- [{course['name']}]({course['url']}) - Structured learning about {ai_topic}")
                        
                        # Books and Articles
                        st.write("**Books and Academic Articles:**")
                        for book in books_articles:
                            st.markdown(f"- [{book['name']}]({book['url']}) - In-depth reading materials")
                        
                        # Videos and Podcasts
                        st.write("**Videos and Podcasts:**")
                        for media in videos_podcasts:
                            st.markdown(f"- [{media['name']}]({media['url']}) - Multimedia learning resources")
                        
                        # Store the lesson plan data in session state for download
                        st.session_state.lesson_plan_data = f"""# {ai_topic} - {ai_grade} Lesson Plan

Subject: {ai_subject}
Duration: {ai_duration} minutes

## Learning Objectives
By the end of this lesson, students will be able to:
{chr(10).join(f"{i+1}. {obj}" for i, obj in enumerate(objectives))}

## Materials Needed
{chr(10).join(f"- {material}" for material in materials)}

## Lesson Structure

### Introduction (10 minutes)
{topic_intro}

### Main Activity (25 minutes)
{chr(10).join(f"- {activity}" for activity in activities)}

### Conclusion (10 minutes)
- Have students summarize the key points about {ai_topic} in their own words
- Address any questions or misconceptions that arose during the lesson
- Preview how {ai_topic} connects to upcoming content

## Assessment
{chr(10).join(f"- {assess}" for assess in assessment)}

## Next Steps and Resources

### Suggested Next Topics:
{chr(10).join(f"- {topic}" for topic in next_topics)}

### Interactive Teaching Tools:
{chr(10).join(f"- {resource['name']}: {resource['url']}" for resource in resources)}

### Online Courses:
{chr(10).join(f"- {course['name']}: {course['url']}" for course in online_courses)}

### Books and Academic Articles:
{chr(10).join(f"- {book['name']}: {book['url']}" for book in books_articles)}

### Videos and Podcasts:
{chr(10).join(f"- {media['name']}: {media['url']}" for media in videos_podcasts)}
"""
    
    with tools_tab2:
        st.subheader("Assessment Tools")
        
        # Assessment types
        assessment_type = st.selectbox(
            "Select Assessment Type",
            options=["Quizzes", "Rubrics", "Peer Assessment", "Self-Assessment"]
        )
        
        if assessment_type == "Quizzes":
            st.write("### Quiz Generator")
            
            # Initialize quiz_data in session state if not exists
            if 'quiz_data' not in st.session_state:
                st.session_state.quiz_data = None
            
            with st.form("quiz_generator"):
                quiz_subject = st.text_input("Subject")
                quiz_topic = st.text_input("Topic")
                quiz_level = st.select_slider("Difficulty Level", options=["Easy", "Medium", "Hard"])
                quiz_questions = st.number_input("Number of Questions", min_value=5, max_value=30, value=10)
                
                # Question types
                question_types = st.multiselect(
                    "Question Types",
                    options=["Multiple Choice", "True/False", "Short Answer", "Fill in the Blank"],
                    default=["Multiple Choice", "True/False"]
                )
                
                # Generate button
                generate_quiz = st.form_submit_button("Generate Quiz")
                
                if generate_quiz:
                    if not quiz_subject or not quiz_topic:
                        st.error("Please enter both subject and topic.")
                    elif not question_types:
                        st.error("Please select at least one question type.")
                    else:
                        with st.spinner("Generating quiz..."):
                            # Simulate AI processing
                            time.sleep(2)
                            
                            # Create quiz content
                            quiz_content = f"# {quiz_topic} Quiz\n\nSubject: {quiz_subject}\nDifficulty: {quiz_level}\nTotal Questions: {quiz_questions}\n\n"
                            
                            # Add sample questions based on types
                            if "Multiple Choice" in question_types:
                                quiz_content += "**Multiple Choice**\n\n"
                                quiz_content += "1. What is the main concept of this topic?\n"
                                quiz_content += "   a) Option A\n   b) Option B\n   c) Option C\n   d) Option D\n\n"
                            
                            if "True/False" in question_types:
                                quiz_content += "**True/False**\n\n"
                                quiz_content += "2. This statement about the topic is correct.\n"
                                quiz_content += "   True / False\n\n"
                            
                            if "Short Answer" in question_types:
                                quiz_content += "**Short Answer**\n\n"
                                quiz_content += "3. Explain the relationship between concept A and concept B.\n\n"
                            
                            if "Fill in the Blank" in question_types:
                                quiz_content += "**Fill in the Blank**\n\n"
                                quiz_content += "4. The process of ________ is essential to understanding this topic.\n"
                            
                            # Store quiz content in session state
                            st.session_state.quiz_data = {
                                'content': quiz_content,
                                'filename': f"{quiz_subject}_{quiz_topic}_quiz.txt"
                            }
                            
                            st.success("Quiz generated successfully!")
            
            # Display quiz preview and download button outside the form
            if st.session_state.quiz_data:
                st.write("### Quiz Preview")
                st.markdown(st.session_state.quiz_data['content'])
                
                # Download button outside the form
                st.download_button(
                    label="Download Quiz",
                    data=st.session_state.quiz_data['content'],
                    file_name=st.session_state.quiz_data['filename'],
                    mime="text/plain"
                )
        
        elif assessment_type == "Rubrics":
            st.write("### Rubric Creator")
            
            # Get rubric templates from teacher tools
            rubric_templates = teacher_tools.get_rubric_templates()
            
            if rubric_templates:
                st.write("Select a rubric template to customize:")
                
                selected_template = st.selectbox(
                    "Rubric Template",
                    options=[template.get('name') for template in rubric_templates],
                    format_func=lambda x: x
                )
                
                # Find the selected template
                template = next((t for t in rubric_templates if t.get('name') == selected_template), None)
                
                if template:
                    st.write(f"**Description:** {template.get('description', 'No description available.')}")
                    
                    # Display and edit criteria
                    st.write("### Customize Criteria")
                    
                    criteria = template.get('criteria', [])
                    updated_criteria = []
                    
                    for i, criterion in enumerate(criteria):
                        st.write(f"**Criterion {i+1}**")
                        criterion_name = st.text_input(f"Name", value=criterion.get('name', ''), key=f"criterion_name_{i}")
                        criterion_description = st.text_area(f"Description", value=criterion.get('description', ''), key=f"criterion_desc_{i}")
                        
                        # Levels
                        st.write("Performance Levels:")
                        levels = criterion.get('levels', [])
                        updated_levels = []
                        
                        for j, level in enumerate(levels):
                            level_name = st.text_input(f"Level {j+1} Name", value=level.get('name', ''), key=f"level_name_{i}_{j}")
                            level_description = st.text_area(f"Level {j+1} Description", value=level.get('description', ''), key=f"level_desc_{i}_{j}")
                            updated_levels.append({
                                'name': level_name,
                                'description': level_description
                            })
                        
                        updated_criteria.append({
                            'name': criterion_name,
                            'description': criterion_description,
                            'levels': updated_levels
                        })
                    
                    # Save button
                    if st.button("Save Rubric"):
                        st.success("Rubric saved successfully!")
                        
                        # Preview the rubric
                        st.write("### Rubric Preview")
                        
                        # Create a DataFrame for the rubric
                        rubric_data = []
                        for criterion in updated_criteria:
                            for level in criterion['levels']:
                                rubric_data.append({
                                    'Criterion': criterion['name'],
                                    'Level': level['name'],
                                    'Description': level['description']
                                })
                        
                        if rubric_data:
                            st.dataframe(pd.DataFrame(rubric_data))
            else:
                st.info("No rubric templates available.")
        
        elif assessment_type == "Peer Assessment":
            st.write("### Peer Assessment Setup")
            
            with st.form("peer_assessment"):
                assessment_title = st.text_input("Assessment Title")
                assessment_description = st.text_area("Description")
                
                # Peer review questions
                st.write("### Review Questions")
                
                questions = []
                for i in range(5):
                    question = st.text_input(f"Question {i+1}", key=f"peer_q_{i}")
                    if question:
                        questions.append(question)
                
                # Settings
                st.write("### Settings")
                anonymous = st.checkbox("Anonymous Reviews", value=True)
                deadline = st.date_input("Deadline")
                
                # Submit button
                submit_assessment = st.form_submit_button("Create Peer Assessment")
                
                if submit_assessment:
                    if not assessment_title:
                        st.error("Please enter an assessment title.")
                    elif not questions:
                        st.error("Please add at least one review question.")
                    else:
                        st.success(f"Peer assessment '{assessment_title}' created successfully!")
        
        elif assessment_type == "Self-Assessment":
            st.write("### Self-Assessment Creator")
            
            with st.form("self_assessment"):
                assessment_title = st.text_input("Assessment Title")
                assessment_description = st.text_area("Description")
                
                # Self-assessment questions
                st.write("### Assessment Questions")
                
                questions = []
                for i in range(5):
                    question = st.text_input(f"Question {i+1}", key=f"self_q_{i}")
                    if question:
                        questions.append(question)
                
                # Rating scale
                st.write("### Rating Scale")
                scale_type = st.selectbox("Scale Type", ["1-5 Likert Scale", "1-10 Scale", "Custom Scale"])
                
                if scale_type == "Custom Scale":
                    scale_options = st.text_input("Enter scale options separated by commas (e.g., Poor, Fair, Good, Excellent)")
                
                # Submit button
                submit_assessment = st.form_submit_button("Create Self-Assessment")
                
                if submit_assessment:
                    if not assessment_title:
                        st.error("Please enter an assessment title.")
                    elif not questions:
                        st.error("Please add at least one assessment question.")
                    else:
                        st.success(f"Self-assessment '{assessment_title}' created successfully!")
    
    with tools_tab3:
        st.subheader("Resource Library")
        
        # Get resources from teacher tools
        resources = teacher_tools.get_teaching_resources()
        
        if resources:
            # Resource categories
            resource_categories = {}
            for resource in resources:
                category = resource.get('category', 'Other')
                if category not in resource_categories:
                    resource_categories[category] = []
                resource_categories[category].append(resource)
            
            # Display resources by category
            for category, category_resources in resource_categories.items():
                st.write(f"### {category}")
                
                for resource in category_resources:
                    with st.expander(resource.get('title', 'Unknown Resource')):
                        st.write(f"**Description:** {resource.get('description', 'No description available.')}")
                        
                        # Display resource details
                        st.write(f"**Type:** {resource.get('type', 'N/A')}")
                        st.write(f"**Subject:** {resource.get('subject', 'N/A')}")
                        st.write(f"**Grade Level:** {resource.get('grade_level', 'All levels')}")
                        
                        # Display link if available
                        if resource.get('url'):
                            st.markdown(f"[Resource Link]({resource.get('url')})")
                        
                        # Display tags
                        if resource.get('tags'):
                            st.write("**Tags:**")
                            st.write(", ".join(resource.get('tags')))
            
            # Search resources
            st.write("### Search Resources")
            search_query = st.text_input("Search for resources by keyword")
            
            if search_query:
                # Filter resources by search query
                search_results = []
                for resource in resources:
                    # Search in title, description, and tags
                    title = resource.get('title', '').lower()
                    description = resource.get('description', '').lower()
                    tags = ' '.join(resource.get('tags', [])).lower()
                    
                    if search_query.lower() in title or search_query.lower() in description or search_query.lower() in tags:
                        search_results.append(resource)
                
                if search_results:
                    st.write(f"Found {len(search_results)} resources matching '{search_query}'")
                    for resource in search_results:
                        st.markdown(f"**[{resource.get('title')}]({resource.get('url')})**")
                        st.write(resource.get('description', 'No description available.'))
                        st.markdown("---")
                else:
                    st.info("No resources found matching your search query.")
        else:
            st.info("No teaching resources available.")
        
        # Upload resource
        st.write("### Share Your Resources")
        
        with st.form("resource_upload"):
            resource_title = st.text_input("Resource Title")
            resource_description = st.text_area("Description")
            resource_type = st.selectbox("Resource Type", ["Lesson Plan", "Worksheet", "Presentation", "Activity", "Assessment", "Other"])
            resource_subject = st.text_input("Subject")
            resource_grade = st.multiselect("Grade Level", ["Elementary", "Middle School", "High School", "College"])
            resource_url = st.text_input("Resource URL (optional)")
            resource_file = st.file_uploader("Upload File (optional)")
            
            # Tags
            resource_tags = st.text_input("Tags (comma-separated)")
            
            # Submit button
            submit_resource = st.form_submit_button("Share Resource")
            
            if submit_resource:
                if not resource_title or not resource_description or not resource_subject:
                    st.error("Please fill in all required fields.")
                else:
                    st.success(f"Resource '{resource_title}' shared successfully!")
    
    with tools_tab4:
        st.subheader("Professional Development")
        
        # Get professional development resources from teacher tools
        pd_resources = teacher_tools.get_professional_development()
        
        if pd_resources:
            # PD categories
            pd_categories = {}
            for resource in pd_resources:
                category = resource.get('category', 'Other')
                if category not in pd_categories:
                    pd_categories[category] = []
                pd_categories[category].append(resource)
            
            # Display PD resources by category
            for category, category_resources in pd_categories.items():
                st.write(f"### {category}")
                
                for resource in category_resources:
                    with st.expander(resource.get('title', 'Unknown Resource')):
                        st.write(f"**Description:** {resource.get('description', 'No description available.')}")
                        
                        # Display resource details
                        st.write(f"**Type:** {resource.get('type', 'N/A')}")
                        st.write(f"**Duration:** {resource.get('duration', 'N/A')}")
                        
                        # Display link if available
                        if resource.get('url'):
                            st.markdown(f"[Resource Link]({resource.get('url')})")
                        
                        # Display provider
                        if resource.get('provider'):
                            st.write(f"**Provider:** {resource.get('provider')}")
                        
                        # Display certification
                        if resource.get('certification'):
                            st.write(f"**Certification:** {resource.get('certification')}")
        else:
            st.info("No professional development resources available.")
        
        # Professional Learning Communities
        st.write("### Professional Learning Communities")
        
        plc_options = [
            "Technology Integration", 
            "Project-Based Learning", 
            "Differentiated Instruction",
            "Assessment Strategies",
            "Classroom Management"
        ]
        
        selected_plc = st.selectbox("Join a Professional Learning Community", plc_options)
        
        if selected_plc:
            st.write(f"**{selected_plc} Community**")
            st.write("Connect with other educators interested in this topic.")
            
            # Simulated community features
            st.write("### Recent Discussions")
            st.write("- Best practices for implementing this approach")
            st.write("- Resources and tools that have worked well")
            st.write("- Challenges and solutions from experienced educators")
            
            # Join button
            if st.button("Join This Community"):
                st.success(f"You have joined the {selected_plc} community!")
                
                # Show community features
                st.write("### Community Features")
                st.write("- Monthly virtual meetings")
                st.write("- Resource sharing platform")
                st.write("- Mentorship opportunities")
                st.write("- Collaborative projects")

def show_student_dashboard():
    # Get student's courses and data
    student_id = st.session_state.current_user['id']
    student_name = st.session_state.current_user['name']
    
    st.title(f"Welcome, {student_name}!")
    
    # Get student's courses
    courses = get_student_courses(student_id)
    
    # Show different tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Courses", "Assignments", "Performance Analytics", "Career Planning"])
    
    # Overview tab
    with tab1:
        st.header("Dashboard Overview")
        
        # Show quick stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Courses", len(courses))
        
        # Get submissions data
        submissions = get_student_submissions(student_id)
        
        with col2:
            st.metric("Assignments", len(submissions))
        
        with col3:
            # Calculate completion rate
            completed = sum(1 for s in submissions if s.get('score') is not None)
            completion_rate = (completed / max(1, len(submissions))) * 100
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Recent activity
        st.subheader("Recent Activity")
        
        if not submissions:
            st.info("No submissions yet. Enroll in courses and start submitting assignments!")
        else:
            # Sort submissions by date (newest first)
            submissions.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
            
            # Show last 5 submissions
            for submission in submissions[:5]:
                assignment = get_assignment_by_id(submission.get('assignment_id'))
                if assignment:
                    course = get_course_by_id(assignment.get('course_id'))
                    course_name = course.get('name', 'Unknown Course') if course else 'Unknown Course'
                    
                    # Format submission date
                    submission_date = submission.get('submitted_at', '')
                    if submission_date:
                        try:
                            submission_date = datetime.fromisoformat(submission_date).strftime("%B %d, %Y")
                        except:
                            pass
                    
                    # Check if graded
                    if submission.get('score') is not None:
                        st.success(f"{assignment.get('title')} - {course_name} - Scored {submission.get('score')}/{assignment.get('points', 100)} - {submission_date}")
                else:
                        st.info(f"{assignment.get('title')} - {course_name} - Submitted (awaiting grade) - {submission_date}")
        
        # Quick links
        st.subheader("Quick Links")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Study Recommendations", use_container_width=True):
                st.session_state.current_page = 'study_recommendations'
                st.rerun()
        
        with col2:
            if st.button("Learning Path", use_container_width=True):
                st.session_state.current_page = 'learning_path'
                st.rerun()
        
        with col3:
            if st.button("View My Groups", use_container_width=True):
                st.session_state.current_page = 'group_management'
                st.rerun()
    
    # Courses tab
    with tab2:
        st.header("My Courses")
        if not courses:
            st.info("You haven't enrolled in any courses yet.")
        else:
            for course in courses:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{course['name']} ({course['code']})")
                    st.write(course['description'])
                    st.write(f"**Period:** {course['start_date']} to {course['end_date']}")
                    
                    # Get enrollment count
                    enrolled_students = [s for s in course.get('students_enrolled', []) 
                                        if isinstance(s, dict) and s.get('status') == 'active']
                    st.write(f"**Students enrolled:** {len(enrolled_students)}")
                    
                    # Get assignment count
                    assignments = get_course_assignments(course['id'])
                    st.write(f"**Assignments:** {len(assignments)}")
                
                with col2:
                    if st.button("View Course", key=f"view_course_{course['id']}"):
                        st.session_state.current_course = course
                        st.session_state.current_page = 'course_detail'
                        st.rerun()
                    
                    if st.button("Add Assignment", key=f"add_assignment_{course['id']}"):
                        st.session_state.current_course = course
                        st.session_state.current_page = 'create_assignment'
                        st.rerun()
                
                st.divider()
    
    # Assignments tab
    with tab3:
        st.header("My Assignments")
        if not submissions:
            st.info("You haven't submitted any assignments yet.")
        else:
            # Sort submissions by date (newest first)
            submissions.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
            
            # Show last 5 submissions
            for submission in submissions[:5]:
                assignment = get_assignment_by_id(submission.get('assignment_id'))
                if assignment:
                    course = get_course_by_id(assignment.get('course_id'))
                    course_name = course.get('name', 'Unknown Course') if course else 'Unknown Course'
                    
                    # Format submission date
                    submission_date = submission.get('submitted_at', '')
                    if submission_date:
                        try:
                            submission_date = datetime.fromisoformat(submission_date).strftime("%B %d, %Y")
                        except:
                            pass
                    
                    # Check if graded
                    if submission.get('score') is not None:
                        st.success(f"{assignment.get('title')} - {course_name} - Scored {submission.get('score')}/{assignment.get('points', 100)} - {submission_date}")
                    else:
                        st.info(f"{assignment.get('title')} - {course_name} - Submitted (awaiting grade) - {submission_date}")
    
    # Performance Analytics tab
    with tab4:
        st.header("Performance Analytics")
        
        # Get student's overall performance
        overall_performance = get_student_overall_performance(student_id)
        
        if overall_performance:
            st.subheader("Overall Performance")
            st.write(f"**Average Score:** {overall_performance['average_score']:.1f}%")
            st.write(f"**Completion Rate:** {overall_performance['completion_rate']:.1f}%")
            st.write(f"**Submissions:** {overall_performance['submissions_count']}")
            
            # Display skill graph
            st.subheader("Skill Graph")
            skill_data = overall_performance['skill_data']
            if skill_data:
                # Convert skill_data dictionary to list
                skills_list = [{'name': k, **v} for k, v in skill_data.items()]
                
                # Generate skill graph
                skill_graph = generate_skill_graph(skills_list)
                
                # Display skill graph
                st.components.v1.html(skill_graph, height=300)
            else:
                st.info("No skill data available.")
            
            # Display skill proficiency summary
            st.subheader("Skill Proficiency Summary")
            skill_summary = get_skill_summary(skills_list)
            if skill_summary:
                # Create DataFrame for skill summary
                skill_summary_df = pd.DataFrame(skill_summary)
                st.dataframe(skill_summary_df)
            else:
                st.info("No skill data available.")
            
            # Display skill gap analysis
            st.subheader("Skill Gap Analysis")
            skill_gaps = get_skill_gaps(skills_list)
            if skill_gaps:
                # Create DataFrame for skill gaps
                skill_gaps_df = pd.DataFrame(skill_gaps)
                st.dataframe(skill_gaps_df)
            else:
                st.info("No skill gaps found.")
            
            # Display skill development over time
            st.subheader("Skill Development Over Time")
            skill_development_data = get_skill_development_data(student_id)
            if skill_development_data:
                # Create DataFrame for skill development
                skill_development_df = pd.DataFrame(skill_development_data)
                st.dataframe(skill_development_df)
            else:
                st.info("No skill development data available.")
            
            # Display most improved and struggling areas
            st.subheader("Most Improved and Struggling Areas")
            trends = get_trends(student_id)
            if trends:
                # Create DataFrame for trends
                trends_df = pd.DataFrame(trends)
                st.dataframe(trends_df)
            else:
                st.info("No trends data available.")
    
    # Career Planning tab
    with tab5:
        st.header("Career Planning")
        
        # Get career data
        career_data = get_career_data()
        
        if career_data:
            st.subheader("Career Path")
            st.write(f"**Current Skill Level:** {career_data['current_skill_level']}")
            st.write(f"**Target Skill Level:** {career_data['target_skill_level']}")
            st.write(f"**Years to Reach Target:** {career_data['years_to_reach_target']}")
            
            # Display skill matrix
            st.subheader("Skill Matrix")
            skill_matrix = career_data['skill_matrix']
            if skill_matrix:
                # Create DataFrame for skill matrix
                skill_matrix_df = pd.DataFrame(skill_matrix)
                st.dataframe(skill_matrix_df)
                    else:
                st.info("No skill matrix data available.")
            
            # Display recommended courses
            st.subheader("Recommended Courses")
            recommended_courses = career_data['recommended_courses']
            if recommended_courses:
                # Create DataFrame for recommended courses
                recommended_courses_df = pd.DataFrame(recommended_courses)
                st.dataframe(recommended_courses_df)
        else:
                st.info("No recommended courses available.")
            
            # Display recommended resources
            st.subheader("Recommended Resources")
            resources = career_data['recommended_resources']
            if resources:
                # Group resources by type
                resource_types = {}
                for resource in resources:
                    res_type = resource.get('type', 'Other')
                    if res_type not in resource_types:
                        resource_types[res_type] = []
                    resource_types[res_type].append(resource)
                
                # Display resources by type
                for res_type, type_resources in resource_types.items():
                    with st.expander(f"{res_type} ({len(type_resources)})"):
                        for resource in type_resources:
                            st.write(f"**{resource.get('title')}**")
                            st.write(resource.get('description', ''))
                            
                            # Display URL if available
                            if 'url' in resource:
                                st.markdown(f"[Access Resource]({resource['url']})")
                            
                            # Display skill relevance if available
                            if 'relevant_skills' in resource:
                                st.caption(f"Relevant skills: {', '.join(resource['relevant_skills'])}")
                            
                            st.divider()
            else:
                st.info("No recommended resources available.")
        else:
            st.info("No career data available.")

def show_language_settings():
    """Display language settings for current user."""
    st.title("Language Settings")
    
    # Get current user
    user_id = st.session_state.current_user['id']
    user_role = st.session_state.current_user['role']
    
    # Get supported languages from service
    supported_languages = multilingual_feedback_service.get_supported_languages()
    
    # Convert to options for selectbox
    language_options = [{
        'code': lang['code'],
        'name': lang['name']
    } for lang in supported_languages]
    
    # Get current language preference
    current_language = get_student_language_preference(user_id)
    
    # Find current language index
    current_index = 0
    for i, lang in enumerate(language_options):
        if lang['code'] == current_language:
            current_index = i
            break
    
    st.subheader("Select Your Preferred Language")
    st.write("This will be used for translating feedback and system messages.")
    
    # Language selector
    selected_language = st.selectbox(
        "Preferred Language",
        options=language_options,
        format_func=lambda x: x['name'],
        index=current_index
    )
    
    # Save button
    if st.button("Save Language Preference"):
        if set_student_language_preference(user_id, selected_language['code']):
            st.success(f"Language preference updated to {selected_language['name']}.")
        else:
            st.error("Failed to update language preference. Please try again.")
    
    # Preview section
    st.subheader("Translation Preview")
    
    # Original text
    original_text = st.text_area(
        "Enter text to preview translation",
        value="Great work on this assignment! Your analysis was thorough and well-structured. Next time, try to include more specific examples to support your arguments."
    )
    
    if original_text:
        # Translate to selected language
        translated_text = translate_feedback(original_text, selected_language['code'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Text (English)**")
            st.write(original_text)
        
        with col2:
            st.write(f"**Translated Text ({selected_language['name']})**")
            st.write(translated_text)
    
    # Language information
    with st.expander("About Language Support"):
        st.write("""
        EduMate provides automatic translation of feedback and important system messages in multiple languages. 
        This helps ensure that all students can access and understand feedback in their preferred language.
        
        **How it works:**
        
        1. Select your preferred language from the dropdown menu above
        2. Save your preference using the button
        3. All feedback from teachers and AI will be automatically translated to your preferred language
        
        You can change your language preference at any time by returning to this page.
        """)
        
        st.caption("Note: All your feedback will be automatically translated to your preferred language.")

def show_teacher_analytics():
    """Display the enhanced Teacher Analytics Dashboard using the TeacherAnalyticsService."""
    st.title("Teacher Analytics Dashboard")
    
    # Get teacher's courses
    teacher_id = st.session_state.current_user['id']
    courses = get_teacher_courses(teacher_id)
    
    if not courses:
        st.info("You haven't created any courses yet. Create a course to access analytics.")
        return
    
    # Select course to analyze
    selected_course = st.selectbox(
        "Select Course",
        options=courses,
        format_func=lambda x: f"{x['name']} ({x['code']})"
    )
    
    if not selected_course:
        st.warning("Please select a course to analyze.")
        return
    
    # Create dashboard options with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Class Dashboard", 
        "Student Comparison", 
        "Longitudinal Analysis",
        "AI Insights"
    ])
    
    # Get assignments and submissions for the selected course
    assignments = get_course_assignments(selected_course['id'])
    
    # Get all submissions for all assignments in this course
    all_submissions = []
    for assignment in assignments:
        assignment_submissions = get_assignment_submissions(assignment['id'])
        for submission in assignment_submissions:
            # Add assignment info to submission for easier access
            submission['assignment_info'] = assignment
            all_submissions.append(submission)
    
    # Get student data
    students = []
    for student_id in selected_course.get('students', []):
        student = get_user_by_id(student_id)
        if student:
            students.append(student)
    
    with tab1:
        st.header("Class Dashboard")
        
        # Use TeacherAnalyticsService to generate class dashboard
        dashboard_data = teacher_analytics_service.generate_class_dashboard(
            course_id=selected_course['id'],
            teacher_id=teacher_id,
            assignments=assignments,
            submissions=all_submissions,
            students=students
        )
        
        if dashboard_data:
            # Display overall class metrics
            st.subheader("Class Performance Overview")
            
            metrics = dashboard_data.get('class_metrics', {})
            col1, col2, col3, col4 = st.columns(4)
            
                with col1:
                avg_score = metrics.get('average_score', 0)
                st.metric("Average Score", f"{avg_score:.1f}%")
            
                with col2:
                completion_rate = metrics.get('completion_rate', 0) * 100
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            
            with col3:
                at_risk_count = metrics.get('at_risk_students', 0)
                st.metric("At-Risk Students", at_risk_count)
            
            with col4:
                on_track_count = metrics.get('on_track_students', 0)
                st.metric("On-Track Students", on_track_count)
            
            # Display grade distribution
            st.subheader("Grade Distribution")
            
            grade_distribution = dashboard_data.get('grade_distribution', [])
            if grade_distribution:
                # Create a DataFrame for the histogram
                grades = [entry.get('score', 0) for entry in grade_distribution]
                
                if grades:
                    # Plot histogram
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.hist(grades, bins=10, range=(0, 100))
                    ax.set_xlabel('Score (%)')
                    ax.set_ylabel('Number of Students')
                    ax.set_title('Grade Distribution')
                    
                    # Add vertical lines for grade boundaries
                    ax.axvline(x=90, color='green', linestyle='--', label='A (90%)')
                    ax.axvline(x=80, color='lightgreen', linestyle='--', label='B (80%)')
                    ax.axvline(x=70, color='yellow', linestyle='--', label='C (70%)')
                    ax.axvline(x=60, color='orange', linestyle='--', label='D (60%)')
                    ax.axvline(x=50, color='red', linestyle='--', label='F (50%)')
                    ax.legend()
                    
                    st.pyplot(fig)
            
            # Display assignment completion rates
            st.subheader("Assignment Completion Rates")
            
            assignment_data = dashboard_data.get('assignment_metrics', [])
            if assignment_data:
                # Create DataFrame
                assignment_df = pd.DataFrame(assignment_data)
                if not assignment_df.empty and 'title' in assignment_df.columns and 'completion_rate' in assignment_df.columns:
                    assignment_df['completion_rate'] = assignment_df['completion_rate'] * 100  # Convert to percentage
                    
                    # Sort by due date if available
                    if 'due_date' in assignment_df.columns:
                        assignment_df = assignment_df.sort_values('due_date')
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(assignment_df['title'], assignment_df['completion_rate'], color='skyblue')
                    ax.set_xlabel('Assignment')
                    ax.set_ylabel('Completion Rate (%)')
                    ax.set_ylim(0, 100)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
            
            # Display skill gap analysis
            st.subheader("Skill Gap Analysis")
            
            skill_data = dashboard_data.get('skill_gaps', [])
            if skill_data:
                # Create DataFrame
                skill_df = pd.DataFrame(skill_data)
                if not skill_df.empty and 'skill' in skill_df.columns and 'proficiency' in skill_df.columns:
                    skill_df['proficiency'] = skill_df['proficiency'] * 100  # Convert to percentage
                    
                    # Sort by proficiency (ascending to show gaps first)
                    skill_df = skill_df.sort_values('proficiency')
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.barh(skill_df['skill'], skill_df['proficiency'], color='skyblue')
                    
                    # Color bars based on proficiency
                    for i, bar in enumerate(bars):
                        proficiency = skill_df['proficiency'].iloc[i]
                        if proficiency < 60:
                            bar.set_color('salmon')
                        elif proficiency < 80:
                            bar.set_color('khaki')
                        else:
                            bar.set_color('lightgreen')
                    
                    ax.set_xlabel('Proficiency (%)')
                    ax.set_xlim(0, 100)
                    
                    # Add vertical lines for proficiency levels
                    ax.axvline(x=80, color='green', linestyle='--', label='Mastery (80%)')
                    ax.axvline(x=60, color='orange', linestyle='--', label='Proficient (60%)')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Display suggestions for addressing skill gaps
                    if 'gap_suggestions' in dashboard_data:
                        st.subheader("Suggested Interventions")
                        for suggestion in dashboard_data['gap_suggestions']:
                            st.write(f"- {suggestion}")
        else:
            st.warning("Unable to generate class dashboard. Ensure students are enrolled and have submitted assignments.")
    
    with tab2:
        st.header("Student Comparison")
        
        if not students:
            st.info("No students enrolled in this course yet.")
                else:
            # Allow comparing specific students
            selected_students = st.multiselect(
                "Select Students to Compare",
                options=[s['id'] for s in students],
                format_func=lambda x: next((s['name'] for s in students if s['id'] == x), str(x)),
                default=[s['id'] for s in students[:min(5, len(students))]]  # Default to first 5 students
            )
            
            if selected_students:
                # Use TeacherAnalyticsService to compare students
                comparison_data = teacher_analytics_service.compare_students(
                    course_id=selected_course['id'],
                    student_ids=selected_students,
                    assignments=assignments,
                    submissions=all_submissions
                )
                
                if comparison_data:
                    # Display overall performance comparison
                    st.subheader("Performance Comparison")
                    
                    student_metrics = comparison_data.get('student_metrics', [])
                    if student_metrics:
                        # Create DataFrame for comparison
                        metrics_df = pd.DataFrame(student_metrics)
                        
                        # Replace student IDs with names
                        metrics_df['student_name'] = metrics_df['student_id'].apply(
                            lambda x: next((s['name'] for s in students if s['id'] == x), str(x))
                        )
                        
                        # Transform to percentage where needed
                        if 'average_score' in metrics_df.columns:
                            metrics_df['average_score'] = metrics_df['average_score'].apply(lambda x: x * 100 if x <= 1 else x)
                        
                        if 'completion_rate' in metrics_df.columns:
                            metrics_df['completion_rate'] = metrics_df['completion_rate'].apply(lambda x: x * 100 if x <= 1 else x)
                        
                        # Create comparison chart
                        st.subheader("Average Scores")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.bar(metrics_df['student_name'], metrics_df['average_score'], color='skyblue')
                        ax.set_ylabel('Average Score (%)')
                        ax.set_ylim(0, 100)
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                        
                        # Display a data table with comparison metrics
                        display_cols = ['student_name', 'average_score', 'completion_rate', 'submissions_count']
                        renamed_cols = {
                            'student_name': 'Student',
                            'average_score': 'Average Score (%)',
                            'completion_rate': 'Completion Rate (%)',
                            'submissions_count': 'Submissions Count'
                        }
                        
                        # Filter and rename columns
                        display_df = metrics_df[display_cols].rename(columns=renamed_cols)
                        st.dataframe(display_df)
                    
                    # Display skill comparison
                    st.subheader("Skill Comparison")
                    
                    skills_data = comparison_data.get('skill_comparison', [])
                    if skills_data:
                        # Prepare data for radar chart
                        student_skills = {}
                        all_skills = set()
                        
                        for entry in skills_data:
                            student_id = entry.get('student_id')
                            student_name = next((s['name'] for s in students if s['id'] == student_id), str(student_id))
                            
                            if student_name not in student_skills:
                                student_skills[student_name] = {}
                            
                            # Add skills and proficiency
                            skills = entry.get('skills', [])
                            for skill in skills:
                                skill_name = skill.get('name')
                                if skill_name:
                                    all_skills.add(skill_name)
                                    # Convert to percentage if needed
                                    proficiency = skill.get('proficiency', 0)
                                    if proficiency <= 1:
                                        proficiency *= 100
                                    student_skills[student_name][skill_name] = proficiency
                        
                        if student_skills and all_skills:
                            # Create radar chart for skill comparison
                            all_skills = sorted(list(all_skills))
                            
                            # Prepare data for radar chart
                            fig = plt.figure(figsize=(10, 10))
                            ax = fig.add_subplot(111, polar=True)
                            
                            # Set angles for each skill
                            angles = np.linspace(0, 2*np.pi, len(all_skills), endpoint=False).tolist()
                            angles += angles[:1]  # Close the loop
                            
                            # Plot each student
                            for student_name, skills in student_skills.items():
                                values = [skills.get(skill, 0) for skill in all_skills]
                                values += values[:1]  # Close the loop
                                
                                ax.plot(angles, values, linewidth=2, label=student_name)
                                ax.fill(angles, values, alpha=0.1)
                            
                            # Set chart properties
                            ax.set_xticks(angles[:-1])
                            ax.set_xticklabels(all_skills)
                            ax.set_yticks([0, 25, 50, 75, 100])
                            ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
                            ax.set_ylim(0, 100)
                            
                            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Display a data table with skill proficiency
                            st.subheader("Skill Proficiency Comparison")
                            
                            # Create a DataFrame with skill proficiency for each student
                            skill_df_data = []
                            for skill_name in all_skills:
                                row_data = {'Skill': skill_name}
                                for student_name, skills in student_skills.items():
                                    row_data[student_name] = f"{skills.get(skill_name, 0):.1f}%"
                                skill_df_data.append(row_data)
                            
                            if skill_df_data:
                                skill_df = pd.DataFrame(skill_df_data)
                                st.dataframe(skill_df)
                                    else:
                    st.warning("Unable to generate student comparison. Please ensure selected students have submitted assignments.")
                                                else:
                st.info("Please select at least one student to compare.")
    
    with tab3:
        st.header("Longitudinal Analysis")
        
        # Use TeacherAnalyticsService to generate longitudinal analysis
        longitudinal_data = teacher_analytics_service.generate_longitudinal_analysis(
            course_id=selected_course['id'],
            assignments=assignments,
            submissions=all_submissions,
            time_periods=5  # Divide the course into 5 time periods
        )
        
        if longitudinal_data:
            # Display overall progression
            st.subheader("Class Performance Over Time")
            
            time_series = longitudinal_data.get('time_series', [])
            if time_series:
                # Create DataFrame for time series
                time_df = pd.DataFrame(time_series)
                
                # Ensure period and average_score columns exist
                if 'period' in time_df.columns and 'average_score' in time_df.columns:
                    # Convert to percentage if needed
                    time_df['average_score'] = time_df['average_score'].apply(lambda x: x * 100 if x <= 1 else x)
                    
                    # Add completion rate as a percentage
                    if 'completion_rate' in time_df.columns:
                        time_df['completion_rate'] = time_df['completion_rate'].apply(lambda x: x * 100 if x <= 1 else x)
                    
                    # Create line chart
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    
                    # Plot average score
                    color = 'tab:blue'
                    ax1.set_xlabel('Time Period')
                    ax1.set_ylabel('Average Score (%)', color=color)
                    ax1.plot(time_df['period'], time_df['average_score'], marker='o', color=color)
                    ax1.tick_params(axis='y', labelcolor=color)
                    ax1.set_ylim(0, 100)
                    
                    # Add completion rate on secondary y-axis if available
                    if 'completion_rate' in time_df.columns:
                        ax2 = ax1.twinx()
                        color = 'tab:red'
                        ax2.set_ylabel('Completion Rate (%)', color=color)
                        ax2.plot(time_df['period'], time_df['completion_rate'], marker='s', color=color, linestyle='--')
                        ax2.tick_params(axis='y', labelcolor=color)
                        ax2.set_ylim(0, 100)
                    
                    fig.tight_layout()
                    st.pyplot(fig)
                    
                    # Display the data in a table
                    st.dataframe(time_df.rename(columns={
                        'period': 'Time Period',
                        'average_score': 'Average Score (%)',
                        'completion_rate': 'Completion Rate (%)',
                        'submissions_count': 'Submissions Count'
                    }))
            
            # Display skill development over time
            st.subheader("Skill Development Over Time")
            
            skill_development = longitudinal_data.get('skill_development', [])
            if skill_development:
                # Get unique skills
                all_skills = set()
                for period_data in skill_development:
                    skills = period_data.get('skills', [])
                    for skill in skills:
                        all_skills.add(skill.get('name'))
                
                all_skills = sorted(list(all_skills))
                if all_skills:
                    # Create DataFrames for each skill's development
                    skill_dfs = {}
                    for skill_name in all_skills:
                        skill_data = []
                        for period_data in skill_development:
                            period = period_data.get('period')
                            skills = period_data.get('skills', [])
                            
                            # Find the specific skill
                            skill_entry = next((s for s in skills if s.get('name') == skill_name), None)
                            
                            if skill_entry:
                                # Convert to percentage if needed
                                proficiency = skill_entry.get('proficiency', 0)
                                if proficiency <= 1:
                                    proficiency *= 100
                                
                                skill_data.append({
                                    'period': period,
                                    'proficiency': proficiency
                                })
                        
                        if skill_data:
                            skill_dfs[skill_name] = pd.DataFrame(skill_data)
                    
                    # Create multi-line chart for skill development
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    for skill_name, df in skill_dfs.items():
                        if 'period' in df.columns and 'proficiency' in df.columns and not df.empty:
                            ax.plot(df['period'], df['proficiency'], marker='o', label=skill_name)
                    
                    ax.set_xlabel('Time Period')
                    ax.set_ylabel('Proficiency (%)')
                    ax.set_ylim(0, 100)
                    ax.legend(title='Skills')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Display most improved and struggling areas
            if 'trends' in longitudinal_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Most Improved Areas")
                    improved_areas = longitudinal_data['trends'].get('improved_areas', [])
                    for area in improved_areas:
                        st.success(f"**{area.get('name', '')}:** {area.get('improvement', 0):.1f}% improvement")
                
                with col2:
                    st.subheader("Struggling Areas")
                    struggling_areas = longitudinal_data['trends'].get('struggling_areas', [])
                    for area in struggling_areas:
                        st.error(f"**{area.get('name', '')}:** {area.get('decline', 0):.1f}% decline")
        else:
            st.warning("Unable to generate longitudinal analysis. Please ensure sufficient data is available over time.")
    
    with tab4:
        st.header("AI Insights")
        
        # Use TeacherAnalyticsService to generate AI insights
        ai_insights = teacher_analytics_service.generate_ai_insights(
            course_id=selected_course['id'],
            teacher_id=teacher_id,
            assignments=assignments,
            submissions=all_submissions,
            students=students
        )
        
        if ai_insights:
            # Display at-risk students
            st.subheader("At-Risk Students")
            
            at_risk_students = ai_insights.get('at_risk_students', [])
            if at_risk_students:
                for student_data in at_risk_students:
                    student_id = student_data.get('student_id')
                    student = next((s for s in students if s['id'] == student_id), {})
                    student_name = student.get('name', f"Student {student_id}")
                    
                    with st.expander(f"{student_name} - Risk Score: {student_data.get('risk_score', 0):.1f}/10"):
                        st.write("**Risk Factors:**")
                        for factor in student_data.get('risk_factors', []):
                            st.warning(f"- {factor}")
                        
                        st.write("**Recommended Interventions:**")
                        for intervention in student_data.get('interventions', []):
                            st.info(f"- {intervention}")
            else:
                st.success("No students currently at risk based on the available data.")
            
            # Display content effectiveness
            st.subheader("Content Effectiveness")
            
            content_analysis = ai_insights.get('content_effectiveness', {})
            if content_analysis:
                # Plot content effectiveness
                effectiveness_data = content_analysis.get('assignments', [])
                if effectiveness_data:
                    # Create DataFrame
                    eff_df = pd.DataFrame(effectiveness_data)
                    
                    if 'title' in eff_df.columns and 'effectiveness' in eff_df.columns:
                        # Convert to percentage if needed
                        eff_df['effectiveness'] = eff_df['effectiveness'].apply(lambda x: x * 100 if x <= 1 else x)
                        
                        # Sort by effectiveness
                        eff_df = eff_df.sort_values('effectiveness', ascending=False)
                        
                        # Create horizontal bar chart
                        fig, ax = plt.subplots(figsize=(10, max(5, len(eff_df) * 0.4)))
                        
                        # Create colorful bars based on effectiveness
                        bars = ax.barh(eff_df['title'], eff_df['effectiveness'])
                        for i, bar in enumerate(bars):
                            effectiveness = eff_df['effectiveness'].iloc[i]
                            if effectiveness < 60:
                                bar.set_color('salmon')
                            elif effectiveness < 80:
                                bar.set_color('khaki')
                            else:
                                bar.set_color('lightgreen')
                        
                        ax.set_xlabel('Effectiveness (%)')
                        ax.set_xlim(0, 100)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Display content improvement suggestions
                if 'improvement_suggestions' in content_analysis:
                    st.subheader("Content Improvement Suggestions")
                    for suggestion in content_analysis['improvement_suggestions']:
                        st.write(f"- {suggestion}")
            
            # Display teaching recommendations
            if 'teaching_recommendations' in ai_insights:
                st.subheader("Teaching Strategy Recommendations")
                
                recommendations = ai_insights['teaching_recommendations']
                for category, items in recommendations.items():
                    st.write(f"**{category.replace('_', ' ').title()}:**")
                    for item in items:
                        st.write(f"- {item}")
        else:
            st.warning("Unable to generate AI insights. Please ensure sufficient data is available for analysis.")

def main():
    # Check if user is logged in
    if not st.session_state.logged_in:
        if st.session_state.current_page == 'login':
            show_login_page()
        elif st.session_state.current_page == 'register':
            show_register_page()
    else:
            show_login_page()
    else:
        # Set up sidebar navigation
        with st.sidebar:
            st.image("https://i.imgur.com/ZVU5qN6.png", width=100)  # Replace with your logo
            st.title("EduMate")
            
            # User info
            st.write(f"Logged in as: **{st.session_state.current_user['name']}**")
            st.write(f"Role: **{st.session_state.current_user['role'].capitalize()}**")
            
            st.divider()
            
            # Navigation options
            if st.button("Dashboard", use_container_width=True):
                st.session_state.current_page = 'dashboard'
                            st.rerun()
            
            # Role-specific navigation
            if st.session_state.current_user['role'] == 'teacher':
                if st.button("Teacher Analytics", use_container_width=True):
                    st.session_state.current_page = 'teacher_analytics'
                    st.rerun()
                
                if st.button("Group Management", use_container_width=True):
                    st.session_state.current_page = 'group_management'
                            st.rerun()
            else:  # student
                if st.button("Study Recommendations", use_container_width=True):
                    st.session_state.current_page = 'study_recommendations'
                    st.rerun()
                
                if st.button("My Groups", use_container_width=True):
                    st.session_state.current_page = 'student_groups'
                    st.rerun()
                
                if st.button("Learning Path", use_container_width=True):
                    st.session_state.current_page = 'learning_path'
                        st.rerun()
                
            st.divider()
            
            # Language settings
            if st.button("Language Settings", use_container_width=True):
                st.session_state.current_page = 'language_settings'
                st.rerun()
            
            # Career planning
            if st.button("Career Planning", use_container_width=True):
                st.session_state.current_page = 'career_planning'
                st.rerun()
            
            # Settings & Help
            if st.button("System Settings", use_container_width=True):
                st.session_state.current_page = 'settings'
                st.rerun()
            
            if st.button("Help & Support", use_container_width=True):
                st.session_state.current_page = 'help'
                st.rerun()
            
            # Logout button at the bottom
            st.divider()
            if st.button("Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.current_user = None
                st.session_state.current_page = 'login'
                st.rerun()
        
        # Main content area
        if st.session_state.current_page == 'dashboard':
            show_dashboard()
        elif st.session_state.current_page == 'course_detail':
            show_course_detail()
        elif st.session_state.current_page == 'create_assignment':
            show_create_assignment()
        elif st.session_state.current_page == 'assignment_detail':
            show_assignment_detail()
        elif st.session_state.current_page == 'submission_detail':
            show_submission_detail()
        elif st.session_state.current_page == 'career_planning':
            show_career_planning()
        # New pages for enhanced services
        elif st.session_state.current_page == 'teacher_analytics':
            show_teacher_analytics()
        elif st.session_state.current_page == 'group_management':
            if st.session_state.current_user['role'] == 'teacher':
                show_group_management()
            else:
                show_student_groups()
        elif st.session_state.current_page == 'student_groups':
            show_student_groups()
        elif st.session_state.current_page == 'study_recommendations':
            show_study_recommendations()
        elif st.session_state.current_page == 'learning_path':
            show_learning_path()
        elif st.session_state.current_page == 'language_settings':
            show_language_settings()
        elif st.session_state.current_page == 'settings':
            show_system_settings()
        elif st.session_state.current_page == 'help':
            show_help_and_support()
        else:
            show_dashboard()

def show_study_recommendations():
    """Display personalized study recommendations for students."""
    st.title("Personalized Study Recommendations")
    
    # Get the current student's ID
    student_id = st.session_state.current_user['id']
    
    # Get student's submissions
    submissions = get_student_submissions(student_id)
    
    # Get student's courses
    courses = get_student_courses(student_id)
    
    if not submissions:
        st.info("You don't have any submissions yet. Here are some general study tips:")
        
        # Show general study tips
        st.subheader("General Study Tips")
        
        tips = [
            "Create a regular study schedule and stick to it",
            "Take breaks using the Pomodoro Technique (25 minutes of study, 5 minutes break)",
            "Find a quiet, distraction-free environment for studying",
            "Stay organized with digital or physical planners",
            "Prioritize tasks based on deadlines and importance",
            "Review notes regularly, not just before exams",
            "Form or join study groups for collaborative learning",
            "Practice active recall instead of passive re-reading",
            "Get enough sleep to consolidate memories and improve focus",
            "Stay hydrated and maintain a balanced diet for optimal brain function"
        ]
        
        for tip in tips:
            st.write(f"• {tip}")
            
        return
    
    # Process submission data to prepare for recommendations
    student_data = {
        'courses': {},
        'skills': {},
        'overall_performance': {
            'average_score': 0,
            'submissions_count': len(submissions),
            'completed_count': 0
        }
    }
    
    # Collect all scores
    all_scores = []
    
    # Process each submission
    for submission in submissions:
        # Get assignment details
        assignment = get_assignment_by_id(submission.get('assignment_id'))
        if not assignment:
            continue
            
        # Get course details
        course_id = assignment.get('course_id')
        course = get_course_by_id(course_id)
        if not course:
            continue
            
        # Initialize course data if not exists
        if course_id not in student_data['courses']:
            student_data['courses'][course_id] = {
                'name': course.get('name'),
                'code': course.get('code'),
                'scores': [],
                'submissions': [],
                'skills': {}
            }
            
        # Add submission data
        score = submission.get('score')
        if score is not None:
            all_scores.append(score)
            student_data['courses'][course_id]['scores'].append(score)
            student_data['overall_performance']['completed_count'] += 1
            
        student_data['courses'][course_id]['submissions'].append(submission)
        
        # Extract skills from assignments and feedback
        assignment_skills = extract_skills_from_assignment(assignment)
        
        # If the assignment has skills defined
        for skill_name in assignment_skills:
            # Update course skills
            if skill_name not in student_data['courses'][course_id]['skills']:
                student_data['courses'][course_id]['skills'][skill_name] = {
                    'scores': [],
                    'count': 0
                }
                
            # Update global skills
            if skill_name not in student_data['skills']:
                student_data['skills'][skill_name] = {
                    'scores': [],
                    'count': 0,
                    'courses': set()
                }
                
            # If submission has score, add it to skill scores
            if score is not None:
                student_data['courses'][course_id]['skills'][skill_name]['scores'].append(score)
                student_data['courses'][course_id]['skills'][skill_name]['count'] += 1
                
                student_data['skills'][skill_name]['scores'].append(score)
                student_data['skills'][skill_name]['count'] += 1
                student_data['skills'][skill_name]['courses'].add(course_id)
    
    # Calculate overall average score
    if all_scores:
        student_data['overall_performance']['average_score'] = sum(all_scores) / len(all_scores)
    
    # Generate recommendations using the service
    recommendations = study_recommendations_service.generate_recommendations(
        student_id=student_id,
        student_data=student_data
    )
    
    if recommendations:
        # Display areas for improvement
        st.subheader("Areas for Improvement")
        improvement_areas = recommendations.get('improvement_areas', [])
        if improvement_areas:
            for area in improvement_areas:
                skill_name = area.get('skill')
                proficiency = area.get('proficiency', 0)
                
                # Calculate proficiency level for progress bar
                if isinstance(proficiency, float) and proficiency <= 1:
                    proficiency_percent = proficiency * 100
                else:
                    proficiency_percent = proficiency
                
                # Determine color based on proficiency
                if proficiency_percent < 60:
                    color = "red"
                elif proficiency_percent < 80:
                    color = "orange"
                else:
                    color = "green"
                
                # Display skill with progress bar
                st.write(f"**{skill_name}**")
                st.progress(float(proficiency_percent / 100))
                st.caption(f"Current proficiency: {proficiency_percent:.1f}%")
                
                # Display improvement tips if available
                if 'tips' in area:
                    with st.expander("Improvement Tips"):
                        for tip in area['tips']:
                            st.write(f"• {tip}")
                
        st.divider()
        else:
            st.success("Great job! You're performing well in all areas.")
        
        # Display recommended resources
        st.subheader("Recommended Resources")
        resources = recommendations.get('resources', [])
        if resources:
            # Group resources by type
            resource_types = {}
            for resource in resources:
                res_type = resource.get('type', 'Other')
                if res_type not in resource_types:
                    resource_types[res_type] = []
                resource_types[res_type].append(resource)
            
            # Display resources by type
            for res_type, type_resources in resource_types.items():
                with st.expander(f"{res_type} ({len(type_resources)})"):
                    for resource in type_resources:
                        st.write(f"**{resource.get('title')}**")
                        st.write(resource.get('description', ''))
                        
                        # Display URL if available
                        if 'url' in resource:
                            st.markdown(f"[Access Resource]({resource['url']})")
                        
                        # Display skill relevance if available
                        if 'relevant_skills' in resource:
                            st.caption(f"Relevant skills: {', '.join(resource['relevant_skills'])}")
                        
                        st.divider()
        else:
            st.info("No specific resources recommended at this time.")
        
        # Display practice activities
        st.subheader("Practice Activities")
        activities = recommendations.get('practice_activities', [])
        if activities:
            for activity in activities:
                with st.expander(activity.get('title', 'Practice Activity')):
                    st.write(activity.get('description', ''))
                    
                    # Display difficulty level if available
                    if 'difficulty' in activity:
                        difficulty = activity['difficulty']
                        st.caption(f"Difficulty: {'●' * difficulty + '○' * (5 - difficulty)}")
                    
                    # Display estimated time if available
                    if 'estimated_time' in activity:
                        st.caption(f"Estimated time: {activity['estimated_time']} minutes")
                    
                    # Display relevant skills if available
                    if 'relevant_skills' in activity:
                        st.caption(f"Skills: {', '.join(activity['relevant_skills'])}")
        else:
            st.info("No specific practice activities recommended at this time.")
        
        # Display study schedule
        st.subheader("Recommended Study Schedule")
        schedule = recommendations.get('schedule', {})
        if schedule:
            # Display time allocation
            if 'time_allocation' in schedule:
                time_allocation = schedule['time_allocation']
                
                # Create DataFrame for time allocation
                allocation_data = []
                for subject, minutes in time_allocation.items():
                    allocation_data.append({
                        'Subject': subject,
                        'Minutes per Day': minutes
                    })
                
                if allocation_data:
                    allocation_df = pd.DataFrame(allocation_data)
                    
                    # Create a horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, max(4, len(allocation_df) * 0.5)))
                    bars = ax.barh(allocation_df['Subject'], allocation_df['Minutes per Day'], color='skyblue')
                    ax.set_xlabel('Minutes per Day')
                    ax.set_title('Recommended Daily Study Time')
                    
                    # Add minutes as text to the right of each bar
                    for i, bar in enumerate(bars):
                        minutes = allocation_df['Minutes per Day'].iloc[i]
                        ax.text(minutes + 2, i, f"{minutes} min", va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Display weekly schedule if available
            if 'weekly_plan' in schedule:
                weekly_plan = schedule['weekly_plan']
                
                # Create a table for the weekly plan
                st.write("**Weekly Study Plan**")
                plan_data = []
                
                days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                for day in days:
                    if day in weekly_plan:
                        day_tasks = weekly_plan[day]
                        plan_data.append({
                            'Day': day,
                            'Focus': ', '.join(day_tasks)
                        })
                
                if plan_data:
                    st.table(pd.DataFrame(plan_data))
        else:
            st.info("No specific study schedule recommended at this time.")
        
        # Display AI study tips
        st.subheader("Personalized Study Tips")
        ai_tips = recommendations.get('ai_tips', [])
        if ai_tips:
            for tip in ai_tips:
                st.write(f"• {tip}")
        else:
            st.info("No personalized study tips available at this time.")
        
        # Display when recommendations were generated
        if 'generated_at' in recommendations:
            try:
                generated_at = datetime.fromisoformat(recommendations['generated_at'])
                st.caption(f"Recommendations generated on {generated_at.strftime('%B %d, %Y at %I:%M %p')}")
            except:
                st.caption("Recommendations recently generated")
    else:
        st.warning("Unable to generate personalized recommendations at this time. Please try again later.")

def extract_skills_from_assignment(assignment):
    """Extract skills from an assignment."""
    skills = []
    
    # Try to get skills from the assignment data
    if 'skills' in assignment:
        assignment_skills = assignment.get('skills', [])
        if isinstance(assignment_skills, list):
            skills.extend(assignment_skills)
        elif isinstance(assignment_skills, str):
            skills.append(assignment_skills)
    
    # If no skills found, try to extract from description
    if not skills and 'description' in assignment:
        # Extract skills from keywords in the description
        description = assignment.get('description', '').lower()
        
        # List of common academic skills
        common_skills = [
            "critical thinking", "problem solving", "research", "writing",
            "analysis", "communication", "presentation", "teamwork",
            "programming", "data analysis", "mathematics", "statistics",
            "reading comprehension", "creativity", "design", "experiment"
        ]
        
        # Check for skills in description
        for skill in common_skills:
            if skill in description:
                skills.append(skill.title())
    
    # If still no skills, add generic skill based on assignment title
    if not skills and 'title' in assignment:
        title = assignment.get('title', '').lower()
        if 'essay' in title or 'writing' in title:
            skills.append('Writing')
        elif 'problem' in title or 'exercise' in title:
            skills.append('Problem Solving')
        elif 'research' in title:
            skills.append('Research')
        elif 'presentation' in title:
            skills.append('Presentation')
        elif 'quiz' in title or 'test' in title:
            skills.append('Knowledge Recall')
        else:
            skills.append('General Academic Skills')
    
    return skills

def show_learning_path():
    """Display the interactive learning path visualization for students."""
    st.title("Interactive Learning Path")
    
    # Verify the user is a student
    if st.session_state.current_user['role'] != 'student':
        st.warning("Learning path visualization is only available for students.")
        return
    
    # Get the student's ID
    student_id = st.session_state.current_user['id']
    
    # Get student's courses
    courses = get_student_courses(student_id)
    
    if not courses:
        st.info("You're not enrolled in any courses yet. Please enroll in courses to see your learning path.")
        return
    
    # Let the student select a course for the learning path
    selected_course = st.selectbox(
        "Select a course to generate learning path",
        options=courses,
        format_func=lambda x: f"{x['name']} ({x['code']})"
    )
    
    if not selected_course:
        st.warning("Please select a course to view your learning path.")
        return
    
    # Get student's submissions for this course
    submissions = get_student_submissions(student_id)
    course_submissions = []
    
    for submission in submissions:
        assignment = get_assignment_by_id(submission.get('assignment_id'))
        if assignment and assignment.get('course_id') == selected_course['id']:
            # Add assignment info to submission
            submission['assignment'] = assignment
            course_submissions.append(submission)
    
    if not course_submissions:
        st.warning(f"You don't have any submissions for {selected_course['name']} yet.")
        st.info("Your learning path will be based on the course curriculum. Submit assignments to get a personalized path.")
        
        # Get course assignments to infer skills
        assignments = get_course_assignments(selected_course['id'])
        if not assignments:
            st.error("This course doesn't have any assignments yet.")
            return
        
        # Extract skills from assignments
        skill_data = {}
        for assignment in assignments:
            skills = extract_skills_from_assignment(assignment)
            for skill in skills:
                if skill not in skill_data:
                    skill_data[skill] = {
                        'name': skill,
                        'status': 'not_started',
                        'proficiency': 0.0,
                        'assignments': []
                    }
                skill_data[skill]['assignments'].append(assignment['id'])
    else:
        # Process submissions to collect skill data
        skill_data = {}
        
        for submission in course_submissions:
            assignment = submission.get('assignment')
            if not assignment:
                continue
                
            # Extract skills from the assignment
            skills = extract_skills_from_assignment(assignment)
            
            for skill in skills:
                if skill not in skill_data:
                    skill_data[skill] = {
                        'name': skill,
                        'scores': [],
                        'proficiency': 0.0,
                        'status': 'not_started',
                        'assignments': []
                    }
                
                # Add assignment to this skill
                if assignment['id'] not in skill_data[skill]['assignments']:
                    skill_data[skill]['assignments'].append(assignment['id'])
                
                # Add score if available
                score = submission.get('score')
                if score is not None:
                    # Normalize score to 0-1 range
                    max_points = assignment.get('points', 100)
                    normalized_score = score / max_points
                    skill_data[skill]['scores'].append(normalized_score)
        
        # Calculate average scores and proficiency levels
        for skill_name, skill in skill_data.items():
            if skill['scores']:
                avg_score = sum(skill['scores']) / len(skill['scores'])
                skill['proficiency'] = avg_score
                
                # Set status based on proficiency
                if avg_score >= 0.8:
                    skill['status'] = 'mastered'
                elif avg_score >= 0.6:
                    skill['status'] = 'proficient'
                elif avg_score > 0:
                    skill['status'] = 'developing'
                else:
                    skill['status'] = 'not_started'
            else:
                skill['status'] = 'not_started'
                skill['proficiency'] = 0.0
    
    # Generate learning path visualization
    if skill_data:
        # Convert skill_data dictionary to list
        skills_list = [{'name': k, **v} for k, v in skill_data.items()]
        
        # Generate learning path using the service
        learning_path = learning_path_service.generate_learning_path(
            student_id=student_id,
            course_id=selected_course['id'],
            skills=skills_list
        )
        
        if learning_path:
            # Display the skill map visualization
            st.subheader("Your Skill Map")
            
            # Display the visualization from the learning path service
            skill_graph_html = learning_path.get('visualization_html')
            if skill_graph_html:
                st.components.v1.html(skill_graph_html, height=600)
            else:
                # Fallback to a simpler visualization
                st.warning("Interactive visualization couldn't be generated. Showing simplified view.")
                
                # Group skills by status
                status_groups = {
                    'mastered': [],
                    'proficient': [],
                    'developing': [],
                    'not_started': []
                }
                
                for skill in skills_list:
                    status = skill.get('status', 'not_started')
                    status_groups[status].append(skill)
                
                # Display skills by status
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Mastered Skills**")
                    if status_groups['mastered']:
                        for skill in status_groups['mastered']:
                            st.success(f"{skill['name']} ({skill['proficiency']*100:.1f}%)")
                    else:
                        st.info("No mastered skills yet")
                    
                    st.write("**Developing Skills**")
                    if status_groups['developing']:
                        for skill in status_groups['developing']:
                            st.warning(f"{skill['name']} ({skill['proficiency']*100:.1f}%)")
                else:
                        st.info("No developing skills yet")
                
                with col2:
                    st.write("**Proficient Skills**")
                    if status_groups['proficient']:
                        for skill in status_groups['proficient']:
                            st.info(f"{skill['name']} ({skill['proficiency']*100:.1f}%)")
                    else:
                        st.info("No proficient skills yet")
                    
                    st.write("**Not Started Skills**")
                    if status_groups['not_started']:
                        for skill in status_groups['not_started']:
                            st.error(f"{skill['name']} (0%)")
                    else:
                        st.info("No skills remaining to start")
            
            # Display skill proficiency summary
            st.subheader("Skill Proficiency Summary")
            
            # Create a DataFrame for the skills
            skill_summary = []
            for skill in skills_list:
                skill_summary.append({
                    'Skill': skill['name'],
                    'Proficiency': skill['proficiency'] * 100,
                    'Status': skill['status'].replace('_', ' ').title()
                })
            
            if skill_summary:
                # Sort by proficiency (highest first)
                skill_df = pd.DataFrame(skill_summary).sort_values('Proficiency', ascending=False)
                
                # Display as a bar chart
                fig, ax = plt.subplots(figsize=(10, max(5, len(skill_df) * 0.4)))
                
                # Create bars with colors based on status
                bars = ax.barh(skill_df['Skill'], skill_df['Proficiency'])
                for i, bar in enumerate(bars):
                    status = skill_df['Status'].iloc[i]
                    if status == 'Mastered':
                        bar.set_color('green')
                    elif status == 'Proficient':
                        bar.set_color('skyblue')
                    elif status == 'Developing':
                        bar.set_color('orange')
                    else:
                        bar.set_color('lightgray')
                
                # Add proficiency as text to the right of each bar
                for i, v in enumerate(skill_df['Proficiency']):
                    ax.text(v + 1, i, f"{v:.1f}%", va='center')
                
                ax.set_xlabel('Proficiency (%)')
                ax.set_xlim(0, 100)
                plt.tight_layout()
                
                st.pyplot(fig)
            
            # Display as a table
                st.dataframe(skill_df)
            
            # Display recommended resources
            if 'recommended_resources' in learning_path:
                st.subheader("Recommended Resources")
                resources = learning_path['recommended_resources']
                
                for resource in resources:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{resource.get('title')}**")
                        st.write(resource.get('description', ''))
                        
                        # Display related skills
                        if 'related_skills' in resource:
                            st.caption(f"Related skills: {', '.join(resource['related_skills'])}")
                    
                    with col2:
                        if 'url' in resource:
                            st.link_button("Open Resource", resource['url'])
                    
                    st.divider()
            
            # Display suggested next steps
            if 'next_steps' in learning_path:
                st.subheader("Suggested Next Steps")
                steps = learning_path['next_steps']
                
                for i, step in enumerate(steps, 1):
                    st.write(f"**{i}. {step.get('title')}**")
                    st.write(step.get('description', ''))
                    
                    # Display priority if available
                    if 'priority' in step:
                        priority = step['priority']
                        if priority == 'high':
                            st.error("Priority: High")
                        elif priority == 'medium':
                            st.warning("Priority: Medium")
                        else:
                            st.info("Priority: Low")
                    
                    st.divider()
        else:
            st.error("Unable to generate learning path. Please try again later.")
    else:
        st.error("No skill data available for generating a learning path.")

def show_group_management():
    """Display interface for creating and managing student groups (for teachers)."""
    st.title("Group Management")
    
    # Verify the user is a teacher
    if st.session_state.current_user['role'] != 'teacher':
        st.warning("Group management is only available for teachers.")
        return
    
    # Get teacher's courses
    teacher_id = st.session_state.current_user['id']
    courses = get_teacher_courses(teacher_id)
    
    if not courses:
        st.info("You haven't created any courses yet. Create a course to manage student groups.")
        return
    
    # Let the teacher select a course
    selected_course = st.selectbox(
        "Select Course",
        options=courses,
        format_func=lambda x: f"{x['name']} ({x['code']})"
    )
    
    if not selected_course:
        st.warning("Please select a course to manage groups.")
        return
    
    # Get students enrolled in this course
    student_ids = selected_course.get('students', [])
    students = []
    
    for student_id in student_ids:
        student = get_user_by_id(student_id)
        if student:
            # For demonstration purposes, add mock skill levels, learning styles, interests
            # In a real app, these would come from a database
            student['mock_skills'] = {
                'programming': random.randint(1, 10),
                'writing': random.randint(1, 10),
                'problem_solving': random.randint(1, 10),
                'communication': random.randint(1, 10),
                'teamwork': random.randint(1, 10)
            }
            student['mock_learning_style'] = random.choice(['visual', 'auditory', 'kinesthetic', 'reading/writing'])
            student['mock_interests'] = random.sample(['AI', 'Web Development', 'Data Science', 'Mobile Apps', 'Cybersecurity', 'Game Development'], k=random.randint(1, 3))
            
            # Get student's submissions and calculate performance
            submissions = get_student_submissions(student_id)
            course_submissions = []
            total_score = 0
            submission_count = 0
            
            for submission in submissions:
                assignment = get_assignment_by_id(submission.get('assignment_id'))
                if assignment and assignment.get('course_id') == selected_course['id']:
                    score = submission.get('score')
                    if score is not None:
                        total_score += score
                        submission_count += 1
            
            if submission_count > 0:
                student['performance'] = total_score / submission_count
            else:
                student['performance'] = 50  # Default performance
            
            students.append(student)
    
    # Display the number of students
    st.write(f"Total students enrolled: **{len(students)}**")
    
    # Create tabs for different actions
    tab1, tab2, tab3 = st.tabs(["Create Groups", "View Groups", "Group Analysis"])
    
    with tab1:
        st.subheader("Create New Groups")
        
        # Group size input
        group_size = st.number_input("Target Group Size", min_value=2, max_value=10, value=4)
        
        # Grouping algorithm selection
        algorithm = st.selectbox(
            "Grouping Algorithm",
            options=["balanced", "homogeneous", "heterogeneous", "random"],
            format_func=lambda x: {
                "balanced": "Balanced Groups (mix of skill levels)",
                "homogeneous": "Similar Skills Groups (students with similar abilities)",
                "heterogeneous": "Diverse Skills Groups (mix of different skills)",
                "random": "Random Groups"
            }.get(x, x)
        )
        
        # Explain the selected algorithm
        algorithm_descriptions = {
            "balanced": "Creates groups with balanced overall skill levels. Each group will have a mix of high and low performers.",
            "homogeneous": "Groups students with similar skill profiles together. Good for targeted instruction.",
            "heterogeneous": "Maximizes skill diversity within each group. Good for peer learning and cross-skill collaboration.",
            "random": "Assigns students to groups randomly. Simple and unbiased."
        }
        
        st.info(algorithm_descriptions.get(algorithm, ""))
        
        # Grouping criteria
        st.write("**Grouping Criteria**")
        criteria = st.multiselect(
            "Select criteria to consider when forming groups",
            options=["performance", "skills", "learning_style", "interests"],
            default=["performance"],
            format_func=lambda x: {
                "performance": "Academic Performance",
                "skills": "Skill Levels",
                "learning_style": "Learning Styles",
                "interests": "Interests/Topics"
            }.get(x, x)
        )
        
        # Constraints
        st.write("**Group Constraints (Optional)**")
        st.caption("Add specific constraints for student pairings")
        
        # Allow adding constraints
        if 'group_constraints' not in st.session_state:
            st.session_state.group_constraints = []
        
        constraint_type = st.selectbox(
            "Constraint Type",
            options=["must_be_together", "cannot_be_together"]
        )
        
        # Student selection for constraints
        if students:
            col1, col2 = st.columns(2)
            with col1:
                student1 = st.selectbox("Student 1", options=students, format_func=lambda x: x['name'])
            with col2:
                # Filter out the first student
                remaining_students = [s for s in students if s['id'] != student1['id']]
                student2 = st.selectbox("Student 2", options=remaining_students, format_func=lambda x: x['name'])
            
            # Add constraint button
            if st.button("Add Constraint"):
                new_constraint = {
                    "type": constraint_type,
                    "student1_id": student1['id'],
                    "student2_id": student2['id'],
                    "student1_name": student1['name'],
                    "student2_name": student2['name']
                }
                
                # Check if this constraint already exists
                if not any(c['student1_id'] == new_constraint['student1_id'] and 
                          c['student2_id'] == new_constraint['student2_id'] and
                          c['type'] == new_constraint['type'] for c in st.session_state.group_constraints):
                    st.session_state.group_constraints.append(new_constraint)
        
        # Display current constraints
        if st.session_state.group_constraints:
            st.write("**Current Constraints:**")
            for i, constraint in enumerate(st.session_state.group_constraints):
                if constraint['type'] == "must_be_together":
                    st.success(f"{constraint['student1_name']} must be in the same group as {constraint['student2_name']}")
                else:
                    st.error(f"{constraint['student1_name']} cannot be in the same group as {constraint['student2_name']}")
                
                # Allow removing constraints
                if st.button(f"Remove", key=f"remove_constraint_{i}"):
                    st.session_state.group_constraints.pop(i)
                        st.rerun()
        
        # Smart recommendation option
        st.write("**Smart Recommendation**")
        smart_recommendation = st.checkbox("Get AI recommendation for best grouping strategy")
        
        if smart_recommendation:
            # Analyze past group performance and recommend the best strategy
            st.info("Analyzing past group performance to suggest optimal grouping strategy...")
            
            # In a real app, this would analyze actual past data
            # For demo, we'll provide a suggestion based on algorithm
            if algorithm == "balanced":
                st.success("Recommendation: Balanced groups are optimal for this course based on past performance data.")
            elif algorithm == "heterogeneous":
                st.success("Recommendation: Diverse skill groups have shown the best performance in similar courses.")
            elif algorithm == "homogeneous":
                st.success("Recommendation: Similar skill groups work well for this type of course content.")
                    else:
                st.warning("Recommendation: Consider using a skill-based grouping algorithm rather than random assignment.")
        
        # Create groups button
        if st.button("Create Groups"):
            if not criteria:
                st.error("Please select at least one grouping criterion.")
            else:
                with st.spinner("Creating optimal student groups..."):
                    # Prepare student data for the group formation service
                    student_data = []
                    for student in students:
                        student_data.append({
                            'id': student['id'],
                            'name': student['name'],
                            'performance': student['performance'],
                            'skills': student['mock_skills'],
                            'learning_style': student['mock_learning_style'],
                            'interests': student['mock_interests']
                        })
                    
                    # Get group assignments using the service
                    groups = group_formation_service.create_groups(
                        course_id=selected_course['id'],
                        students=student_data,
                        group_size=group_size,
                        algorithm=algorithm,
                        criteria=criteria,
                        constraints=st.session_state.group_constraints
                    )
                    
                    if groups:
                        st.success(f"Successfully created {len(groups)} groups!")
                        
                        # Display the groups
                        for i, group in enumerate(groups, 1):
                            with st.expander(f"Group {i} ({len(group['members'])} students)"):
                                # Display group members
                                for member in group['members']:
                                    st.write(f"• {member['name']}")
                                
                                # Display group metrics if available
                                if 'metrics' in group:
                                    st.divider()
                                    st.write("**Group Metrics:**")
                                    metrics = group['metrics']
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Avg. Performance", f"{metrics.get('avg_performance', 0):.1f}%")
                                    with col2:
                                        st.metric("Skill Diversity", f"{metrics.get('skill_diversity', 0):.1f}/10")
                        
                        # Save the group assignment
                        st.session_state.last_group_assignment = {
                            'course_id': selected_course['id'],
                            'groups': groups,
                            'algorithm': algorithm,
                            'criteria': criteria,
                            'created_at': datetime.now().isoformat()
                        }
                    else:
                        st.error("Failed to create groups. Please try different parameters.")
    
    with tab2:
        st.subheader("View Existing Groups")
        
        # Get group assignments for this course
        group_assignments = group_formation_service.get_group_assignments(course_id=selected_course['id'])
        
        if not group_assignments:
            st.info("No group assignments found for this course.")
                    else:
            # Selectbox to choose which assignment to view
            selected_assignment = st.selectbox(
                "Select Group Assignment",
                options=group_assignments,
                format_func=lambda x: f"{x.get('name', 'Unnamed Assignment')} - {x.get('created_at', 'Unknown date')}"
            )
            
            if selected_assignment:
                st.write(f"**Created:** {selected_assignment.get('created_at')}")
                st.write(f"**Algorithm:** {selected_assignment.get('algorithm')}")
                
                # Display groups
                groups = selected_assignment.get('groups', [])
                for i, group in enumerate(groups, 1):
                    with st.expander(f"Group {i} ({len(group.get('members', []))} students)"):
                        # Display group members
                        for member in group.get('members', []):
                            st.write(f"• {member.get('name', 'Unknown')}")
                        
                        # Display group feedback if available
                        if 'feedback' in group:
                            st.divider()
                            st.write("**Group Feedback:**")
                            st.write(group['feedback'])
                        
                        # Display group performance if available
                        if 'performance' in group:
                            st.divider()
                            st.write("**Group Performance:**")
                            st.metric("Performance Score", f"{group['performance']:.1f}/10")
    
    with tab3:
        st.subheader("Group Effectiveness Analysis")
        
        # Get group assignments for this course
        group_assignments = group_formation_service.get_group_assignments(course_id=selected_course['id'])
        
        if not group_assignments or len(group_assignments) < 2:
            st.info("Group analysis requires at least two group assignments with performance data.")
                            else:
            st.write("Analyzing the effectiveness of different grouping algorithms based on past assignments...")
            
            # Analyze group performance by algorithm
            algorithm_performance = group_formation_service.analyze_group_effectiveness(
                course_id=selected_course['id'],
                assignments=group_assignments
            )
            
            if algorithm_performance:
                # Display algorithm performance
                st.subheader("Algorithm Performance")
                
                # Create DataFrame for the chart
                perf_data = []
                for alg, score in algorithm_performance.get('algorithm_scores', {}).items():
                    perf_data.append({
                        'Algorithm': alg,
                        'Average Score': score
                    })
                
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(perf_df['Algorithm'], perf_df['Average Score'], color='skyblue')
                    
                    # Add score as text on top of each bar
                    for i, bar in enumerate(bars):
                        score = perf_df['Average Score'].iloc[i]
                        ax.text(i, score + 0.1, f"{score:.1f}", ha='center')
                    
                    ax.set_ylabel('Average Group Score')
                    ax.set_ylim(0, 10)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                
                # Display recommendations
                st.subheader("Recommendations")
                
                best_algorithm = algorithm_performance.get('best_algorithm', '')
                best_score = algorithm_performance.get('best_score', 0)
                
                st.success(f"Based on the analysis, the **{best_algorithm}** algorithm has performed best with an average score of **{best_score:.1f}/10**.")

def show_student_groups():
    """Display group assignments for a student."""
    st.title("My Group Assignments")
    
    # Verify the user is a student
    if st.session_state.current_user['role'] != 'teacher':
        # Get the student's ID
        student_id = st.session_state.current_user['id']
        
        # Get student's group assignments using the service
        with st.spinner("Loading your groups..."):
            group_assignments = group_formation_service.get_student_groups(student_id=student_id)
        
        if not group_assignments:
            st.info("You haven't been assigned to any groups yet.")
                else:
            # Display each group assignment
            for assignment in group_assignments:
                course_id = assignment.get('course_id')
                course = get_course_by_id(course_id)
                course_name = course.get('name', 'Unknown Course') if course else 'Unknown Course'
                
                # Find the group containing this student
                student_group = None
                group_number = 0
                
                for i, group in enumerate(assignment.get('groups', []), 1):
                    if any(member.get('id') == student_id for member in group.get('members', [])):
                        student_group = group
                        group_number = i
                        break
                
                if student_group:
                    with st.expander(f"{course_name} - Group {group_number}"):
                        # Display creation date
                        created_at = assignment.get('created_at', '')
                        if created_at:
                            try:
                                created_date = datetime.fromisoformat(created_at).strftime("%B %d, %Y")
                                st.caption(f"Created on {created_date}")
                            except:
                                st.caption(f"Created on {created_at}")
                        
                        # Display group members
                        st.write("**Group Members:**")
                        for member in student_group.get('members', []):
                            if member.get('id') == student_id:
                                st.write(f"• **{member.get('name', 'You')} (You)**")
                            else:
                                st.write(f"• {member.get('name', 'Unknown')}")
                        
                        # Display group feedback if available
                        if 'feedback' in student_group:
                            st.divider()
                            st.write("**Group Feedback:**")
                            st.write(student_group['feedback'])
                        
                        # Display group performance if available
                        if 'performance' in student_group:
                            st.divider()
                            st.write("**Group Performance:**")
                            score = student_group['performance']
                            
                            if score >= 8:
                                st.success(f"Performance Score: {score:.1f}/10")
                            elif score >= 6:
                                st.info(f"Performance Score: {score:.1f}/10")
                            else:
                                st.warning(f"Performance Score: {score:.1f}/10")
    else:
        # For teachers, redirect to the group management page
        show_group_management()

if __name__ == "__main__":
    main()