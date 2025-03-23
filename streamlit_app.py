from edumate.services.gemini_service import GeminiService
from edumate.services.feedback_service import FeedbackService
from edumate.services.plagiarism_service import PlagiarismService
from edumate.services.grading_service import GradingService
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
from edumate.utils.teacher_tools import TeacherTools
from edumate.utils.classroom_manager import ClassroomManager
from edumate.utils.exam_manager import ExamManager
from edumate.utils.indian_education import IndianEducationSystem
from edumate.utils.career_planner import CareerPlanner
from edumate.utils.audit import AuditTrail
from edumate.utils.analytics import Analytics
from edumate.utils.encryption import Encryptor
from edumate.utils.logger import log_system_event, log_access, log_error, log_audit
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
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

# Import modular page components
from edumate.pages.test_creator import show_test_creator
from edumate.pages.ai_tutor import show_ai_tutor_page

# Set matplotlib style for better looking charts
plt.style.use('seaborn-v0_8')

# Import utilities

# Import services

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
study_recommendations_service = StudyRecommendationsService(
    gemini_service=gemini_service, data_dir='data')
group_formation_service = GroupFormationService()
learning_path_service = LearningPathService()
multilingual_feedback_service = MultilingualFeedbackService(
    gemini_service=gemini_service, data_dir='data')
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
                {"title": "Python for Everybody", "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/python"},
                {"title": "The Complete Web Developer Course", "platform": "Udemy",
                    "url": "https://www.udemy.com/course/the-complete-web-developer-course-2"}
            ],
            "data_analysis": [
                {"title": "Data Science Specialization", "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/jhu-data-science"},
                {"title": "Data Analyst Nanodegree", "platform": "Udacity",
                    "url": "https://www.udacity.com/course/data-analyst-nanodegree--nd002"}
            ],
            "communication": [
                {"title": "Effective Communication", "platform": "LinkedIn Learning",
                    "url": "https://www.linkedin.com/learning/topics/communication"},
                {"title": "Public Speaking", "platform": "Coursera",
                    "url": "https://www.coursera.org/learn/public-speaking"}
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


def create_assignment(
    title,
    description,
    course_id,
    teacher_id,
    due_date,
     points=100):
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
    return [
    assignment for assignment in assignments if assignment['course_id'] == course_id]


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
    assignment_submissions = [
    sub for sub in submissions if sub['assignment_id'] == assignment_id]

    if assignment_submissions:
        # If there are submissions, we should handle them
        # Option 1: Delete all submissions (implemented here)
        # Option 2: Prevent deletion if submissions exist (alternative
        # approach)

        # Delete all related submissions and their files
        for submission in assignment_submissions:
            # Delete any attached files
            if submission.get(
                'file_info') and submission['file_info'].get('file_path'):
                file_path = submission['file_info']['file_path']
                if os.path.exists(file_path):
                    try:
                        # removing real file instead of placeholder
                        os.remove(file_path)
                    except Exception as e:
                        st.error(f"Failed to delete file {file_path}: {e}")

        # Remove all submissions for this assignment
        submissions = [
    sub for sub in submissions if sub['assignment_id'] != assignment_id]
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
    if any(sub['assignment_id'] == assignment_id and sub['student_id']
           == student_id for sub in submissions):
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


def analyze_with_gemini(
    content_type,
    file_path,
    prompt,
    mime_type,
     model="gemini-2.0-pro"):
    """Analyze content using specific Gemini model"""
    if not GEMINI_API_KEY:
        st.error(
            "Gemini API key not found. Please add GEMINI_API_KEY to your .env file.")
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
                        return part.get('text', 'No text found')

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

                image_filename = f"page{
    page_num +
    1}_img{
        img_index +
         1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)

                with open(image_path, "wb") as image_file:

                    image_file.write(image_bytes)

                image_paths.append(image_path)

        # If no images found, try extracting as whole page images
        if not image_paths:
            for page_num, page in enumerate(pdf_document):
                pix = page.get_pixmap()
                image_filename = f"page{page_num + 1}.png"
                image_path = os.path.join(output_dir, image_filename)
                pix.save(image_path)
                image_paths.append(image_path)

        return image_paths

    except ImportError as e:
        st.error(
    f"Error: {
        str(e)}. Please install PyMuPDF using: pip install PyMuPDF")
        return []
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
        return []


# Add imports for our enhanced grading services at the top of the file


def auto_grade_submission(submission_id):
    """
    Automatically grade a submission using AI.
    Returns: (success, message)
    """
    try:
        # Load submissions and get the specific submission
        submissions = load_data('submissions')
        submission = next(
    (s for s in submissions if s['id'] == submission_id), None)

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
                file_analysis = analyze_file_content(
                    file_content, submission['file_info']['filename'])
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
                    {"name": "Content", "weight": 40,
                        "description": "Completeness and quality of content"},
                    {"name": "Understanding", "weight": 30,
                        "description": "Demonstrates understanding of concepts"},
                    {"name": "Presentation", "weight": 20,
                        "description": "Organization and clarity"},
                    {"name": "Technical", "weight": 10,
                        "description": "Technical correctness"}
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
                # Max 25% reduction
                reduction = min(
                    grading_result["score"] * (plagiarism_result["score"] / 200), 25)
                grading_result["score"] -= reduction
                grading_result["feedback"] += f"\n\n⚠️ **Plagiarism Warning**: This submission shows {plagiarism_result['score']:.1f}% similarity with existing sources. Points have been deducted accordingly."

        except Exception as e:
            log_error(f"Error checking plagiarism: {str(e)}")

        # Generate AI feedback (comprehensive version from service)
        ai_feedback = grading_result.get("feedback", "")
        if not ai_feedback:
            # Fallback to legacy feedback generator if service doesn't provide feedback
            ai_feedback = generate_ai_feedback(
    submission, file_content, file_analysis)

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
    strengths_match = re.search(
    r'(?:STRENGTHS|Strengths):(.*?)(?:\n\n|\n[A-Z]|$)',
    feedback,
     re.DOTALL | re.IGNORECASE)
    if strengths_match:
        strengths_text = strengths_match.group(1)
        strength_points = re.findall(
    r'[-*â€¢]\s*(.*?)(?:\n[-*â€¢]|\n\n|$)',
    strengths_text,
     re.DOTALL)
        for point in strength_points:
            if point.strip():
                points.append({"type": "strength", "text": point.strip()})

    # Extract weaknesses/areas for improvement section
    weaknesses_match = re.search(r'(?:WEAKNESSES|AREAS FOR IMPROVEMENT|Areas for improvement|Weaknesses):(.*?)(?:\n\n|\n[A-Z]|$)',
                                feedback, re.DOTALL | re.IGNORECASE)
    if weaknesses_match:
        weaknesses_text = weaknesses_match.group(1)
        weakness_points = re.findall(
    r'[-*â€¢]\s*(.*?)(?:\n[-*â€¢]|\n\n|$)',
    weaknesses_text,
     re.DOTALL)
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
        analysis += f"- Average sentence length: {
    avg_sentence_length:.1f} words\n"

    # File type specific analysis
    if file_ext == '.pdf':
        analysis += "- Document appears to be a PDF, which typically indicates a formal submission.\n"
    elif file_ext == '.docx':
        analysis += "- Document is in Word format, which is appropriate for academic submissions.\n"
    elif file_ext == '.txt':
        analysis += "- Document is in plain text format. Consider using a more formatted document type for future submissions.\n"

    return analysis


def generate_ai_feedback(
    submission,
    file_content="",
    file_analysis="",
     gemini_analysis=""):
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
        improvements.append(
            "Your text submission is quite brief. Consider expanding your answer with more details.")

    else:
        strengths.append(
            "You provided a detailed text response with good length.")

    if "because" in content.lower() or "therefore" in content.lower():
        strengths.append(
            "Good use of reasoning and logical connections in your answer.")

    else:
        improvements.append(
            "Try to include more reasoning and logical connections in your answer.")

    if len(content.split('.')) > 5:
        strengths.append(
            "Good structure with multiple sentences in your text submission.")

    else:
        improvements.append(
            "Consider structuring your text answer with more complete sentences.")

    # File submission analysis
    if submission.get('file_info'):
        file_info = submission.get('file_info')
        strengths.append(
    f"You submitted a file ({
        file_info['filename']}) which demonstrates thoroughness.")

        # Add file-specific feedback
        if file_info['file_type'].endswith('pdf'):
            strengths.append(
                "Your PDF submission is in a professional format suitable for academic work.")

            # If we have Gemini analysis, it means it was a handwritten or
            # complex PDF
            if gemini_analysis:
                strengths.append(
                    "Your PDF was analyzed directly by our AI system, including any handwritten content.")

                # Check if the analysis mentions handwritten content
                if "handwritten" in gemini_analysis.lower(
                ) or "handwriting" in gemini_analysis.lower():
                    strengths.append(
                        "Your handwritten work shows dedication and personal effort in your submission.")
        elif file_info['file_type'].endswith('docx'):
            strengths.append(
                "Your Word document submission follows standard academic formatting.")
        elif file_info['file_type'].endswith('txt'):
            improvements.append(
                "Consider using a more formatted document type (like PDF or DOCX) for future submissions.")

    else:
        improvements.append(
            "Consider attaching a file with your submission for more comprehensive work.")

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
        return False
    else:
        return True


def get_assignment_submissions(assignment_id):
    try:
        submissions = load_data('submissions')
        return [sub for sub in submissions if sub['assignment_id'] == assignment_id]
    except Exception as e:
        st.error(f"Error loading submissions: {e}")
        return []


def get_student_submissions(student_id):
    try:
        submissions = load_data('submissions')
        return [sub for sub in submissions if sub['student_id'] == student_id]
    except Exception as e:
        st.error(f"Error loading submissions: {e}")
        return []

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
                st.error("Please fill in all fields.")
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


def show_student_dashboard():
    """Display the student dashboard."""
    st.subheader("Your Courses")
    
    # Get student's courses
    student_id = st.session_state.current_user['id']
    courses = get_student_courses(student_id)
    
    if not courses:
        st.info("You are not enrolled in any courses yet.")
    else:
        # Display courses in a grid
        cols = st.columns(3)
        for i, course in enumerate(courses):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(course['name'])
                    st.caption(f"Code: {course['code']}")
                    st.write(course['description'][:100] + "..." if len(course['description']) > 100 else course['description'])
                    
                    # View course button
                    if st.button("View Course", key=f"view_course_{course['id']}"):
                        st.session_state.selected_course_id = course['id']
                        st.session_state.current_page = 'course_detail'
                        st.rerun()
    
    # Recent assignments section
    st.subheader("Recent Assignments")
    
    # Get student's submissions
    submissions = get_student_submissions(student_id)
    
    if not submissions:
        st.info("No recent assignment submissions.")
    else:
        # Sort by submission date, most recent first
        submissions.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
        
        # Show recent submissions
        for submission in submissions[:5]:  # Show only 5 most recent
            assignment = get_assignment_by_id(submission['assignment_id'])
            course = get_course_by_id(assignment['course_id']) if assignment else None
            
            if assignment and course:
                with st.container(border=True):
                    cols = st.columns([3, 2, 1, 1])
                    with cols[0]:
                        st.write(f"**{assignment['title']}**")
                        st.caption(f"Course: {course['name']}")
                    with cols[1]:
                        st.write("Submitted: " + submission.get('submitted_at', 'N/A')[:10])
                    with cols[2]:
                        if 'score' in submission:
                            st.write(f"Score: {submission['score']}/{assignment['points']}")
                        else:
                            st.write("Not graded")
                    with cols[3]:
                        if st.button("View", key=f"view_sub_{submission['id']}"):
                            st.session_state.selected_submission_id = submission['id']
                            st.session_state.current_page = 'submission_detail'
                            st.rerun()


def show_test_creator():
    """Display the test creation interface for teachers."""
    st.header("Create Test")

    # Get current teacher and course
    teacher_id = st.session_state.current_user['id']
    selected_course = st.session_state.get('selected_course')

    if not selected_course:
        st.error("Please select a course first.")
        return

        # Load existing tests data
    try:
        tests_data = load_data('tests')
    except:
        tests_data = []

    # Generate a unique test ID if creating a new test
    if 'editing_test' not in st.session_state:
        test_id = f"test_{int(time.time())}_{teacher_id}"
    else:
        test_id = st.session_state.editing_test.get('id')

    # Check if we're editing an existing test
    editing = 'editing_test' in st.session_state
    current_test = st.session_state.get('editing_test', {})

    # Create test form
    with st.form("create_test_form"):
        test_title = st.text_input(
    "Test Title", value=current_test.get(
        'name', ''))
        test_description = st.text_area(
    "Description", value=current_test.get(
        'description', ''))

        # Test settings
        st.subheader("Test Settings")
        col1, col2 = st.columns(2)

        with col1:
            time_limit = st.number_input(
    "Time Limit (minutes)",
    min_value=5,
    value=current_test.get(
        'duration',
         60))
            max_attempts = st.number_input(
    "Maximum Attempts",
    min_value=1,
    value=current_test.get(
        'max_attempts',
         1))
            due_date = st.date_input(
    "Due Date",
    value=datetime.strptime(
        current_test.get(
            'due_date',
            datetime.now().isoformat()),
             "%Y-%m-%dT%H:%M:%S") if 'due_date' in current_test else None)

        with col2:
            passing_score = st.slider(
    "Passing Score (%)",
    min_value=50,
    max_value=100,
    value=current_test.get(
        'passing_score',
         70))
            show_answers = st.checkbox(
    "Show Answers After Completion",
    value=current_test.get(
        'show_answers',
         False))
            points = st.number_input(
    "Total Points",
    min_value=1,
    value=current_test.get(
        'points',
         100))

        # Questions section
        st.subheader("Questions")

        # Extract existing questions if editing
        questions = current_test.get('questions', [])

        # Add questions interface (simplified for brevity)
        question_types = [
    "Multiple Choice",
    "True/False",
    "Short Answer",
    "Essay",
     "Fill in the Blank"]

        # Display existing questions
        for i, question in enumerate(questions):
            with st.expander(f"Question {i + 1}: {question.get('text', '')[0:50]}..."):
                st.text_input(
    f"Question Text {
        i + 1}",
        value=question.get(
            'text',
            ''),
             key=f"q_text_{i}")

                q_type = st.selectbox(f"Question Type {i + 1}",
                                     options=question_types,
                                     index=question_types.index(
                                         question.get('type', 'Multiple Choice')),
                                     key=f"q_type_{i}")

                q_points = st.number_input(
    f"Points {
        i + 1}",
        min_value=1,
        value=question.get(
            'points',
            10),
             key=f"q_points_{i}")

                # Show options for multiple choice
                if q_type == "Multiple Choice":
                    options = question.get('options', ['', '', '', ''])
                    correct = question.get('correct', 0)

                    st.write("Options (mark correct answer):")
                    for j, option in enumerate(options):
                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            is_correct = st.checkbox("", value=(
                                j == correct), key=f"q_{i}_correct_{j}")
                            if is_correct:
                                correct = j
        with col2:
                            options[j] = st.text_input(
                                f"Option {j + 1}", value=option, key=f"q_{i}_opt_{j}")

                # Update question in the list
                questions[i] = {
                    'text': st.session_state.get(f"q_text_{i}"),
                    'type': st.session_state.get(f"q_type_{i}"),
                    'points': st.session_state.get(f"q_points_{i}"),
                    'options': options if q_type == "Multiple Choice" else None,
                    'correct': correct if q_type == "Multiple Choice" else None
                }
        
        # Option to use AI for question generation
        use_ai = st.checkbox("Use AI to Generate Questions")
        if use_ai:
            ai_topic = st.text_input("Topic for AI-generated questions")
            ai_count = st.slider("Number of questions to generate", 1, 10, 5)
            ai_difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
        
        # Save/Create button
        button_label = "Update Test" if editing else "Create Test"
        save_test = st.form_submit_button(button_label)
        
        if save_test:
            if not test_title:
                st.error("Please enter a test title.")
            elif not questions and not use_ai:
                st.error("Please add at least one question or use AI generation.")
            else:
                # Create test object
                test_data = {
                    'id': test_id,
                    'name': test_title,
                    'description': test_description,
                    'course_id': selected_course['id'],
                    'teacher_id': teacher_id,
                    'duration': time_limit,
                    'max_attempts': max_attempts,
                    'passing_score': passing_score,
                    'show_answers': show_answers,
                    'points': points,
                    'due_date': due_date.isoformat() if isinstance(due_date, datetime) else due_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'created_at': datetime.now().isoformat(),
                    'questions': questions,
                    'status': 'draft'
                }
                
                # Generate AI questions if requested
                if use_ai and ai_topic:
                    with st.spinner("Generating questions with AI..."):
                        # This would typically call an AI service
                        # For now, create some sample questions
                        ai_generated = [
                            {
                                'text': f"Sample AI question {i+1} about {ai_topic}",
                                'type': 'Multiple Choice',
                                'points': 10,
                                'options': [f'Option A for {ai_topic}', f'Option B for {ai_topic}', 
                                           f'Option C for {ai_topic}', f'Option D for {ai_topic}'],
                                'correct': 0
                            } for i in range(ai_count)
                        ]
                        test_data['questions'].extend(ai_generated)
                
                # Save to database
                if editing:
                    # Update existing test
                    tests_data = [test if test.get('id') != test_id else test_data for test in tests_data]
                    success_message = f"Test '{test_title}' updated successfully!"
                else:
                    # Add new test
                    tests_data.append(test_data)
                    success_message = f"Test '{test_title}' created successfully!"
                
                save_data(tests_data, 'tests')
                
                # Log the action
                log_audit(
                    teacher_id,
                    'create' if not editing else 'update',
                    'test',
                    test_id,
                    True,
                    f"{'Created' if not editing else 'Updated'} test: {test_title}"
                )
                
                st.success(success_message)
                
                # Clear editing state
                if 'editing_test' in st.session_state:
                    del st.session_state.editing_test
                
                # Refresh the page
                st.rerun()
    
    # Display existing tests
    st.subheader("Your Tests")
    
    # Filter tests for this course and teacher
    course_tests = [test for test in tests_data if test.get('course_id') == selected_course['id'] and test.get('teacher_id') == teacher_id]
    
    if not course_tests:
        st.info("You haven't created any tests for this course yet.")
    else:
        for test in course_tests:
            with st.expander(f"{test['name']} - Due: {test.get('due_date', 'No due date')} - Status: {test.get('status', 'Draft')}"):
                st.write(f"**Description:** {test.get('description', 'No description')}")
                st.write(f"**Duration:** {test.get('duration', 0)} minutes")
                st.write(f"**Total Points:** {test.get('points', 0)}")
                
                # Show questions count
                questions = test.get('questions', [])
                st.write(f"**Questions:** {len(questions)}")
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("Edit", key=f"edit_{test['id']}"):
                        st.session_state.editing_test = test
                        st.rerun()
                
                with col2:
                    status = "Unpublish" if test.get('status') == 'published' else "Publish"
                    if st.button(status, key=f"publish_{test['id']}"):
                        # Update test status
                        new_status = 'draft' if test.get('status') == 'published' else 'published'
                        test['status'] = new_status
                        updated_tests = [t if t.get('id') != test.get('id') else test for t in tests_data]
                        save_data(updated_tests, 'tests')
                        st.success(f"Test '{test['name']}' {status.lower()}ed!")
                        st.rerun()
                
                with col3:
                    if st.button("Results", key=f"results_{test['id']}"):
                        # Show test results (would link to results page)
                        st.session_state.viewing_test_results = test['id']
                        st.rerun()
                
                with col4:
                    if st.button("Delete", key=f"delete_{test['id']}"):
                        # Remove test
                        updated_tests = [t for t in tests_data if t.get('id') != test.get('id')]
                        save_data(updated_tests, 'tests')
                        
                        # Log action
                        log_audit(
                            teacher_id,
                            'delete',
                            'test',
                            test['id'],
                            True,
                            f"Deleted test: {test['name']}"
                        )
                        
                        st.success(f"Test '{test['name']}' deleted.")
                        st.rerun()


def show_test_creator():
    """Display the test creation interface for teachers."""
    st.header("Create Test")
    
    # Get current teacher and course
    teacher_id = st.session_state.current_user['id']
    selected_course = st.session_state.get('selected_course')

    if not selected_course:
        st.error("Please select a course first.")
        return

    # Load existing tests data
    try:
        tests_data = load_data('tests')
    except:
        tests_data = []

    # Generate a unique test ID if creating a new test
    if 'editing_test' not in st.session_state:
        test_id = f"test_{int(time.time())}_{teacher_id}"
    else:
        test_id = st.session_state.editing_test.get('id')

    # Check if we're editing an existing test
    editing = 'editing_test' in st.session_state
    current_test = st.session_state.get('editing_test', {})

    # Create test form
    with st.form("create_test_form"):
        test_title = st.text_input(
            "Test Title", value=current_test.get(
                'name', ''))
        test_description = st.text_area(
            "Description", value=current_test.get(
                'description', ''))

        # Test settings
        st.subheader("Test Settings")
        col1, col2 = st.columns(2)

        with col1:
            time_limit = st.number_input(
                "Time Limit (minutes)",
                min_value=5,
                value=current_test.get(
                    'duration',
                    60))
            max_attempts = st.number_input(
                "Maximum Attempts",
                min_value=1,
                value=current_test.get(
                    'max_attempts',
                    1))
            due_date = st.date_input(
                "Due Date",
                value=datetime.strptime(
                    current_test.get(
                        'due_date',
                        datetime.now().isoformat()),
                    "%Y-%m-%dT%H:%M:%S") if 'due_date' in current_test else None)

        with col2:
            passing_score = st.slider(
                "Passing Score (%)",
                min_value=50,
                max_value=100,
                value=current_test.get(
                    'passing_score',
                    70))
            show_answers = st.checkbox(
                "Show Answers After Completion",
                value=current_test.get(
                    'show_answers',
                    False))
            points = st.number_input(
                "Total Points",
                min_value=1,
                value=current_test.get(
                    'points',
                    100))

        # Questions section
        st.subheader("Questions")

        # Extract existing questions if editing
        questions = current_test.get('questions', [])

        # Add questions interface (simplified for brevity)
        question_types = [
            "Multiple Choice",
            "True/False",
            "Short Answer",
            "Essay",
            "Fill in the Blank"]

        # Display existing questions
        for i, question in enumerate(questions):
            with st.expander(f"Question {i + 1}: {question.get('text', '')[0:50]}..."):
                st.text_input(
                    f"Question Text {
                        i + 1}",
                    value=question.get(
                        'text',
                        ''),
                    key=f"q_text_{i}")

                q_type = st.selectbox(f"Question Type {i + 1}",
                                      options=question_types,
                                      index=question_types.index(
                                          question.get('type', 'Multiple Choice')),
                                      key=f"q_type_{i}")

                q_points = st.number_input(
                    f"Points {
                        i + 1}",
                    min_value=1,
                    value=question.get(
                        'points',
                        10),
                    key=f"q_points_{i}")

                # Show options for multiple choice
                if q_type == "Multiple Choice":
                    options = question.get('options', ['', '', '', ''])
                    correct = question.get('correct', 0)

                    st.write("Options (mark correct answer):")
                    for j, option in enumerate(options):
                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            is_correct = st.checkbox("", value=(
                                j == correct), key=f"q_{i}_correct_{j}")
                            if is_correct:
                                correct = j
                        with col2:
                            options[j] = st.text_input(
                                f"Option {j + 1}", value=option, key=f"q_{i}_opt_{j}")

                # Update question in the list
                questions[i] = {
                    'text': st.session_state.get(f"q_text_{i}"),
                    'type': st.session_state.get(f"q_type_{i}"),
                    'points': st.session_state.get(f"q_points_{i}"),
                    'options': options if q_type == "Multiple Choice" else None,
                    'correct': correct if q_type == "Multiple Choice" else None
                }
                
        # Option to use AI for question generation
        use_ai = st.checkbox("Use AI to Generate Questions")
        if use_ai:
            ai_topic = st.text_input("Topic for AI-generated questions")
            ai_count = st.slider("Number of questions to generate", 1, 10, 5)
            ai_difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
        
        # Save/Create button
        button_label = "Update Test" if editing else "Create Test"
        save_test = st.form_submit_button(button_label)
        
        if save_test:
            if not test_title:
                st.error("Please enter a test title.")
            elif not questions and not use_ai:
                st.error("Please add at least one question or use AI generation.")
        else:
                # Create test object
                test_data = {
                    'id': test_id,
                    'name': test_title,
                    'description': test_description,
                    'course_id': selected_course['id'],
                    'teacher_id': teacher_id,
                    'duration': time_limit,
                    'max_attempts': max_attempts,
                    'passing_score': passing_score,
                    'show_answers': show_answers,
                    'points': points,
                    'due_date': due_date.isoformat() if isinstance(due_date, datetime) else due_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'created_at': datetime.now().isoformat(),
                    'questions': questions,
                    'status': 'draft'
                }
                
                # Generate AI questions if requested
                if use_ai and ai_topic:
                    with st.spinner("Generating questions with AI..."):
                        # This would typically call an AI service
                        # For now, create some sample questions
                        ai_generated = [
                            {
                                'text': f"Sample AI question {i+1} about {ai_topic}",
                                'type': 'Multiple Choice',
                                'points': 10,
                                'options': [f'Option A for {ai_topic}', f'Option B for {ai_topic}', 
                                           f'Option C for {ai_topic}', f'Option D for {ai_topic}'],
                                'correct': 0
                            } for i in range(ai_count)
                        ]
                        test_data['questions'].extend(ai_generated)
                
                # Save to database
                if editing:
                    # Update existing test
                    tests_data = [test if test.get('id') != test_id else test_data for test in tests_data]
                    success_message = f"Test '{test_title}' updated successfully!"
                else:
                    # Add new test
                    tests_data.append(test_data)
                    success_message = f"Test '{test_title}' created successfully!"
                
                save_data(tests_data, 'tests')
                
                # Log the action
                log_audit(
                    teacher_id,
                    'create' if not editing else 'update',
                    'test',
                    test_id,
                    True,
                    f"{'Created' if not editing else 'Updated'} test: {test_title}"
                )
                
                st.success(success_message)
                
                # Clear editing state
                if 'editing_test' in st.session_state:
                    del st.session_state.editing_test
                
                # Refresh the page
                st.rerun()
    
    # Display existing tests
    st.subheader("Your Tests")
    
    # Filter tests for this course and teacher
    course_tests = [test for test in tests_data if test.get('course_id') == selected_course['id'] and test.get('teacher_id') == teacher_id]
    
    if not course_tests:
        st.info("You haven't created any tests for this course yet.")
    else:
        for test in course_tests:
            with st.expander(f"{test['name']} - Due: {test.get('due_date', 'No due date')} - Status: {test.get('status', 'Draft')}"):
                st.write(f"**Description:** {test.get('description', 'No description')}")
                st.write(f"**Duration:** {test.get('duration', 0)} minutes")
                st.write(f"**Total Points:** {test.get('points', 0)}")
                
                # Show questions count
                questions = test.get('questions', [])
                st.write(f"**Questions:** {len(questions)}")
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("Edit", key=f"edit_{test['id']}"):
                        st.session_state.editing_test = test
                        st.rerun()
                
                with col2:
                    status = "Unpublish" if test.get('status') == 'published' else "Publish"
                    if st.button(status, key=f"publish_{test['id']}"):
                        # Update test status
                        new_status = 'draft' if test.get('status') == 'published' else 'published'
                        test['status'] = new_status
                        updated_tests = [t if t.get('id') != test.get('id') else test for t in tests_data]
                        save_data(updated_tests, 'tests')
                        st.success(f"Test '{test['name']}' {status.lower()}ed!")
                        st.rerun()
                
                with col3:
                    if st.button("Results", key=f"results_{test['id']}"):
                        # Show test results (would link to results page)
                        st.session_state.viewing_test_results = test['id']
                        st.rerun()
                
                with col4:
                    if st.button("Delete", key=f"delete_{test['id']}"):
                        # Remove test
                        updated_tests = [t for t in tests_data if t.get('id') != test.get('id')]
                        save_data(updated_tests, 'tests')
                        
                        # Log action
                        log_audit(
                            teacher_id,
                            'delete',
                            'test',
                            test['id'],
                            True,
                            f"Deleted test: {test['name']}"
                        )
                        
                        st.success(f"Test '{test['name']}' deleted.")
                        st.rerun()


def show_dashboard():
    st.title(f"Welcome, {st.session_state.current_user['name']}!")

    # Display different content based on user role
    if st.session_state.current_user['role'] == 'teacher':
        show_teacher_dashboard()
    else:
        show_student_dashboard()


def show_student_dashboard():
    """Display the student dashboard."""
    st.subheader("Your Courses")
    
    # Get student's courses
    student_id = st.session_state.current_user['id']
    courses = get_student_courses(student_id)
    
    if not courses:
        st.info("You are not enrolled in any courses yet.")
    else:
        # Display courses in a grid
        cols = st.columns(3)
        for i, course in enumerate(courses):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(course['name'])
                    st.caption(f"Code: {course['code']}")
                    st.write(course['description'][:100] + "..." if len(course['description']) > 100 else course['description'])
                    
                    # View course button
                    if st.button("View Course", key=f"view_course_{course['id']}"):
                        st.session_state.selected_course_id = course['id']
                        st.session_state.current_page = 'course_detail'
                        st.rerun()
    
    # Recent assignments section
    st.subheader("Recent Assignments")
    
    # Get student's submissions
    submissions = get_student_submissions(student_id)
    
    if not submissions:
        st.info("No recent assignment submissions.")
    else:
        # Sort by submission date, most recent first
        submissions.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
        
        # Show recent submissions
        for submission in submissions[:5]:  # Show only 5 most recent
            assignment = get_assignment_by_id(submission['assignment_id'])
            course = get_course_by_id(assignment['course_id']) if assignment else None
            
            if assignment and course:
                with st.container(border=True):
                    cols = st.columns([3, 2, 1, 1])
                    with cols[0]:
                        st.write(f"**{assignment['title']}**")
                        st.caption(f"Course: {course['name']}")
                    with cols[1]:
                        st.write("Submitted: " + submission.get('submitted_at', 'N/A')[:10])
                    with cols[2]:
                        if 'score' in submission:
                            st.write(f"Score: {submission['score']}/{assignment['points']}")
                        else:
                            st.write("Not graded")
                    with cols[3]:
                        if st.button("View", key=f"view_sub_{submission['id']}"):
                            st.session_state.selected_submission_id = submission['id']
                            st.session_state.current_page = 'submission_detail'
                            st.rerun()


def show_dashboard():
    st.title(f"Welcome, {st.session_state.current_user['name']}!")

    # Display different content based on user role
    if st.session_state.current_user['role'] == 'teacher':
        show_teacher_dashboard()
    else:
        show_student_dashboard()

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
                                    st.success(f"Lesson plan '{lesson_title}' saved successfully!")
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")

        
        with tools_tab2:

            st.subheader("Assessment Tools")
            
            # Assessment creation tools
            st.write("Create and manage assessments for your courses.")
            
            # Assessment types
            assessment_type = st.selectbox("Select Assessment Type", ["Multiple Choice", "True/False", "Short Answer", "Essay"])
            
            # Assessment creation form
            with st.form("assessment_creation_form"):

                st.write("### Create New Assessment")
                
                assessment_name = st.text_input("Assessment Name")
                assessment_description = st.text_area("Description")
                assessment_duration = st.number_input("Duration (minutes)", min_value=10, max_value=120, value=30, step=5)
                
                # Assessment questions
                st.write("### Assessment Questions")
                
                num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)
                
                questions = []
                for i in range(num_questions):
                    question_text = st.text_area(f"Question {i + 1}")
                    if assessment_type == "Multiple Choice":
                        options = st.text_area(f"Options for Question {i + 1} (separate with commas)")
                        correct_answer = st.text_input(f"Correct Answer for Question {i + 1}")
                    elif assessment_type == "True/False":
                        correct_answer = st.radio(f"Correct Answer for Question {i + 1}", ["True", "False"])
                    elif assessment_type == "Short Answer":
                        correct_answer = st.text_input(f"Correct Answer for Question {i + 1}")
                    elif assessment_type == "Essay":
                        correct_answer = st.text_area(f"Example Answer for Question {i + 1}")
                    
                    questions.append({
                        "text": question_text,
                        "type": assessment_type,
                        "options": options.split(",") if assessment_type == "Multiple Choice" else None,
                        "correct_answer": correct_answer
                    })
                
from edumate.services.gemini_service import GeminiService
from edumate.services.feedback_service import FeedbackService
from edumate.services.plagiarism_service import PlagiarismService
from edumate.services.grading_service import GradingService
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
from edumate.utils.teacher_tools import TeacherTools
from edumate.utils.classroom_manager import ClassroomManager
from edumate.utils.exam_manager import ExamManager
from edumate.utils.indian_education import IndianEducationSystem
from edumate.utils.career_planner import CareerPlanner
from edumate.utils.audit import AuditTrail
from edumate.utils.analytics import Analytics
from edumate.utils.encryption import Encryptor
from edumate.utils.logger import log_system_event, log_access, log_error, log_audit
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
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

# Import modular page components
from edumate.pages.test_creator import show_test_creator
from edumate.pages.ai_tutor import show_ai_tutor_page

# Set matplotlib style for better looking charts
plt.style.use('seaborn-v0_8')

# Import utilities

# Import services

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
study_recommendations_service = StudyRecommendationsService(
    gemini_service=gemini_service, data_dir='data')
group_formation_service = GroupFormationService()
learning_path_service = LearningPathService()
multilingual_feedback_service = MultilingualFeedbackService(
    gemini_service=gemini_service, data_dir='data')
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
                {"title": "Python for Everybody", "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/python"},
                {"title": "The Complete Web Developer Course", "platform": "Udemy",
                    "url": "https://www.udemy.com/course/the-complete-web-developer-course-2"}
            ],
            "data_analysis": [
                {"title": "Data Science Specialization", "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/jhu-data-science"},
                {"title": "Data Analyst Nanodegree", "platform": "Udacity",
                    "url": "https://www.udacity.com/course/data-analyst-nanodegree--nd002"}
            ],
            "communication": [
                {"title": "Effective Communication", "platform": "LinkedIn Learning",
                    "url": "https://www.linkedin.com/learning/topics/communication"},
                {"title": "Public Speaking", "platform": "Coursera",
                    "url": "https://www.coursera.org/learn/public-speaking"}
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


def create_assignment(
    title,
    description,
    course_id,
    teacher_id,
    due_date,
     points=100):
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
    return [
    assignment for assignment in assignments if assignment['course_id'] == course_id]


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
    assignment_submissions = [
    sub for sub in submissions if sub['assignment_id'] == assignment_id]

    if assignment_submissions:
        # If there are submissions, we should handle them
        # Option 1: Delete all submissions (implemented here)
        # Option 2: Prevent deletion if submissions exist (alternative
        # approach)

        # Delete all related submissions and their files
        for submission in assignment_submissions:
            # Delete any attached files
            if submission.get(
                'file_info') and submission['file_info'].get('file_path'):
                file_path = submission['file_info']['file_path']
                if os.path.exists(file_path):
                    try:
                        # removing real file instead of placeholder
                        os.remove(file_path)
                    except Exception as e:
                        st.error(f"Failed to delete file {file_path}: {e}")

        # Remove all submissions for this assignment
        submissions = [
    sub for sub in submissions if sub['assignment_id'] != assignment_id]
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
    if any(sub['assignment_id'] == assignment_id and sub['student_id']
           == student_id for sub in submissions):
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


def analyze_with_gemini(
    content_type,
    file_path,
    prompt,
    mime_type,
     model="gemini-2.0-pro"):
    """Analyze content using specific Gemini model"""
    if not GEMINI_API_KEY:
        st.error(
            "Gemini API key not found. Please add GEMINI_API_KEY to your .env file.")
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
                        return part.get('text', 'No text found')

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

                image_filename = f"page{
    page_num +
    1}_img{
        img_index +
         1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)

                with open(image_path, "wb") as image_file:

                    image_file.write(image_bytes)

                image_paths.append(image_path)

        # If no images found, try extracting as whole page images
        if not image_paths:
            for page_num, page in enumerate(pdf_document):
                pix = page.get_pixmap()
                image_filename = f"page{page_num + 1}.png"
                image_path = os.path.join(output_dir, image_filename)
                pix.save(image_path)
                image_paths.append(image_path)

        return image_paths

    except ImportError as e:
        st.error(
    f"Error: {
        str(e)}. Please install PyMuPDF using: pip install PyMuPDF")
        return []
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
        return []


# Add imports for our enhanced grading services at the top of the file


def auto_grade_submission(submission_id):
    """
    Automatically grade a submission using AI.
    Returns: (success, message)
    """
    try:
        # Load submissions and get the specific submission
        submissions = load_data('submissions')
        submission = next(
            (s for s in submissions if s['id'] == submission_id), None)

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
                file_analysis = analyze_file_content(
                    file_content, submission['file_info']['filename'])
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
                    {"name": "Content", "weight": 40,
                        "description": "Completeness and quality of content"},
                    {"name": "Understanding", "weight": 30,
                        "description": "Demonstrates understanding of concepts"},
                    {"name": "Presentation", "weight": 20,
                        "description": "Organization and clarity"},
                    {"name": "Technical", "weight": 10,
                        "description": "Technical correctness"}
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
                # Max 25% reduction
                reduction = min(
                    grading_result["score"] * (plagiarism_result["score"] / 200), 25)
                grading_result["score"] -= reduction
                grading_result["feedback"] += f"\n\n⚠️ **Plagiarism Warning**: This submission shows {plagiarism_result['score']:.1f}% similarity with existing sources. Points have been deducted accordingly."

        except Exception as e:
            log_error(f"Error checking plagiarism: {str(e)}")

        # Generate AI feedback (comprehensive version from service)
        ai_feedback = grading_result.get("feedback", "")
        if not ai_feedback:
            # Fallback to legacy feedback generator if service doesn't provide feedback
            ai_feedback = generate_ai_feedback(
                submission, file_content, file_analysis)

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
    strengths_match = re.search(
        r'(?:STRENGTHS|Strengths):(.*?)(?:\n\n|\n[A-Z]|$)',
        feedback,
        re.DOTALL | re.IGNORECASE)
    if strengths_match:
        strengths_text = strengths_match.group(1)
        strength_points = re.findall(
            r'[-*â€¢]\s*(.*?)(?:\n[-*â€¢]|\n\n|$)',
            strengths_text,
            re.DOTALL)
        for point in strength_points:
            if point.strip():
                points.append({"type": "strength", "text": point.strip()})

    # Extract weaknesses/areas for improvement section
    weaknesses_match = re.search(r'(?:WEAKNESSES|AREAS FOR IMPROVEMENT|Areas for improvement|Weaknesses):(.*?)(?:\n\n|\n[A-Z]|$)',
                                 feedback, re.DOTALL | re.IGNORECASE)
    if weaknesses_match:
        weaknesses_text = weaknesses_match.group(1)
        weakness_points = re.findall(
            r'[-*â€¢]\s*(.*?)(?:\n[-*â€¢]|\n\n|$)',
            weaknesses_text,
            re.DOTALL)
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
        analysis += f"- Average sentence length: {
            avg_sentence_length:.1f} words\n"

    # File type specific analysis
    if file_ext == '.pdf':
        analysis += "- Document appears to be a PDF, which typically indicates a formal submission.\n"
    elif file_ext == '.docx':
        analysis += "- Document is in Word format, which is appropriate for academic submissions.\n"
    elif file_ext == '.txt':
        analysis += "- Document is in plain text format. Consider using a more formatted document type for future submissions.\n"

    return analysis


def generate_ai_feedback(
    submission,
    file_content="",
    file_analysis="",
    gemini_analysis=""):
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
        improvements.append(
            "Your text submission is quite brief. Consider expanding your answer with more details.")

    else:
        strengths.append(
            "You provided a detailed text response with good length.")

    if "because" in content.lower() or "therefore" in content.lower():
        strengths.append(
            "Good use of reasoning and logical connections in your answer.")

    else:
        improvements.append(
            "Try to include more reasoning and logical connections in your answer.")

    if len(content.split('.')) > 5:
        strengths.append(
            "Good structure with multiple sentences in your text submission.")

    else:
        improvements.append(
            "Consider structuring your text answer with more complete sentences.")

    # File submission analysis
    if submission.get('file_info'):
        file_info = submission.get('file_info')
        strengths.append(
            f"You submitted a file ({
                file_info['filename']}) which demonstrates thoroughness.")

        # Add file-specific feedback
        if file_info['file_type'].endswith('pdf'):
            strengths.append(
                "Your PDF submission is in a professional format suitable for academic work.")

            # If we have Gemini analysis, it means it was a handwritten or
            # complex PDF
            if gemini_analysis:
                strengths.append(
                    "Your PDF was analyzed directly by our AI system, including any handwritten content.")

                # Check if the analysis mentions handwritten content
                if "handwritten" in gemini_analysis.lower(
                ) or "handwriting" in gemini_analysis.lower():
                    strengths.append(
                        "Your handwritten work shows dedication and personal effort in your submission.")
        elif file_info['file_type'].endswith('docx'):
            strengths.append(
                "Your Word document submission follows standard academic formatting.")
        elif file_info['file_type'].endswith('txt'):
            improvements.append(
                "Consider using a more formatted document type (like PDF or DOCX) for future submissions.")

    else:
        improvements.append(
            "Consider attaching a file with your submission for more comprehensive work.")

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
        return False
    else:
        return True


def get_assignment_submissions(assignment_id):
    try:
        submissions = load_data('submissions')
        return [sub for sub in submissions if sub['assignment_id'] == assignment_id]
    except Exception as e:
        st.error(f"Error loading submissions: {e}")
        return []


def get_student_submissions(student_id):
    try:
        submissions = load_data('submissions')
        return [sub for sub in submissions if sub['student_id'] == student_id]
    except Exception as e:
        st.error(f"Error loading submissions: {e}")
        return []

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
                st.error("Please fill in all fields.")
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


def show_student_dashboard():
    """Display the student dashboard."""
    st.subheader("Your Courses")
    
    # Get student's courses
    student_id = st.session_state.current_user['id']
    courses = get_student_courses(student_id)
    
    if not courses:
        st.info("You are not enrolled in any courses yet.")
    else:
        # Display courses in a grid
        cols = st.columns(3)
        for i, course in enumerate(courses):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(course['name'])
                    st.caption(f"Code: {course['code']}")
                    st.write(course['description'][:100] + "..." if len(course['description']) > 100 else course['description'])
                    
                    # View course button
                    if st.button("View Course", key=f"view_course_{course['id']}"):
                        st.session_state.selected_course_id = course['id']
                        st.session_state.current_page = 'course_detail'
                        st.rerun()
                    
    # Recent assignments section
    st.subheader("Recent Assignments")
    
    # Get student's submissions
    submissions = get_student_submissions(student_id)
    
    if not submissions:
        st.info("No recent assignment submissions.")
    else:
        # Sort by submission date, most recent first
        submissions.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
        
        # Show recent submissions
        for submission in submissions[:5]:  # Show only 5 most recent
            assignment = get_assignment_by_id(submission['assignment_id'])
            course = get_course_by_id(assignment['course_id']) if assignment else None
            
            if assignment and course:
                with st.container(border=True):
                    cols = st.columns([3, 2, 1, 1])
                    with cols[0]:
                        st.write(f"**{assignment['title']}**")
                        st.caption(f"Course: {course['name']}")
                    with cols[1]:
                        st.write("Submitted: " + submission.get('submitted_at', 'N/A')[:10])
                    with cols[2]:
                        if 'score' in submission:
                            st.write(f"Score: {submission['score']}/{assignment['points']}")
                        else:
                            st.write("Not graded")
                    with cols[3]:
                        if st.button("View", key=f"view_sub_{submission['id']}"):
                            st.session_state.selected_submission_id = submission['id']
                            st.session_state.current_page = 'submission_detail'
                        st.rerun()
                

def show_test_creator():
    """Display the test creation interface for teachers."""
    st.header("Create Test")
    
    # Get current teacher and course
    teacher_id = st.session_state.current_user['id']
    selected_course = st.session_state.get('selected_course')

    if not selected_course:
        st.error("Please select a course first.")
        return

    # Load existing tests data
    try:
        tests_data = load_data('tests')
    except:
        tests_data = []

    # Generate a unique test ID if creating a new test
    if 'editing_test' not in st.session_state:
        test_id = f"test_{int(time.time())}_{teacher_id}"
    else:
        test_id = st.session_state.editing_test.get('id')

    # Check if we're editing an existing test
    editing = 'editing_test' in st.session_state
    current_test = st.session_state.get('editing_test', {})

    # Create test form
    with st.form("create_test_form"):
        test_title = st.text_input(
            "Test Title", value=current_test.get(
                'name', ''))
        test_description = st.text_area(
            "Description", value=current_test.get(
                'description', ''))

        # Test settings
        st.subheader("Test Settings")
            col1, col2 = st.columns(2)

            with col1:
            time_limit = st.number_input(
                "Time Limit (minutes)",
                min_value=5,
                value=current_test.get(
                    'duration',
                    60))
            max_attempts = st.number_input(
                "Maximum Attempts",
                min_value=1,
                value=current_test.get(
                    'max_attempts',
                    1))
            due_date = st.date_input(
                "Due Date",
                value=datetime.strptime(
                    current_test.get(
                        'due_date',
                        datetime.now().isoformat()),
                    "%Y-%m-%dT%H:%M:%S") if 'due_date' in current_test else None)

        with col2:
            passing_score = st.slider(
                "Passing Score (%)",
                min_value=50,
                max_value=100,
                value=current_test.get(
                    'passing_score',
                    70))
            show_answers = st.checkbox(
                "Show Answers After Completion",
                value=current_test.get(
                    'show_answers',
                    False))
            points = st.number_input(
                "Total Points",
                min_value=1,
                value=current_test.get(
                    'points',
                    100))

        # Questions section
        st.subheader("Questions")

        # Extract existing questions if editing
        questions = current_test.get('questions', [])

        # Add questions interface (simplified for brevity)
        question_types = [
            "Multiple Choice",
            "True/False",
            "Short Answer",
            "Essay",
            "Fill in the Blank"]

        # Display existing questions
        for i, question in enumerate(questions):
            with st.expander(f"Question {i + 1}: {question.get('text', '')[0:50]}..."):
                st.text_input(
                    f"Question Text {
                        i + 1}",
                    value=question.get(
                        'text',
                        ''),
                    key=f"q_text_{i}")

                q_type = st.selectbox(f"Question Type {i + 1}",
                                      options=question_types,
                                      index=question_types.index(
                                          question.get('type', 'Multiple Choice')),
                                      key=f"q_type_{i}")

                q_points = st.number_input(
                    f"Points {
                        i + 1}",
                    min_value=1,
                    value=question.get(
                        'points',
                        10),
                    key=f"q_points_{i}")

                # Show options for multiple choice
                if q_type == "Multiple Choice":
                    options = question.get('options', ['', '', '', ''])
                    correct = question.get('correct', 0)

                    st.write("Options (mark correct answer):")
                    for j, option in enumerate(options):
                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            is_correct = st.checkbox("", value=(
                                j == correct), key=f"q_{i}_correct_{j}")
                            if is_correct:
                                correct = j
            with col2:
                            options[j] = st.text_input(
                                f"Option {j + 1}", value=option, key=f"q_{i}_opt_{j}")

                # Update question in the list
                questions[i] = {
                    'text': st.session_state.get(f"q_text_{i}"),
                    'type': st.session_state.get(f"q_type_{i}"),
                    'points': st.session_state.get(f"q_points_{i}"),
                    'options': options if q_type == "Multiple Choice" else None,
                    'correct': correct if q_type == "Multiple Choice" else None
                }
                
        # Option to use AI for question generation
        use_ai = st.checkbox("Use AI to Generate Questions")
        if use_ai:
            ai_topic = st.text_input("Topic for AI-generated questions")
            ai_count = st.slider("Number of questions to generate", 1, 10, 5)
            ai_difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
        
        # Save/Create button
        button_label = "Update Test" if editing else "Create Test"
        save_test = st.form_submit_button(button_label)
        
        if save_test:
            if not test_title:
                st.error("Please enter a test title.")
            elif not questions and not use_ai:
                st.error("Please add at least one question or use AI generation.")
                else:
                # Create test object
                test_data = {
                    'id': test_id,
                    'name': test_title,
                    'description': test_description,
                    'course_id': selected_course['id'],
                    'teacher_id': teacher_id,
                    'duration': time_limit,
                    'max_attempts': max_attempts,
                    'passing_score': passing_score,
                    'show_answers': show_answers,
                    'points': points,
                    'due_date': due_date.isoformat() if isinstance(due_date, datetime) else due_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'created_at': datetime.now().isoformat(),
                    'questions': questions,
                    'status': 'draft'
                }
                
                # Generate AI questions if requested
                if use_ai and ai_topic:
                    with st.spinner("Generating questions with AI..."):
                        # This would typically call an AI service
                        # For now, create some sample questions
                        ai_generated = [
                            {
                                'text': f"Sample AI question {i+1} about {ai_topic}",
                                'type': 'Multiple Choice',
                                'points': 10,
                                'options': [f'Option A for {ai_topic}', f'Option B for {ai_topic}', 
                                           f'Option C for {ai_topic}', f'Option D for {ai_topic}'],
                                'correct': 0
                            } for i in range(ai_count)
                        ]
                        test_data['questions'].extend(ai_generated)
                
                # Save to database
                if editing:
                    # Update existing test
                    tests_data = [test if test.get('id') != test_id else test_data for test in tests_data]
                    success_message = f"Test '{test_title}' updated successfully!"
                else:
                    # Add new test
                    tests_data.append(test_data)
                    success_message = f"Test '{test_title}' created successfully!"
                
                save_data(tests_data, 'tests')
                
                # Log the action
                log_audit(
                    teacher_id,
                    'create' if not editing else 'update',
                    'test',
                    test_id,
                    True,
                    f"{'Created' if not editing else 'Updated'} test: {test_title}"
                )
                
                st.success(success_message)
                
                # Clear editing state
                if 'editing_test' in st.session_state:
                    del st.session_state.editing_test
                
                # Refresh the page
                st.rerun()
    
    # Display existing tests
    st.subheader("Your Tests")
    
    # Filter tests for this course and teacher
    course_tests = [test for test in tests_data if test.get('course_id') == selected_course['id'] and test.get('teacher_id') == teacher_id]
    
    if not course_tests:
        st.info("You haven't created any tests for this course yet.")
    else:
        for test in course_tests:
            with st.expander(f"{test['name']} - Due: {test.get('due_date', 'No due date')} - Status: {test.get('status', 'Draft')}"):
                st.write(f"**Description:** {test.get('description', 'No description')}")
                st.write(f"**Duration:** {test.get('duration', 0)} minutes")
                st.write(f"**Total Points:** {test.get('points', 0)}")
                
                # Show questions count
                questions = test.get('questions', [])
                st.write(f"**Questions:** {len(questions)}")
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("Edit", key=f"edit_{test['id']}"):
                        st.session_state.editing_test = test
                        st.rerun()
                
                with col2:
                    status = "Unpublish" if test.get('status') == 'published' else "Publish"
                    if st.button(status, key=f"publish_{test['id']}"):
                        # Update test status
                        new_status = 'draft' if test.get('status') == 'published' else 'published'
                        test['status'] = new_status
                        updated_tests = [t if t.get('id') != test.get('id') else test for t in tests_data]
                        save_data(updated_tests, 'tests')
                        st.success(f"Test '{test['name']}' {status.lower()}ed!")
                        st.rerun()
                
                with col3:
                    if st.button("Results", key=f"results_{test['id']}"):
                        # Show test results (would link to results page)
                        st.session_state.viewing_test_results = test['id']
                        st.rerun()
                
                with col4:
                    if st.button("Delete", key=f"delete_{test['id']}"):
                        # Remove test
                        updated_tests = [t for t in tests_data if t.get('id') != test.get('id')]
                        save_data(updated_tests, 'tests')
                        
                        # Log action
                        log_audit(
                            teacher_id,
                            'delete',
                            'test',
                            test['id'],
                            True,
                            f"Deleted test: {test['name']}"
                        )
                        
                        st.success(f"Test '{test['name']}' deleted.")
                        st.rerun()


def show_test_creator():
    """Display the test creation interface for teachers."""
    st.header("Create Test")
    
    # Get current teacher and course
    teacher_id = st.session_state.current_user['id']
    selected_course = st.session_state.get('selected_course')

    if not selected_course:
        st.error("Please select a course first.")
        return

    # Load existing tests data
    try:
        tests_data = load_data('tests')
    except:
        tests_data = []

    # Generate a unique test ID if creating a new test
    if 'editing_test' not in st.session_state:
        test_id = f"test_{int(time.time())}_{teacher_id}"
    else:
        test_id = st.session_state.editing_test.get('id')

    # Check if we're editing an existing test
    editing = 'editing_test' in st.session_state
    current_test = st.session_state.get('editing_test', {})

    # Create test form
    with st.form("create_test_form"):
        test_title = st.text_input(
            "Test Title", value=current_test.get(
                'name', ''))
        test_description = st.text_area(
            "Description", value=current_test.get(
                'description', ''))

        # Test settings
        st.subheader("Test Settings")
        col1, col2 = st.columns(2)

        with col1:
            time_limit = st.number_input(
                "Time Limit (minutes)",
                min_value=5,
                value=current_test.get(
                    'duration',
                    60))
            max_attempts = st.number_input(
                "Maximum Attempts",
                min_value=1,
                value=current_test.get(
                    'max_attempts',
                    1))
            due_date = st.date_input(
                "Due Date",
                value=datetime.strptime(
                    current_test.get(
                        'due_date',
                        datetime.now().isoformat()),
                    "%Y-%m-%dT%H:%M:%S") if 'due_date' in current_test else None)

        with col2:
            passing_score = st.slider(
                "Passing Score (%)",
                min_value=50,
                max_value=100,
                value=current_test.get(
                    'passing_score',
                    70))
            show_answers = st.checkbox(
                "Show Answers After Completion",
                value=current_test.get(
                    'show_answers',
                    False))
            points = st.number_input(
                "Total Points",
                min_value=1,
                value=current_test.get(
                    'points',
                    100))

        # Questions section
        st.subheader("Questions")

        # Extract existing questions if editing
        questions = current_test.get('questions', [])

        # Add questions interface (simplified for brevity)
        question_types = [
            "Multiple Choice",
            "True/False",
            "Short Answer",
            "Essay",
            "Fill in the Blank"]

        # Display existing questions
        for i, question in enumerate(questions):
            with st.expander(f"Question {i + 1}: {question.get('text', '')[0:50]}..."):
                st.text_input(
                    f"Question Text {
                        i + 1}",
                    value=question.get(
                        'text',
                        ''),
                    key=f"q_text_{i}")

                q_type = st.selectbox(f"Question Type {i + 1}",
                                      options=question_types,
                                      index=question_types.index(
                                          question.get('type', 'Multiple Choice')),
                                      key=f"q_type_{i}")

                q_points = st.number_input(
                    f"Points {
                        i + 1}",
                    min_value=1,
                    value=question.get(
                        'points',
                        10),
                    key=f"q_points_{i}")

                # Show options for multiple choice
                if q_type == "Multiple Choice":
                    options = question.get('options', ['', '', '', ''])
                    correct = question.get('correct', 0)

                    st.write("Options (mark correct answer):")
                    for j, option in enumerate(options):
                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            is_correct = st.checkbox("", value=(
                                j == correct), key=f"q_{i}_correct_{j}")
                            if is_correct:
                                correct = j
                        with col2:
                            options[j] = st.text_input(
                                f"Option {j + 1}", value=option, key=f"q_{i}_opt_{j}")

                # Update question in the list
                questions[i] = {
                    'text': st.session_state.get(f"q_text_{i}"),
                    'type': st.session_state.get(f"q_type_{i}"),
                    'points': st.session_state.get(f"q_points_{i}"),
                    'options': options if q_type == "Multiple Choice" else None,
                    'correct': correct if q_type == "Multiple Choice" else None
                }
                
        # Option to use AI for question generation
        use_ai = st.checkbox("Use AI to Generate Questions")
        if use_ai:
            ai_topic = st.text_input("Topic for AI-generated questions")
            ai_count = st.slider("Number of questions to generate", 1, 10, 5)
            ai_difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
        
        # Save/Create button
        button_label = "Update Test" if editing else "Create Test"
        save_test = st.form_submit_button(button_label)
        
        if save_test:
            if not test_title:
                st.error("Please enter a test title.")
            elif not questions and not use_ai:
                st.error("Please add at least one question or use AI generation.")
            else:
                # Create test object
                test_data = {
                    'id': test_id,
                    'name': test_title,
                    'description': test_description,
                    'course_id': selected_course['id'],
                    'teacher_id': teacher_id,
                    'duration': time_limit,
                    'max_attempts': max_attempts,
                    'passing_score': passing_score,
                    'show_answers': show_answers,
                    'points': points,
                    'due_date': due_date.isoformat() if isinstance(due_date, datetime) else due_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'created_at': datetime.now().isoformat(),
                    'questions': questions,
                    'status': 'draft'
                }
                
                # Generate AI questions if requested
                if use_ai and ai_topic:
                    with st.spinner("Generating questions with AI..."):
                        # This would typically call an AI service
                        # For now, create some sample questions
                        ai_generated = [
                            {
                                'text': f"Sample AI question {i+1} about {ai_topic}",
                                'type': 'Multiple Choice',
                                'points': 10,
                                'options': [f'Option A for {ai_topic}', f'Option B for {ai_topic}', 
                                           f'Option C for {ai_topic}', f'Option D for {ai_topic}'],
                                'correct': 0
                            } for i in range(ai_count)
                        ]
                        test_data['questions'].extend(ai_generated)
                
                # Save to database
                if editing:
                    # Update existing test
                    tests_data = [test if test.get('id') != test_id else test_data for test in tests_data]
                    success_message = f"Test '{test_title}' updated successfully!"
                else:
                    # Add new test
                    tests_data.append(test_data)
                    success_message = f"Test '{test_title}' created successfully!"
                
                save_data(tests_data, 'tests')
                
                        # Log the action
                        log_audit(
                            teacher_id,
                    'create' if not editing else 'update',
                    'test',
                    test_id,
                            True,
                    f"{'Created' if not editing else 'Updated'} test: {test_title}"
                )
                
                st.success(success_message)
                
                # Clear editing state
                if 'editing_test' in st.session_state:
                    del st.session_state.editing_test
                
                        # Refresh the page
                        st.rerun()
    
    # Display existing tests
    st.subheader("Your Tests")
    
    # Filter tests for this course and teacher
    course_tests = [test for test in tests_data if test.get('course_id') == selected_course['id'] and test.get('teacher_id') == teacher_id]
    
    if not course_tests:
        st.info("You haven't created any tests for this course yet.")
                    else:
        for test in course_tests:
            with st.expander(f"{test['name']} - Due: {test.get('due_date', 'No due date')} - Status: {test.get('status', 'Draft')}"):
                st.write(f"**Description:** {test.get('description', 'No description')}")
                st.write(f"**Duration:** {test.get('duration', 0)} minutes")
                st.write(f"**Total Points:** {test.get('points', 0)}")
                
                # Show questions count
                questions = test.get('questions', [])
                st.write(f"**Questions:** {len(questions)}")
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("Edit", key=f"edit_{test['id']}"):
                        st.session_state.editing_test = test
                        st.rerun()
                
                with col2:
                    status = "Unpublish" if test.get('status') == 'published' else "Publish"
                    if st.button(status, key=f"publish_{test['id']}"):
                        # Update test status
                        new_status = 'draft' if test.get('status') == 'published' else 'published'
                        test['status'] = new_status
                        updated_tests = [t if t.get('id') != test.get('id') else test for t in tests_data]
                        save_data(updated_tests, 'tests')
                        st.success(f"Test '{test['name']}' {status.lower()}ed!")
                        st.rerun()
                
                with col3:
                    if st.button("Results", key=f"results_{test['id']}"):
                        # Show test results (would link to results page)
                        st.session_state.viewing_test_results = test['id']
                        st.rerun()
                
                with col4:
                    if st.button("Delete", key=f"delete_{test['id']}"):
                        # Remove test
                        updated_tests = [t for t in tests_data if t.get('id') != test.get('id')]
                        save_data(updated_tests, 'tests')
                        
                        # Log action
                        log_audit(
                            teacher_id,
                            'delete',
                            'test',
                            test['id'],
                            True,
                            f"Deleted test: {test['name']}"
                        )
                        
                        st.success(f"Test '{test['name']}' deleted.")
                        st.rerun()


def show_dashboard():
    st.title(f"Welcome, {st.session_state.current_user['name']}!")

    # Display different content based on user role
    if st.session_state.current_user['role'] == 'teacher':
        show_teacher_dashboard()
    else:
        show_student_dashboard()


def show_student_dashboard():
    """Display the student dashboard."""
    st.subheader("Your Courses")
    
    # Get student's courses
    student_id = st.session_state.current_user['id']
    courses = get_student_courses(student_id)
    
        if not courses:
        st.info("You are not enrolled in any courses yet.")
        else:
        # Display courses in a grid
        cols = st.columns(3)
        for i, course in enumerate(courses):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(course['name'])
                    st.caption(f"Code: {course['code']}")
                    st.write(course['description'][:100] + "..." if len(course['description']) > 100 else course['description'])
                    
                    # View course button
                    if st.button("View Course", key=f"view_course_{course['id']}"):
                        st.session_state.selected_course_id = course['id']
                        st.session_state.current_page = 'course_detail'
                        st.rerun()
    
    # Recent assignments section
    st.subheader("Recent Assignments")
    
    # Get student's submissions
    submissions = get_student_submissions(student_id)
    
    if not submissions:
        st.info("No recent assignment submissions.")
                else:
        # Sort by submission date, most recent first
        submissions.sort(key=lambda x: x.get('submitted_at', ''), reverse=True)
        
        # Show recent submissions
        for submission in submissions[:5]:  # Show only 5 most recent
            assignment = get_assignment_by_id(submission['assignment_id'])
            course = get_course_by_id(assignment['course_id']) if assignment else None
            
            if assignment and course:
                with st.container(border=True):
                    cols = st.columns([3, 2, 1, 1])
                    with cols[0]:
                        st.write(f"**{assignment['title']}**")
                        st.caption(f"Course: {course['name']}")
                    with cols[1]:
                        st.write("Submitted: " + submission.get('submitted_at', 'N/A')[:10])
                    with cols[2]:
                        if 'score' in submission:
                            st.write(f"Score: {submission['score']}/{assignment['points']}")
                        else:
                            st.write("Not graded")
                    with cols[3]:
                        if st.button("View", key=f"view_sub_{submission['id']}"):
                            st.session_state.selected_submission_id = submission['id']
                            st.session_state.current_page = 'submission_detail'
                            st.rerun()


def show_dashboard():
    st.title(f"Welcome, {st.session_state.current_user['name']}!")

    # Display different content based on user role
    if st.session_state.current_user['role'] == 'teacher':
        show_teacher_dashboard()
    else:
        show_student_dashboard()

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
                                    st.success(f"Lesson plan '{lesson_title}' saved successfully!")
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")

        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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
                                    st.success(f"Lesson plan '{lesson_title}' saved successfully!")
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")

        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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
                                    st.success(f"Lesson plan '{lesson_title}' saved successfully!")
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")

        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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
from edumate.services.gemini_service import GeminiService
from edumate.services.feedback_service import FeedbackService
from edumate.services.plagiarism_service import PlagiarismService
from edumate.services.grading_service import GradingService
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
from edumate.utils.teacher_tools import TeacherTools
from edumate.utils.classroom_manager import ClassroomManager
from edumate.utils.exam_manager import ExamManager
from edumate.utils.indian_education import IndianEducationSystem
from edumate.utils.career_planner import CareerPlanner
from edumate.utils.audit import AuditTrail
from edumate.utils.analytics import Analytics
from edumate.utils.encryption import Encryptor
from edumate.utils.logger import log_system_event, log_access, log_error, log_audit
from dotenv import load_dotenv
from typing import List, Dict, Any, Union
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
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

# Import modular page components
from edumate.pages.test_creator import show_test_creator
from edumate.pages.ai_tutor import show_ai_tutor_page

# Set matplotlib style for better looking charts
plt.style.use('seaborn-v0_8')

# Import utilities

# Import services

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
study_recommendations_service = StudyRecommendationsService(
    gemini_service=gemini_service, data_dir='data')
group_formation_service = GroupFormationService()
learning_path_service = LearningPathService()
multilingual_feedback_service = MultilingualFeedbackService(
    gemini_service=gemini_service, data_dir='data')
teacher_analytics_service = TeacherAnalyticsService()

# Set page configuration
st.set_page_config(
    page_title="EduMate - AI-Powered Education Platform",
    page_icon="ðŸŽ",
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
                {"title": "Python for Everybody", "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/python"},
                {"title": "The Complete Web Developer Course", "platform": "Udemy",
                    "url": "https://www.udemy.com/course/the-complete-web-developer-course-2"}
            ],
            "data_analysis": [
                {"title": "Data Science Specialization", "platform": "Coursera",
                    "url": "https://www.coursera.org/specializations/jhu-data-science"},
                {"title": "Data Analyst Nanodegree", "platform": "Udacity",
                    "url": "https://www.udacity.com/course/data-analyst-nanodegree--nd002"}
            ],
            "communication": [
                {"title": "Effective Communication", "platform": "LinkedIn Learning",
                    "url": "https://www.linkedin.com/learning/topics/communication"},
                {"title": "Public Speaking", "platform": "Coursera",
                    "url": "https://www.coursera.org/learn/public-speaking"}
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


def create_assignment(
    title,
    description,
    course_id,
    teacher_id,
    due_date,
     points=100):
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
    return [
    assignment for assignment in assignments if assignment['course_id'] == course_id]


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
    assignment_submissions = [
    sub for sub in submissions if sub['assignment_id'] == assignment_id]

    if assignment_submissions:
        # If there are submissions, we should handle them
        # Option 1: Delete all submissions (implemented here)
        # Option 2: Prevent deletion if submissions exist (alternative
        # approach)

        # Delete all related submissions and their files
        for submission in assignment_submissions:
            # Delete any attached files
            if submission.get(
                'file_info') and submission['file_info'].get('file_path'):
                file_path = submission['file_info']['file_path']
                if os.path.exists(file_path):
                    try:
                        # removing real file instead of placeholder
                        os.remove(file_path)
                    except Exception as e:
                        st.error(f"Failed to delete file {file_path}: {e}")

        # Remove all submissions for this assignment
        submissions = [
    sub for sub in submissions if sub['assignment_id'] != assignment_id]
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
    if any(sub['assignment_id'] == assignment_id and sub['student_id']
           == student_id for sub in submissions):
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


def analyze_with_gemini(
    content_type,
    file_path,
    prompt,
    mime_type,
     model="gemini-2.0-pro"):
    """Analyze content using specific Gemini model"""
    if not GEMINI_API_KEY:
        st.error(
            "Gemini API key not found. Please add GEMINI_API_KEY to your .env file.")
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
                        return part.get('text', 'No text found')

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

                image_filename = f"page{
    page_num +
    1}_img{
        img_index +
         1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)

                with open(image_path, "wb") as image_file:

                    image_file.write(image_bytes)

                image_paths.append(image_path)

        # If no images found, try extracting as whole page images
        if not image_paths:
            for page_num, page in enumerate(pdf_document):
                pix = page.get_pixmap()
                image_filename = f"page{page_num + 1}.png"
                image_path = os.path.join(output_dir, image_filename)
                pix.save(image_path)
                image_paths.append(image_path)

        return image_paths

    except ImportError as e:
        st.error(
    f"Error: {
        str(e)}. Please install PyMuPDF using: pip install PyMuPDF")
        return []
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
        return []


# Add imports for our enhanced grading services at the top of the file

'''
def auto_grade_submission(submission_id):
    """
    Automatically grade a submission using AI.
    Returns: (success, message)
    """
    try:
        # Load submissions and get the specific submission
        submissions = load_data('submissions')
        submission = next(
    (s for s in submissions if s['id'] == submission_id), None)

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

            
except Exception as e:
    print(f"Error: {e}")

except Exception as e:
    print(f"Error: {e}")
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
                file_analysis = analyze_file_content(
    file_content, submission['file_info']['filename'])
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
                    {"name": "Content", "weight": 40,
                        "description": "Completeness and quality of content"},
                    {"name": "Understanding", "weight": 30,
                        "description": "Demonstrates understanding of concepts"},
                    {"name": "Presentation", "weight": 20,
                        "description": "Organization and clarity"},
                    {"name": "Technical", "weight": 10,
                        "description": "Technical correctness"}
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
                # Max 25% reduction
                reduction = min(
                    grading_result["score"] * (plagiarism_result["score"] / 200), 25)
                grading_result["score"] -= reduction
                grading_result["feedback"] += f"\n\nâš ï¸ **Plagiarism Warning**: This submission shows {
    plagiarism_result['score']:.1f}% similarity with existing sources. Points have been deducted accordingly."

        except Exception as e:
            log_error(f"Error checking plagiarism: {str(e)}")

        # Generate AI feedback (comprehensive version from service)
        ai_feedback = grading_result.get("feedback", "")
        if not ai_feedback:
            # Fallback to legacy feedback generator if service doesn't provide
            # feedback
            ai_feedback = generate_ai_feedback(
    submission, file_content, file_analysis)

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

        return True, f"Submission auto-graded successfully with score: {
    submission['score']}"

    except Exception as e:
        error_message = f"Error auto-grading submission: {str(e)}"
        log_error(error_message)
        return False, error_message

'''
def extract_feedback_points(feedback):
    """Extract key feedback points from grading feedback."""
    points = []

    # Extract strengths section
    strengths_match = re.search(
    r'(?:STRENGTHS|Strengths):(.*?)(?:\n\n|\n[A-Z]|$)',
    feedback,
     re.DOTALL | re.IGNORECASE)
    if strengths_match:
        strengths_text = strengths_match.group(1)
        strength_points = re.findall(
    r'[-*â€¢]\s*(.*?)(?:\n[-*â€¢]|\n\n|$)',
    strengths_text,
     re.DOTALL)
        for point in strength_points:
            if point.strip():
                points.append({"type": "strength", "text": point.strip()})

    # Extract weaknesses/areas for improvement section
    weaknesses_match = re.search(r'(?:WEAKNESSES|AREAS FOR IMPROVEMENT|Areas for improvement|Weaknesses):(.*?)(?:\n\n|\n[A-Z]|$)',
                                feedback, re.DOTALL | re.IGNORECASE)
    if weaknesses_match:
        weaknesses_text = weaknesses_match.group(1)
        weakness_points = re.findall(
    r'[-*â€¢]\s*(.*?)(?:\n[-*â€¢]|\n\n|$)',
    weaknesses_text,
     re.DOTALL)
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
        analysis += f"- Average sentence length: {
    avg_sentence_length:.1f} words\n"

    # File type specific analysis
    if file_ext == '.pdf':
        analysis += "- Document appears to be a PDF, which typically indicates a formal submission.\n"
    elif file_ext == '.docx':
        analysis += "- Document is in Word format, which is appropriate for academic submissions.\n"
    elif file_ext == '.txt':
        analysis += "- Document is in plain text format. Consider using a more formatted document type for future submissions.\n"

    return analysis


def generate_ai_feedback(
    submission,
    file_content="",
    file_analysis="",
     gemini_analysis=""):
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
        improvements.append(
            "Your text submission is quite brief. Consider expanding your answer with more details.")

    else:
        strengths.append(
            "You provided a detailed text response with good length.")

    if "because" in content.lower() or "therefore" in content.lower():
        strengths.append(
            "Good use of reasoning and logical connections in your answer.")

    else:
        improvements.append(
            "Try to include more reasoning and logical connections in your answer.")

    if len(content.split('.')) > 5:
        strengths.append(
            "Good structure with multiple sentences in your text submission.")

    else:
        improvements.append(
            "Consider structuring your text answer with more complete sentences.")

    # File submission analysis
    if submission.get('file_info'):
        file_info = submission.get('file_info')
        strengths.append(
    f"You submitted a file ({
        file_info['filename']}) which demonstrates thoroughness.")

        # Add file-specific feedback
        if file_info['file_type'].endswith('pdf'):
            strengths.append(
                "Your PDF submission is in a professional format suitable for academic work.")

            # If we have Gemini analysis, it means it was a handwritten or
            # complex PDF
            if gemini_analysis:
                strengths.append(
                    "Your PDF was analyzed directly by our AI system, including any handwritten content.")

                # Check if the analysis mentions handwritten content
                if "handwritten" in gemini_analysis.lower(
                ) or "handwriting" in gemini_analysis.lower():
                    strengths.append(
                        "Your handwritten work shows dedication and personal effort in your submission.")
        elif file_info['file_type'].endswith('docx'):
            strengths.append(
                "Your Word document submission follows standard academic formatting.")
        elif file_info['file_type'].endswith('txt'):
            improvements.append(
                "Consider using a more formatted document type (like PDF or DOCX) for future submissions.")

    else:
        improvements.append(
            "Consider attaching a file with your submission for more comprehensive work.")

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
        return False
    else:
        return True


def get_assignment_submissions(assignment_id):
    try:
        submissions = load_data('submissions')
        return [sub for sub in submissions if sub['assignment_id'] == assignment_id]
    except Exception as e:
        st.error(f"Error loading submissions: {e}")
        return []


def get_student_submissions(student_id):
    try:
        submissions = load_data('submissions')
        return [sub for sub in submissions if sub['student_id'] == student_id]
    except Exception as e:
        st.error(f"Error loading submissions: {e}")
        return []

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
                st.error("Please fill in all fields.")
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


def show_test_creator():
    """Display the test creation interface for teachers."""
    st.header("Create Test")

    # Get current teacher and course
    teacher_id = st.session_state.current_user['id']
    selected_course = st.session_state.get('selected_course')

    if not selected_course:
        st.error("Please select a course first.")
        return

        # Load existing tests data
    try:
        tests_data = load_data('tests')
    except Exception as e:
        print(f"Error: {e}")
        # Create tests.json if it doesn't exist
        tests_data = []
        save_data(tests_data, 'tests')

    # Generate a unique test ID if creating a new test
    if 'editing_test' not in st.session_state:
        test_id = f"test_{int(time.time())}_{teacher_id}"
    else:
        test_id = st.session_state.editing_test.get('id')

    # Check if we're editing an existing test
    editing = 'editing_test' in st.session_state
    current_test = st.session_state.get('editing_test', {})

    # Create test form
    with st.form("create_test_form"):
        test_title = st.text_input(
    "Test Title", value=current_test.get(
        'name', ''))
        test_description = st.text_area(
    "Description", value=current_test.get(
        'description', ''))

        # Test settings
        st.subheader("Test Settings")
        col1, col2 = st.columns(2)

        with col1:
            time_limit = st.number_input(
    "Time Limit (minutes)",
    min_value=5,
    value=current_test.get(
        'duration',
         60))
            max_attempts = st.number_input(
    "Maximum Attempts",
    min_value=1,
    value=current_test.get(
        'max_attempts',
         1))
            due_date = st.date_input(
    "Due Date",
    value=datetime.strptime(
        current_test.get(
            'due_date',
            datetime.now().isoformat()),
             "%Y-%m-%dT%H:%M:%S") if 'due_date' in current_test else None)

        with col2:
            passing_score = st.slider(
    "Passing Score (%)",
    min_value=50,
    max_value=100,
    value=current_test.get(
        'passing_score',
         70))
            show_answers = st.checkbox(
    "Show Answers After Completion",
    value=current_test.get(
        'show_answers',
         False))
            points = st.number_input(
    "Total Points",
    min_value=1,
    value=current_test.get(
        'points',
         100))

        # Questions section
        st.subheader("Questions")

        # Extract existing questions if editing
        questions = current_test.get('questions', [])

        # Add questions interface (simplified for brevity)
        question_types = [
    "Multiple Choice",
    "True/False",
    "Short Answer",
    "Essay",
     "Fill in the Blank"]

        # Display existing questions
        for i, question in enumerate(questions):
            with st.expander(f"Question {i + 1}: {question.get('text', '')[0:50]}..."):
                st.text_input(
    f"Question Text {
        i + 1}",
        value=question.get(
            'text',
            ''),
             key=f"q_text_{i}")

                q_type = st.selectbox(f"Question Type {i + 1}",
                                     options=question_types,
                                     index=question_types.index(
                                         question.get('type', 'Multiple Choice')),
                                     key=f"q_type_{i}")

                q_points = st.number_input(
    f"Points {
        i + 1}",
        min_value=1,
        value=question.get(
            'points',
            10),
             key=f"q_points_{i}")

                # Show options for multiple choice
                if q_type == "Multiple Choice":
                    options = question.get('options', ['', '', '', ''])
                    correct = question.get('correct', 0)

                    st.write("Options (mark correct answer):")
                    for j, option in enumerate(options):
                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            is_correct = st.checkbox("", value=(
                                j == correct), key=f"q_{i}_correct_{j}")
                            if is_correct:
                                correct = j
        with col2:
                            options[j] = st.text_input(
                                f"Option {j + 1}", value=option, key=f"q_{i}_opt_{j}")

                # Update question in the list
                questions[i] = {
                    'text': st.session_state.get(f"q_text_{i}"),
                    'type': st.session_state.get(f"q_type_{i}"),
                    'points': st.session_state.get(f"q_points_{i}"),
                    'options': options if q_type == "Multiple Choice" else None,
                    'correct': correct if q_type == "Multiple Choice" else None
                }
                
        
        # Option to use AI for question generation
        use_ai = st.checkbox("Use AI to Generate Questions")
        if use_ai:
            ai_topic = st.text_input("Topic for AI-generated questions")
            ai_count = st.slider("Number of questions to generate", 1, 10, 5)
            ai_difficulty = st.select_slider("Difficulty", options=["Easy", "Medium", "Hard"])
        
        # Save/Create button
        button_label = "Update Test" if editing else "Create Test"
        save_test = st.form_submit_button(button_label)
        
        if save_test:
            if not test_title:
                st.error("Please enter a test title.")
            elif not questions and not use_ai:
                st.error("Please add at least one question or use AI generation.")
            else:
                # Create test object
                test_data = {
                    'id': test_id,
                    'name': test_title,
                    'description': test_description,
                    'course_id': selected_course['id'],
                    'teacher_id': teacher_id,
                    'duration': time_limit,
                    'max_attempts': max_attempts,
                    'passing_score': passing_score,
                    'show_answers': show_answers,
                    'points': points,
                    'due_date': due_date.isoformat() if isinstance(due_date, datetime) else due_date.strftime('%Y-%m-%dT%H:%M:%S'),
                    'created_at': datetime.now().isoformat(),
                    'questions': questions,
                    'status': 'draft'
                }
                
                # Generate AI questions if requested
                if use_ai and ai_topic:
                    with st.spinner("Generating questions with AI..."):
                        # This would typically call an AI service
                        # For now, create some sample questions
                        ai_generated = [
                            {
                                'text': f"Sample AI question {i+1} about {ai_topic}",
                                'type': 'Multiple Choice',
                                'points': 10,
                                'options': [f'Option A for {ai_topic}', f'Option B for {ai_topic}', 
                                           f'Option C for {ai_topic}', f'Option D for {ai_topic}'],
                                'correct': 0
                            } for i in range(ai_count)
                        ]
                        test_data['questions'].extend(ai_generated)
                
                # Save to database
                if editing:
                    # Update existing test
                    tests_data = [test if test.get('id') != test_id else test_data for test in tests_data]
                    success_message = f"Test '{test_title}' updated successfully!"
                else:
                    # Add new test
                    tests_data.append(test_data)
                    success_message = f"Test '{test_title}' created successfully!"
                
                save_data(tests_data, 'tests')
                
                # Log the action
                log_audit(
                    teacher_id,
                    'create' if not editing else 'update',
                    'test',
                    test_id,
                    True,
                    f"{'Created' if not editing else 'Updated'} test: {test_title}"
                )
                
                st.success(success_message)
                
                # Clear editing state
                if 'editing_test' in st.session_state:
                    del st.session_state.editing_test
                
                # Refresh the page
                st.rerun()
    
    # Display existing tests
    st.subheader("Your Tests")
    
    # Filter tests for this course and teacher
    course_tests = [test for test in tests_data if test.get('course_id') == selected_course['id'] and test.get('teacher_id') == teacher_id]
    
    if not course_tests:
        st.info("You haven't created any tests for this course yet.")
    else:
        for test in course_tests:
            with st.expander(f"{test['name']} - Due: {test.get('due_date', 'No due date')} - Status: {test.get('status', 'Draft')}"):
                st.write(f"**Description:** {test.get('description', 'No description')}")
                st.write(f"**Duration:** {test.get('duration', 0)} minutes")
                st.write(f"**Total Points:** {test.get('points', 0)}")
                
                # Show questions count
                questions = test.get('questions', [])
                st.write(f"**Questions:** {len(questions)}")
                
                # Action buttons
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("Edit", key=f"edit_{test['id']}"):
                        st.session_state.editing_test = test
                        st.rerun()
                
                with col2:
                    status = "Unpublish" if test.get('status') == 'published' else "Publish"
                    if st.button(status, key=f"publish_{test['id']}"):
                        # Update test status
                        new_status = 'draft' if test.get('status') == 'published' else 'published'
                        test['status'] = new_status
                        updated_tests = [t if t.get('id') != test.get('id') else test for t in tests_data]
                        save_data(updated_tests, 'tests')
                        st.success(f"Test '{test['name']}' {status.lower()}ed!")
                        st.rerun()
                
                with col3:
                    if st.button("Results", key=f"results_{test['id']}"):
                        # Show test results (would link to results page)
                        st.session_state.viewing_test_results = test['id']
                        st.rerun()
                
                with col4:
                    if st.button("Delete", key=f"delete_{test['id']}"):
                        # Remove test
                        updated_tests = [t for t in tests_data if t.get('id') != test.get('id')]
                        save_data(updated_tests, 'tests')
                        
                        # Log action
                        log_audit(
                            teacher_id,
                            'delete',
                            'test',
                            test['id'],
                            True,
                            f"Deleted test: {test['name']}"
                        )
                        
                        st.success(f"Test '{test['name']}' deleted.")
                        st.rerun()

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
                                    st.success(f"Lesson plan '{lesson_title}' saved successfully!")
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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

                        

                        # Display generated lesson plan
                        st.success("Lesson plan generated successfully!")
        
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
                    with st.spinner("Generating lesson plan..."):
                        # Simulate AI processing
                        import time
                        time.sleep(2)
                        st.success("Lesson plan generated successfully!")
        if st.session_state.current_page == 'home':
            show_home_page()
        elif st.session_state.current_page == 'dashboard':
            show_dashboard()
        elif st.session_state.current_page == 'courses':
            show_courses()
        elif st.session_state.current_page == 'assignment_creator':
            show_create_assignment()
        elif st.session_state.current_page == 'test_creator':
            show_test_creator()
        elif st.session_state.current_page == 'grade_book':
            show_grade_book()
        elif st.session_state.current_page == 'analytics':
            show_analytics()
        elif st.session_state.current_page == 'ai_assistant':
            show_ai_assistant()
        elif st.session_state.current_page == 'ai_tutor':
            show_ai_tutor_page()
        elif st.session_state.current_page == 'account_settings':
            show_account_settings()
        elif st.session_state.current_page == 'course_detail' and 'selected_course_id' in st.session_state:
            show_course_detail(st.session_state.selected_course_id)
        elif st.session_state.current_page == 'assignment_detail' and 'selected_assignment_id' in st.session_state:
            show_assignment_detail(st.session_state.selected_assignment_id)
        elif st.session_state.current_page == 'submission_detail' and 'selected_submission_id' in st.session_state:
            show_submission_detail(st.session_state.selected_submission_id)
        elif st.session_state.current_page == 'career_planning':
            show_career_planning()
        elif st.session_state.current_page == 'system_settings':
            show_system_settings()
        elif st.session_state.current_page == 'help':
            show_help_and_support()
        else:
            # Default to dashboard if page not found
            show_dashboard()

if __name__ == "__main__":
    main()