"""
Quiz Generator Module
Contains utilities for generating educational quizzes based on various parameters
"""
import time
import random
import streamlit as st

def generate_quiz(quiz_subject, quiz_topic, quiz_level, quiz_questions, question_types):
    """
    Generate a quiz based on specified parameters.
    
    Args:
        quiz_subject: The subject of the quiz
        quiz_topic: The topic within the subject
        quiz_level: Difficulty level of the quiz
        quiz_questions: Number of questions to generate
        question_types: Types of questions to include
        
    Returns:
        Formatted quiz content as a string
    """
    # Create quiz content
    quiz_content = f"# {quiz_topic} Quiz\n\nSubject: {quiz_subject}\nDifficulty: {quiz_level}\nTotal Questions: {quiz_questions}\n\n"
    
    question_count = 1
    
    # Add sample questions based on types
    if "Multiple Choice" in question_types:
        quiz_content += generate_multiple_choice_questions(quiz_subject, quiz_topic, quiz_level, min(2, quiz_questions), question_count)
        question_count += 2
    
    if "True/False" in question_types:
        quiz_content += generate_true_false_questions(quiz_subject, quiz_topic, quiz_level, min(2, quiz_questions), question_count)
        question_count += 2
    
    if "Short Answer" in question_types:
        quiz_content += generate_short_answer_questions(quiz_subject, quiz_topic, quiz_level, min(2, quiz_questions), question_count)
        question_count += 2
        
    if "Essay" in question_types:
        quiz_content += generate_essay_questions(quiz_subject, quiz_topic, quiz_level, min(1, quiz_questions), question_count)
        question_count += 1

    return quiz_content

def generate_multiple_choice_questions(subject, topic, level, count, start_num):
    """Generate multiple choice questions"""
    content = ""
    for i in range(count):
        content += f"\n## Question {start_num + i}\n"
        content += f"What is a key concept related to {topic} in {subject}?\n\n"
        content += "A. First possible answer\n"
        content += "B. Second possible answer\n"
        content += "C. Third possible answer\n"
        content += "D. Fourth possible answer\n\n"
        content += "**Answer:** C\n\n"
    return content

def generate_true_false_questions(subject, topic, level, count, start_num):
    """Generate true/false questions"""
    content = ""
    for i in range(count):
        content += f"\n## Question {start_num + i}\n"
        content += f"True or False: {topic} is a fundamental concept in {subject}.\n\n"
        content += "**Answer:** True\n\n"
    return content

def generate_short_answer_questions(subject, topic, level, count, start_num):
    """Generate short answer questions"""
    content = ""
    for i in range(count):
        content += f"\n## Question {start_num + i}\n"
        content += f"Briefly explain how {topic} is applied in {subject}.\n\n"
        content += "**Sample Answer:** A short explanation would be provided here...\n\n"
    return content

def generate_essay_questions(subject, topic, level, count, start_num):
    """Generate essay questions"""
    content = ""
    for i in range(count):
        content += f"\n## Question {start_num + i}\n"
        content += f"Write a comprehensive essay about the importance of {topic} in {subject}. Include historical context, current applications, and future implications.\n\n"
        content += "**Grading Rubric:**\n"
        content += "- Content Understanding: 40%\n"
        content += "- Critical Analysis: 30%\n"
        content += "- Organization: 20%\n"
        content += "- Writing Mechanics: 10%\n\n"
    return content

def get_resource_suggestions(ai_subject, ai_topic):
    """
    Get resource suggestions based on subject and topic.
    
    Args:
        ai_subject: Subject area
        ai_topic: Specific topic
        
    Returns:
        Dictionary containing resources, online courses, and next topics
    """
    resources = []
    online_courses = []
    next_topics = []
    
    # Generate relevant resources based on subject and topic
    if ai_subject.lower() in ["math", "mathematics"]:
        resources = [
            {"name": f"Khan Academy: {ai_topic}", "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={ai_topic.replace(' ', '+')}"},
            {"name": f"Desmos Activities for {ai_topic}", "url": f"https://teacher.desmos.com/search?q={ai_topic.replace(' ', '+')}"},
            {"name": "NCTM Illuminations", "url": "https://illuminations.nctm.org/"},
            {"name": f"GeoGebra Materials for {ai_topic}", "url": f"https://www.geogebra.org/search/{ai_topic.replace(' ', '%20')}"},
        ]
        online_courses = [
            {"name": f"Coursera - {ai_topic} Courses", "url": f"https://www.coursera.org/search?query={ai_topic.replace(' ', '%20')}"},
            {"name": f"Khan Academy - {ai_topic}", "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={ai_topic.replace(' ', '+')}"},
        ]
        next_topics = [
            f"Advanced applications of {ai_topic}",
            f"Historical development of {ai_topic}",
            f"{ai_topic} in real-world contexts"
        ]
    elif ai_subject.lower() in ["science", "biology", "chemistry", "physics"]:
        resources = [
            {"name": f"PhET Interactive Simulations for {ai_topic}", "url": f"https://phet.colorado.edu/en/simulations/filter?sort=alpha&view=grid&q={ai_topic.replace(' ', '+')}"}
        ]
        online_courses = [
            {"name": f"EdX - {ai_topic} Courses", "url": f"https://www.edx.org/search?q={ai_topic.replace(' ', '%20')}"},
            {"name": f"Coursera - {ai_topic} Courses", "url": f"https://www.coursera.org/search?query={ai_topic.replace(' ', '%20')}"},
        ]
        next_topics = [
            f"Current research in {ai_topic}",
            f"Ethical considerations related to {ai_topic}",
            f"Technological applications of {ai_topic}"
        ]
    elif ai_subject.lower() in ["english", "language arts", "literature"]:
        resources = [
            {"name": f"CommonLit: {ai_topic}", "url": f"https://www.commonlit.org/en/search?query={ai_topic.replace(' ', '%20')}"},
            {"name": f"Project Gutenberg: {ai_topic}", "url": f"https://www.gutenberg.org/ebooks/search/?query={ai_topic.replace(' ', '+')}"},
            {"name": "ReadWriteThink", "url": "http://www.readwritethink.org/"},
            {"name": f"Poetry Foundation: {ai_topic}", "url": f"https://www.poetryfoundation.org/search?query={ai_topic.replace(' ', '%20')}"},
        ]
        next_topics = [
            f"Comparative analysis of {ai_topic}",
            f"Critical perspectives on {ai_topic}",
            f"Creative projects inspired by {ai_topic}"
        ]
    else:
        resources = [
            {"name": f"Google Scholar: {ai_topic}", "url": f"https://scholar.google.com/scholar?q={ai_topic.replace(' ', '+')}"},
            {"name": f"Khan Academy: {ai_topic}", "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={ai_topic.replace(' ', '+')}"},
        ]
        next_topics = [
            f"Advanced study of {ai_topic}",
            f"Interdisciplinary connections to {ai_topic}",
            f"Project-based learning with {ai_topic}"
        ]
        
    return {
        "resources": resources,
        "online_courses": online_courses,
        "next_topics": next_topics
    } 