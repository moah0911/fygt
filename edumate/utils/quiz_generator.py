"""
Quiz Generator Module
Contains utilities for generating educational quizzes based on various parameters
"""
import time
import random
import streamlit as st
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from edumate.integration import analyze_with_gemini

def generate_quiz(quiz_subject, quiz_topic, quiz_level, quiz_questions, question_types):
    """
    Generate a quiz based on specified parameters using AI.
    
    Args:
        quiz_subject: The subject of the quiz
        quiz_topic: The topic within the subject
        quiz_level: Difficulty level of the quiz
        quiz_questions: Number of questions to generate
        question_types: Types of questions to include
        
    Returns:
        Formatted quiz content as a string
    """
    # Initialize quiz content with header
    quiz_content = f"# {quiz_topic} Quiz\n\nSubject: {quiz_subject}\nDifficulty: {quiz_level}\nTotal Questions: {quiz_questions}\n\n"
    
    # Prepare questions using AI
    questions_per_type = max(1, quiz_questions // len(question_types))
    remaining_questions = quiz_questions - (questions_per_type * len(question_types))
    
    # If analyze_with_gemini is available (imported from original app)
    if analyze_with_gemini:
        # Generate each type of question using AI
        question_count = 1
        for q_type in question_types:
            # Determine how many questions of this type to generate
            num_questions = questions_per_type
            if remaining_questions > 0:
                num_questions += 1
                remaining_questions -= 1
                
            # Generate questions using Gemini
            type_content = generate_questions_with_ai(
                quiz_subject, 
                quiz_topic, 
                quiz_level, 
                q_type, 
                num_questions, 
                question_count
            )
            
            # Add to quiz content
            quiz_content += type_content
            question_count += num_questions
    else:
        # Fallback to static generation if AI is not available
        question_count = 1
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

def generate_questions_with_ai(subject, topic, level, q_type, count, start_num):
    """
    Generate questions of a specific type using Gemini AI.
    
    Args:
        subject: The subject area
        topic: The specific topic
        level: Difficulty level
        q_type: Type of questions to generate
        count: Number of questions to generate
        start_num: Starting question number
    
    Returns:
        Formatted questions as string
    """
    prompt = f"""
    Generate {count} {q_type} questions about {topic} in {subject} at a {level} difficulty level.
    
    Format the output as follows:
    
    For Multiple Choice questions:
    ## Question [number]
    [Question text]
    
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    
    **Answer:** [Correct letter]
    
    For True/False questions:
    ## Question [number]
    [Question text]
    
    **Answer:** [True or False]
    
    For Short Answer questions:
    ## Question [number]
    [Question text]
    
    **Sample Answer:** [Brief answer]
    
    For Essay questions:
    ## Question [number]
    [Essay prompt]
    
    **Grading Rubric:**
    - Content Understanding: 40%
    - Critical Analysis: 30%
    - Organization: 20%
    - Writing Mechanics: 10%
    
    Start numbering from {start_num}.
    Ensure all questions are factually accurate and educational.
    """
    
    try:
        ai_response = analyze_with_gemini(
            content_type="text",
            file_path=None,
            prompt=prompt,
            mime_type="text/plain"
        )
        
        if ai_response and 'result' in ai_response:
            return "\n" + ai_response['result']
        else:
            # Fallback to static generation if AI fails
            if q_type == "Multiple Choice":
                return generate_multiple_choice_questions(subject, topic, level, count, start_num)
            elif q_type == "True/False":
                return generate_true_false_questions(subject, topic, level, count, start_num)
            elif q_type == "Short Answer":
                return generate_short_answer_questions(subject, topic, level, count, start_num)
            elif q_type == "Essay":
                return generate_essay_questions(subject, topic, level, count, start_num)
    except Exception as e:
        st.error(f"Error generating questions with AI: {e}")
        # Fallback to static generation if AI fails
        if q_type == "Multiple Choice":
            return generate_multiple_choice_questions(subject, topic, level, count, start_num)
        elif q_type == "True/False":
            return generate_true_false_questions(subject, topic, level, count, start_num)
        elif q_type == "Short Answer":
            return generate_short_answer_questions(subject, topic, level, count, start_num)
        elif q_type == "Essay":
            return generate_essay_questions(subject, topic, level, count, start_num)

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
    Get resource suggestions based on subject and topic using AI when available.
    
    Args:
        ai_subject: Subject area
        ai_topic: Specific topic
        
    Returns:
        Dictionary containing resources, online courses, and next topics
    """
    # Try using AI to generate resources
    if analyze_with_gemini:
        try:
            prompt = f"""
            Generate educational resources and suggestions for learning more about {ai_topic} in {ai_subject}.
            
            Format the response as a JSON object with the following structure:
            {{
                "resources": [
                    {{"name": "Resource Name 1", "url": "https://example.com/resource1"}},
                    {{"name": "Resource Name 2", "url": "https://example.com/resource2"}}
                ],
                "online_courses": [
                    {{"name": "Course Name 1", "url": "https://example.com/course1"}},
                    {{"name": "Course Name 2", "url": "https://example.com/course2"}}
                ],
                "next_topics": [
                    "Next Topic 1",
                    "Next Topic 2",
                    "Next Topic 3"
                ]
            }}
            
            Ensure all URLs are valid and resources are reputable educational sources.
            Include at least 3 resources, 2 online courses, and 3 suggested next topics.
            """
            
            ai_response = analyze_with_gemini(
                content_type="text",
                file_path=None,
                prompt=prompt,
                mime_type="text/plain"
            )
            
            if ai_response and 'result' in ai_response:
                # Try to parse JSON from the AI response
                import json
                try:
                    # Extract JSON object if it's embedded in text
                    text = ai_response['result']
                    # Find JSON object in text (between curly braces)
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    if start >= 0 and end > start:
                        json_str = text[start:end]
                        resources_data = json.loads(json_str)
                        return resources_data
                except json.JSONDecodeError:
                    # Fall back to static suggestions if JSON parsing fails
                    pass
        except Exception as e:
            st.error(f"Error generating resource suggestions with AI: {e}")
    
    # Fallback to static generation if AI is not available or fails
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