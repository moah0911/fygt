"""
Lesson Plan Generator Module
Contains utilities for generating educational lesson plans based on various parameters
"""
import sys
import os
import time
import streamlit as st

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from edumate.integration import analyze_with_gemini

def generate_lesson_plan(subject, topic, grade_level, duration, include_differentiation=False, 
                        include_assessment=False, include_resources=True):
    """
    Generate a comprehensive lesson plan using AI.
    
    Args:
        subject: The subject of the lesson
        topic: The specific topic for the lesson
        grade_level: The grade level the lesson is intended for
        duration: The planned duration of the lesson
        include_differentiation: Whether to include differentiation strategies
        include_assessment: Whether to include assessment strategies
        include_resources: Whether to include relevant resources
        
    Returns:
        A dictionary with formatted lesson plan sections
    """
    # If AI generation is available
    if analyze_with_gemini:
        try:
            # Build the prompt for AI
            prompt = f"""
            Create a detailed lesson plan for a {duration} lesson on {topic} in {subject} for {grade_level} students.
            
            Structure the lesson plan with the following sections:
            
            1. Learning Objectives (3-5 clear, measurable objectives)
            2. Materials Needed (list of required materials)
            3. Preparation Steps (what the teacher needs to prepare beforehand)
            4. Introduction ({int(int(duration.split()[0]) * 0.15)} minutes - hook and activation of prior knowledge)
            5. Main Activity ({int(int(duration.split()[0]) * 0.5)} minutes - core learning activities)
            6. Group Work/Practice ({int(int(duration.split()[0]) * 0.25)} minutes - collaborative or individual practice)
            7. Reflection and Assessment ({int(int(duration.split()[0]) * 0.1)} minutes - closing activities)
            """
            
            # Add optional sections based on parameters
            if include_differentiation:
                prompt += """
                8. Differentiation Strategies
                   - For advanced students
                   - For struggling students
                   - For English language learners
                """
                
            if include_assessment:
                prompt += """
                9. Assessment Strategies
                   - Formative assessment methods
                   - Summative assessment options
                   - Rubric or scoring guide
                """
                
            if include_resources:
                prompt += """
                10. Resources and References
                    - Teacher resources
                    - Student resources
                    - Additional materials for extending learning
                """
                
            prompt += """
            Format each section with markdown headers and bullet points for clarity.
            Make the lesson engaging, standards-aligned, and grade-appropriate.
            Include specific activities, questions, and timing for each section.
            """
            
            # Call the AI to generate the lesson plan
            ai_response = analyze_with_gemini(
                content_type="text",
                file_path=None,
                prompt=prompt,
                mime_type="text/plain"
            )
            
            if ai_response and 'result' in ai_response:
                # Process the response into sections for easy display
                plan_text = ai_response['result']
                sections = parse_lesson_plan(plan_text)
                
                # Add metadata
                sections['metadata'] = {
                    'subject': subject,
                    'topic': topic,
                    'grade_level': grade_level,
                    'duration': duration,
                    'generated_with_ai': True
                }
                
                return sections
                
        except Exception as e:
            st.error(f"Error generating lesson plan with AI: {e}")
    
    # Return a fallback lesson plan if AI generation fails or is unavailable
    return generate_fallback_lesson_plan(subject, topic, grade_level, duration, 
                                        include_differentiation, include_assessment, include_resources)

def parse_lesson_plan(plan_text):
    """
    Parse the AI-generated lesson plan text into structured sections
    
    Args:
        plan_text: The raw text of the lesson plan
        
    Returns:
        Dictionary with structured section content
    """
    sections = {
        'objectives': '',
        'materials': '',
        'preparation': '',
        'introduction': '',
        'main_activity': '',
        'group_work': '',
        'reflection': '',
        'differentiation': '',
        'assessment': '',
        'resources': '',
        'full_text': plan_text  # Store the full text for download
    }
    
    # Extract sections based on common headers
    # This is a simple implementation; a more robust approach would use regex
    current_section = None
    lines = plan_text.split('\n')
    
    for line in lines:
        lower_line = line.lower()
        
        # Identify section based on keywords
        if 'learning objective' in lower_line or 'objective' in lower_line and '#' in line:
            current_section = 'objectives'
            sections[current_section] += line + '\n'
        elif 'material' in lower_line and '#' in line:
            current_section = 'materials'
            sections[current_section] += line + '\n'
        elif 'preparation' in lower_line and '#' in line:
            current_section = 'preparation'
            sections[current_section] += line + '\n'
        elif 'introduction' in lower_line and '#' in line:
            current_section = 'introduction'
            sections[current_section] += line + '\n'
        elif ('main activity' in lower_line or 'core activity' in lower_line) and '#' in line:
            current_section = 'main_activity'
            sections[current_section] += line + '\n'
        elif ('group work' in lower_line or 'practice' in lower_line) and '#' in line:
            current_section = 'group_work'
            sections[current_section] += line + '\n'
        elif ('reflection' in lower_line or 'assessment' in lower_line or 'closing' in lower_line) and '#' in line:
            current_section = 'reflection'
            sections[current_section] += line + '\n'
        elif 'differentiation' in lower_line and '#' in line:
            current_section = 'differentiation'
            sections[current_section] += line + '\n'
        elif ('assessment' in lower_line and 'reflection' not in lower_line) and '#' in line:
            current_section = 'assessment'
            sections[current_section] += line + '\n'
        elif ('resource' in lower_line or 'reference' in lower_line) and '#' in line:
            current_section = 'resources'
            sections[current_section] += line + '\n'
        elif current_section:
            # Add the line to the current section
            sections[current_section] += line + '\n'
    
    return sections

def generate_fallback_lesson_plan(subject, topic, grade_level, duration, include_differentiation, include_assessment, include_resources):
    """
    Generate a fallback lesson plan when AI is unavailable
    
    Args:
        subject, topic, grade_level, duration: Basic lesson parameters
        include_differentiation, include_assessment, include_resources: Optional section flags
        
    Returns:
        Dictionary with structured section content
    """
    # Calculate timing based on duration
    try:
        total_minutes = int(duration.split()[0])
    except:
        total_minutes = 60  # Default to 60 minutes
        
    intro_time = max(5, int(total_minutes * 0.15))
    main_time = int(total_minutes * 0.5)
    group_time = int(total_minutes * 0.25)
    reflection_time = max(5, int(total_minutes * 0.1))
    
    # Create a basic lesson plan structure
    sections = {
        'objectives': f"## Learning Objectives\n\n* Students will be able to describe key concepts related to {topic} in {subject}\n* Students will be able to apply {topic} concepts to solve problems\n* Students will be able to analyze the importance of {topic} in the context of {subject}\n",
        
        'materials': f"## Materials Needed\n\n* Textbook or reference materials on {topic}\n* Worksheets\n* Whiteboard/projector\n* Student notebooks\n",
        
        'preparation': f"## Preparation Steps\n\n* Review lesson content on {topic}\n* Prepare handouts\n* Set up classroom for group activities\n",
        
        'introduction': f"## Introduction ({intro_time} minutes)\n\n* Begin with an engaging question about {topic}\n* Ask students to share prior knowledge about {topic}\n* Introduce the learning objectives for the lesson\n",
        
        'main_activity': f"## Main Activity ({main_time} minutes)\n\n* Present key concepts about {topic}\n* Demonstrate example problems or applications\n* Guide students through initial practice\n* Check for understanding throughout\n",
        
        'group_work': f"## Group Work/Practice ({group_time} minutes)\n\n* Divide students into small groups\n* Provide problem set or discussion questions about {topic}\n* Circulate to provide guidance and answer questions\n* Have groups share findings/solutions\n",
        
        'reflection': f"## Reflection and Assessment ({reflection_time} minutes)\n\n* Review key concepts from the lesson\n* Ask students to summarize what they learned about {topic}\n* Preview upcoming related topics\n* Assign homework or follow-up activities\n",
        
        'differentiation': '',
        'assessment': '',
        'resources': '',
        'metadata': {
            'subject': subject,
            'topic': topic,
            'grade_level': grade_level,
            'duration': duration,
            'generated_with_ai': False
        }
    }
    
    # Add optional sections if requested
    if include_differentiation:
        sections['differentiation'] = f"## Differentiation Strategies\n\n* For advanced students: Provide more complex problems related to {topic}\n* For struggling students: Offer simplified examples and additional support\n* For English language learners: Provide visual aids and vocabulary support\n"
        
    if include_assessment:
        sections['assessment'] = f"## Assessment Strategies\n\n* Formative: Monitor participation and check group work\n* Summative: Short quiz on {topic} concepts\n* Extension: Project applying {topic} to real-world scenarios\n"
        
    if include_resources:
        sections['resources'] = f"## Resources and References\n\n* Textbook chapters on {topic}\n* Online resources for {subject}: [Subject Website](https://example.com/{subject.lower()})\n* Practice worksheets on {topic}\n"
    
    # Combine all sections for full text
    full_text = ""
    for section in sections:
        if section != 'metadata' and section != 'full_text' and sections[section]:
            full_text += sections[section] + "\n"
    
    sections['full_text'] = full_text
    
    return sections 