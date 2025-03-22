"""
Test Creator Page
A Streamlit page that allows teachers to create tests and quizzes
"""

import time
import streamlit as st
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from edumate.utils.quiz_generator import generate_quiz, get_resource_suggestions

def show_test_creator():
    """
    Display the test creator interface
    """
    st.header("Test Creator")
    
    # Create tabs for different test types
    tab1, tab2, tab3 = st.tabs(["Quiz Generator", "Lesson Plan Creator", "Rubric Builder"])
    
    with tab1:
        show_quiz_generator()
    
    with tab2:
        show_lesson_plan_creator()
    
    with tab3:
        show_rubric_builder()

def show_quiz_generator():
    """Display the quiz generator interface"""
    st.subheader("AI-Powered Quiz Generator")
    st.write("Create customized quizzes for your students with just a few clicks.")
    
    with st.form("quiz_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            quiz_subject = st.text_input("Subject (e.g., Math, Science, English)")
        
        with col2:
            quiz_topic = st.text_input("Topic (e.g., Fractions, Photosynthesis, Shakespeare)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quiz_level = st.select_slider(
                "Difficulty Level",
                options=["Elementary", "Middle School", "High School", "College", "Advanced"]
            )
        
        with col2:
            quiz_questions = st.slider("Number of Questions", 5, 30, 10)
        
        st.write("Question Types")
        col1, col2 = st.columns(2)
        
        with col1:
            mc = st.checkbox("Multiple Choice", value=True)
            tf = st.checkbox("True/False")
        
        with col2:
            sa = st.checkbox("Short Answer")
            essay = st.checkbox("Essay")
        
        # Combine selected question types
        question_types = []
        if mc: question_types.append("Multiple Choice")
        if tf: question_types.append("True/False")
        if sa: question_types.append("Short Answer")
        if essay: question_types.append("Essay")
        
        # Generate button
        generate_quiz_button = st.form_submit_button("Generate Quiz")
        
        if generate_quiz_button:
            if not quiz_subject or not quiz_topic:
                st.error("Please enter both subject and topic.")
            elif not question_types:
                st.error("Please select at least one question type.")
            else:
                with st.spinner("Generating quiz..."):
                    # Simulate AI processing
                    time.sleep(2)
                    
                    # Generate the quiz content
                    quiz_content = generate_quiz(quiz_subject, quiz_topic, quiz_level, quiz_questions, question_types)
                    
                    # Display the generated quiz
                    st.markdown("### Preview")
                    st.markdown(quiz_content)
                    
                    # Add download button
                    st.download_button(
                        label="Download Quiz (Markdown)",
                        data=quiz_content,
                        file_name=f"{quiz_subject}_{quiz_topic}_Quiz.md",
                        mime="text/markdown"
                    )
                    
                    # Get and display resource suggestions
                    suggestions = get_resource_suggestions(quiz_subject, quiz_topic)
                    
                    st.write("### Helpful Resources")
                    
                    # Display next topics
                    st.write("**Suggested Next Topics:**")
                    for topic in suggestions["next_topics"]:
                        st.write(f"- {topic}")
                    
                    # Display resources
                    st.write("**Recommended Resources:**")
                    for resource in suggestions["resources"]:
                        st.write(f"- [{resource['name']}]({resource['url']})")
                    
                    # Display online courses
                    if suggestions["online_courses"]:
                        st.write("**Online Courses:**")
                        for course in suggestions["online_courses"]:
                            st.write(f"- [{course['name']}]({course['url']})")

def show_lesson_plan_creator():
    """Display the lesson plan creator interface"""
    st.subheader("AI Lesson Plan Creator")
    st.write("Generate comprehensive lesson plans tailored to your teaching needs.")
    
    with st.form("lesson_plan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ai_subject = st.text_input("Subject")
            ai_grade = st.selectbox("Grade Level", [
                "Elementary (K-2)", "Elementary (3-5)", 
                "Middle School (6-8)", "High School (9-12)",
                "College", "Adult Education"
            ])
        
        with col2:
            ai_topic = st.text_input("Specific Topic")
            ai_duration = st.selectbox("Lesson Duration", [
                "30 minutes", "45 minutes", "60 minutes", 
                "90 minutes", "2 hours", "Multiple sessions"
            ])
        
        st.write("Special Considerations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ai_differentiation = st.checkbox("Include differentiation strategies")
        
        with col2:
            ai_assessment = st.checkbox("Include assessment strategies")
        
        with col3:
            ai_resources = st.checkbox("Include resource links", value=True)
        
        # Submit button
        submit_lesson = st.form_submit_button("Generate Lesson Plan")
        
        if submit_lesson:
            if not ai_subject or not ai_topic:
                st.error("Please enter both subject and topic.")
            else:
                with st.spinner("Creating lesson plan..."):
                    # Simulate processing
                    time.sleep(3)
                    
                    # In a real application, this would call an AI service
                    st.success("Lesson plan generated successfully!")
                    
                    # Display a placeholder lesson plan
                    st.write("### Lesson Plan Overview")
                    st.write(f"**Subject:** {ai_subject}")
                    st.write(f"**Topic:** {ai_topic}")
                    st.write(f"**Grade Level:** {ai_grade}")
                    st.write(f"**Duration:** {ai_duration}")
                    
                    # Get resource suggestions
                    suggestions = get_resource_suggestions(ai_subject, ai_topic)
                    
                    # Display lesson structure
                    st.write("### Lesson Structure")
                    st.write("1. Introduction (10 minutes)")
                    st.write("2. Main Activity (25 minutes)")
                    st.write("3. Group Work (15 minutes)")
                    st.write("4. Reflection and Assessment (10 minutes)")
                    
                    # Display resources
                    if ai_resources:
                        st.write("### Resources")
                        for resource in suggestions["resources"]:
                            st.write(f"- [{resource['name']}]({resource['url']})")

def show_rubric_builder():
    """Display the rubric builder interface"""
    st.subheader("Assessment Rubric Builder")
    st.write("Create detailed grading rubrics for assignments and projects.")
    
    with st.form("rubric_form"):
        assignment_name = st.text_input("Assignment Name")
        assignment_type = st.selectbox(
            "Assignment Type",
            ["Essay", "Project", "Presentation", "Lab Report", "Creative Work", "Other"]
        )
        
        st.write("### Criteria")
        criteria = []
        
        for i in range(4):
            col1, col2 = st.columns([3, 1])
            with col1:
                criterion = st.text_input(f"Criterion {i+1}", value="" if i > 0 else "Content")
            with col2:
                weight = st.number_input(f"Weight {i+1} (%)", min_value=0, max_value=100, value=25)
            
            if criterion:
                criteria.append({"name": criterion, "weight": weight})
        
        # Levels of achievement
        st.write("### Levels of Achievement")
        levels = st.slider("Number of Levels", 3, 5, 4)
        level_names = {
            3: ["Needs Improvement", "Satisfactory", "Excellent"],
            4: ["Needs Improvement", "Satisfactory", "Good", "Excellent"],
            5: ["Poor", "Needs Improvement", "Satisfactory", "Good", "Excellent"]
        }
        
        # Generate button
        generate_rubric = st.form_submit_button("Create Rubric")
        
        if generate_rubric:
            if not assignment_name:
                st.error("Please enter an assignment name.")
            elif sum(c["weight"] for c in criteria if c["name"]) != 100:
                st.error("Weights must add up to 100%.")
            else:
                with st.spinner("Creating rubric..."):
                    # Simulate processing
                    time.sleep(2)
                    
                    # Display the generated rubric
                    st.write(f"## Rubric for: {assignment_name}")
                    st.write(f"**Type:** {assignment_type}")
                    
                    # Create and display the rubric table
                    table_header = ["Criteria"] + level_names[levels]
                    table_rows = []
                    
                    for criterion in criteria:
                        if criterion["name"]:
                            row = [f"{criterion['name']} ({criterion['weight']}%)"]
                            for _ in range(levels):
                                row.append("Description would go here")
                            table_rows.append(row)
                    
                    # Display as markdown table
                    table_md = " | ".join(table_header) + "\n"
                    table_md += " | ".join(["---"] * len(table_header)) + "\n"
                    
                    for row in table_rows:
                        table_md += " | ".join(row) + "\n"
                    
                    st.markdown(table_md)
                    
                    # Add download button
                    st.download_button(
                        label="Download Rubric",
                        data=f"# Rubric: {assignment_name}\n\n{table_md}",
                        file_name=f"{assignment_name}_Rubric.md",
                        mime="text/markdown"
                    ) 