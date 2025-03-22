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
from edumate.utils.lesson_planner import generate_lesson_plan

def show_test_creator():
    """
    Display the test creator interface - integrates with the original streamlit_app.py
    """
    st.header("Test Creator")
    
    # Set original session state variable to maintain compatibility with main app
    if 'new_test_created' not in st.session_state:
        st.session_state.new_test_created = False
    
    # Create tabs for different test types
    tab1, tab2, tab3 = st.tabs(["Quiz Generator", "Lesson Plan Creator", "Rubric Builder"])
    
    with tab1:
        show_quiz_generator()
    
    with tab2:
        show_lesson_plan_creator()
    
    with tab3:
        show_rubric_builder()

def show_quiz_generator():
    """Display the AI-powered quiz generator interface"""
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
            fill_blanks = st.checkbox("Fill in the Blanks")
        
        # Combine selected question types
        question_types = []
        if mc: question_types.append("Multiple Choice")
        if tf: question_types.append("True/False")
        if sa: question_types.append("Short Answer")
        if essay: question_types.append("Essay")
        if fill_blanks: question_types.append("Fill in the Blanks")
        
        # API key option
        use_gemini_api = st.checkbox("Use Gemini AI for enhanced content", value=True)
        if use_gemini_api:
            st.info("Using Gemini API to generate high-quality, curriculum-aligned content")
        
        # Generate button
        generate_quiz_button = st.form_submit_button("Generate Quiz")
        
        if generate_quiz_button:
            if not quiz_subject or not quiz_topic:
                st.error("Please enter both subject and topic.")
            elif not question_types:
                st.error("Please select at least one question type.")
            else:
                with st.spinner("Generating AI-powered quiz..."):
                    # Artificial delay for API call simulation if needed
                    time.sleep(1)
                    
                    # Generate the quiz content using our AI-powered function
                    quiz_content = generate_quiz(quiz_subject, quiz_topic, quiz_level, quiz_questions, question_types)
                    
                    # Set session state variable for compatibility with main app
                    st.session_state.new_test_created = True
                    
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
    """Display the AI-powered lesson plan creator interface"""
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
                "90 minutes", "120 minutes", "180 minutes"
            ])
        
        st.write("Special Considerations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ai_differentiation = st.checkbox("Include differentiation strategies")
        
        with col2:
            ai_assessment = st.checkbox("Include assessment strategies")
        
        with col3:
            ai_resources = st.checkbox("Include resource links", value=True)
        
        # API key option
        use_gemini_api = st.checkbox("Use Gemini AI for enhanced lesson plans", value=True)
        if use_gemini_api:
            st.info("Using Gemini API to generate detailed, standards-aligned lesson plans")
        
        # Submit button
        submit_lesson = st.form_submit_button("Generate Lesson Plan")
        
        if submit_lesson:
            if not ai_subject or not ai_topic:
                st.error("Please enter both subject and topic.")
            else:
                with st.spinner("Creating AI-powered lesson plan..."):
                    # Generate the lesson plan using our AI function
                    lesson_plan = generate_lesson_plan(
                        ai_subject, 
                        ai_topic, 
                        ai_grade, 
                        ai_duration, 
                        ai_differentiation, 
                        ai_assessment, 
                        ai_resources
                    )
                    
                    # Success message
                    if lesson_plan.get('metadata', {}).get('generated_with_ai', False):
                        st.success("✅ AI-powered lesson plan generated successfully!")
                    else:
                        st.success("Lesson plan generated successfully!")
                    
                    # Display the lesson plan overview
                    st.write("### Lesson Plan Overview")
                    st.write(f"**Subject:** {ai_subject}")
                    st.write(f"**Topic:** {ai_topic}")
                    st.write(f"**Grade Level:** {ai_grade}")
                    st.write(f"**Duration:** {ai_duration}")
                    
                    # Display each section of the lesson plan in separate expandable sections
                    if lesson_plan.get('objectives'):
                        with st.expander("Learning Objectives", expanded=True):
                            st.markdown(lesson_plan['objectives'])
                    
                    if lesson_plan.get('materials'):
                        with st.expander("Materials Needed"):
                            st.markdown(lesson_plan['materials'])
                    
                    if lesson_plan.get('preparation'):
                        with st.expander("Preparation Steps"):
                            st.markdown(lesson_plan['preparation'])
                            
                    if lesson_plan.get('introduction'):
                        with st.expander("Introduction"):
                            st.markdown(lesson_plan['introduction'])
                    
                    if lesson_plan.get('main_activity'):
                        with st.expander("Main Activity"):
                            st.markdown(lesson_plan['main_activity'])
                    
                    if lesson_plan.get('group_work'):
                        with st.expander("Group Work/Practice"):
                            st.markdown(lesson_plan['group_work'])
                    
                    if lesson_plan.get('reflection'):
                        with st.expander("Reflection and Assessment"):
                            st.markdown(lesson_plan['reflection'])
                    
                    if ai_differentiation and lesson_plan.get('differentiation'):
                        with st.expander("Differentiation Strategies"):
                            st.markdown(lesson_plan['differentiation'])
                    
                    if ai_assessment and lesson_plan.get('assessment'):
                        with st.expander("Assessment Strategies"):
                            st.markdown(lesson_plan['assessment'])
                    
                    if ai_resources and lesson_plan.get('resources'):
                        with st.expander("Resources and References"):
                            st.markdown(lesson_plan['resources'])
                    
                    # Add download button for the full lesson plan
                    if lesson_plan.get('full_text'):
                        st.download_button(
                            label="Download Complete Lesson Plan",
                            data=lesson_plan['full_text'],
                            file_name=f"{ai_subject}_{ai_topic}_Lesson_Plan.md",
                            mime="text/markdown"
                        )

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
        
        # Add option to generate criteria using AI
        use_ai_criteria = st.checkbox("Suggest criteria using AI", value=False)
        if use_ai_criteria:
            st.info("AI will suggest appropriate criteria based on assignment type")
        
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
                    time.sleep(1)
                    
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

# This allows the component to be imported by the original app
if __name__ == "__main__":
    show_test_creator() 