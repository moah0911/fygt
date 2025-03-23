"""
AI Tutor Page
A Streamlit page that provides AI tutoring for students
"""

import streamlit as st
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from streamlit_integration import get_edumate_instance, INTEGRATION_AVAILABLE
from edumate.services.ai_service import AIService

def show_ai_tutor_page():
    """Display the AI tutor interface for students."""
    st.title("AI Tutor")
    
    # Initialize variables
    ai_topic = ""
    ai_grade = ""
    ai_subject = ""
    ai_duration = ""
    
    # Check if integration is available
    if INTEGRATION_AVAILABLE:
        st.write("Ask me anything about your coursework, and I'll help you understand it better.")
        
        # Input form for questions
        with st.form("ai_tutor_form"):
            user_question = st.text_area("Your Question", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                subject = st.selectbox(
                    "Subject",
                    ["Math", "Science", "History", "English", "Computer Science", "Other"]
                )
            
            with col2:
                learning_context = st.selectbox(
                    "I'm trying to...",
                    [
                        "Understand a concept",
                        "Solve a problem",
                        "Prepare for a test",
                        "Work on a project",
                        "Other"
                    ]
                )
            
            submit_button = st.form_submit_button("Get Help")
            
            if submit_button and user_question:
                try:
                    # Get educated response from Edumate
                    edumate = get_edumate_instance()
                    
                    # Response will contain the answer and any related skills
                    response = edumate.get_personalized_response(
                        question=user_question,
                        subject=subject,
                        context=learning_context,
                        user_id=st.session_state.current_user.get('id', 'anonymous')
                    )
                    
                    # Display the response
                    st.markdown("### Answer")
                    st.markdown(response['answer'])
                    
                    # Display related skills if available
                    if 'related_skills' in response and response['related_skills']:
                        st.markdown("### Related Skills")
                        for skill in response['related_skills']:
                            st.markdown(f"- **{skill['name']}**: {skill['description']}")
                            
                            # Show proficiency if available
                            if 'proficiency' in skill:
                                st.progress(skill['proficiency'] / 100)
                                st.write(f"Your proficiency: {skill['proficiency']}%")
                
                except Exception as e:
                    st.error(f"Error connecting to Edumate: {str(e)}")
                    st.write("Falling back to basic tutor...")
                    
                    # Fallback to basic tutor
                    basic_tutor_response = AIService.get_response(
                        prompt=f"Question: {user_question}\nSubject: {subject}\nContext: {learning_context}",
                        system_message="You are a helpful educational AI tutor. Provide clear, concise explanations that are appropriate for students."
                    )
                    
                    st.markdown("### Answer")
                    st.markdown(basic_tutor_response)
    else:
        # No integration available, use the basic tutor
        st.write("Ask me anything about your coursework, and I'll help you understand it better.")
        
        with st.form("basic_tutor_form"):
            user_question = st.text_area("Your Question", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                subject = st.selectbox(
                    "Subject",
                    ["Math", "Science", "History", "English", "Computer Science", "Other"]
                )
            
            with col2:
                learning_context = st.selectbox(
                    "I'm trying to...",
                    [
                        "Understand a concept",
                        "Solve a problem",
                        "Prepare for a test",
                        "Work on a project",
                        "Other"
                    ]
                )
            
            submit_button = st.form_submit_button("Get Help")
            
            if submit_button and user_question:
                # Use the basic AI service for response
                basic_tutor_response = AIService.get_response(
                    prompt=f"Question: {user_question}\nSubject: {subject}\nContext: {learning_context}",
                    system_message="You are a helpful educational AI tutor. Provide clear, concise explanations that are appropriate for students."
                )
                
                st.markdown("### Answer")
                st.markdown(basic_tutor_response) 