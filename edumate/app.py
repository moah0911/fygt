"""
Edumate - Educational Platform Main Application
"""

import streamlit as st
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import page components
from edumate.pages.test_creator import show_test_creator

# Configure the Streamlit page
st.set_page_config(
    page_title="Edumate - Educational Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    st.sidebar.title("Edumate")
    st.sidebar.image("https://via.placeholder.com/150x150.png?text=Edumate", width=150)
    
    # Navigation options
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Test Creator", "Student Progress", "AI Tutor", "Resources", "Settings"]
    )
    
    # Display selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Test Creator":
        show_test_creator()
    elif page == "Student Progress":
        show_student_progress()
    elif page == "AI Tutor":
        show_ai_tutor()
    elif page == "Resources":
        show_resources()
    elif page == "Settings":
        show_settings()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2023 Edumate - Educational Platform")

def show_dashboard():
    """Display the dashboard page"""
    st.header("Dashboard")
    st.write("Welcome to Edumate, your AI-powered educational platform!")
    
    # Create dashboard cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Recent Activity")
        st.info("No recent activity to display.")
        
    with col2:
        st.subheader("Quick Access")
        st.button("Create New Test")
        st.button("View Student Reports")
        
    with col3:
        st.subheader("System Status")
        st.success("All systems operational")
    
    # Metrics section
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", "0", "0%")
    col2.metric("Quizzes", "0", "0%")
    col3.metric("Resources", "0", "0%")
    col4.metric("AI Sessions", "0", "0%")

def show_student_progress():
    """Display the student progress page"""
    st.header("Student Progress")
    st.write("Track and analyze student performance.")
    
    st.info("Student progress tracking is under development.")

def show_ai_tutor():
    """Display the AI tutor page"""
    st.header("AI Tutor")
    st.write("Interactive AI-powered tutoring sessions.")
    
    st.info("AI tutoring feature is under development.")

def show_resources():
    """Display the resources page"""
    st.header("Educational Resources")
    st.write("Browse and manage educational resources.")
    
    st.info("Resources management is under development.")

def show_settings():
    """Display the settings page"""
    st.header("Settings")
    st.write("Configure your Edumate experience.")
    
    st.info("Settings configuration is under development.")

if __name__ == "__main__":
    main() 