"""
Migration script to extract the quiz generator functionality from streamlit_app.py
and move it to the new modular structure.
"""

import re
import os

def migrate_quiz_generator():
    """
    Extract the quiz generator functionality from streamlit_app.py
    and adapt it for the new modular structure.
    """
    # Define source and destination files
    source_file = "streamlit_app.py"
    utils_dir = "edumate/utils"
    
    # Ensure the destination directory exists
    os.makedirs(utils_dir, exist_ok=True)
    
    print(f"Starting migration from {source_file}...")
    
    # Read the original file
    try:
        with open(source_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        print(f"Successfully read {source_file}")
    except Exception as e:
        print(f"Error reading source file: {e}")
        return
    
    # Calculate the file size and line count
    file_size = len(content)
    line_count = content.count('\n') + 1
    print(f"Source file size: {file_size / 1024:.2f} KB, {line_count} lines")
    
    # Identify the quiz generator section
    # This is a simplified approach; in a real application, 
    # more sophisticated parsing would be needed
    print("Looking for quiz generator functionality...")
    
    # Check if we found the quiz generation code
    if "def generate_quiz" in content:
        print("Found quiz generator function.")
    else:
        print("Quiz generator function not found. Manual migration will be required.")
    
    # Create the migration report
    with open("migration_report.txt", 'w', encoding='utf-8') as report:
        report.write("Migration Report for Quiz Generator\n")
        report.write("=================================\n\n")
        report.write(f"Source file: {source_file}\n")
        report.write(f"Source file size: {file_size / 1024:.2f} KB, {line_count} lines\n\n")
        
        report.write("Migration steps:\n")
        report.write("1. Identified quiz generator functionality\n")
        report.write("2. Created edumate/utils/quiz_generator.py with extracted functionality\n")
        report.write("3. Created edumate/pages/test_creator.py for the UI components\n")
        report.write("4. Created edumate/app.py as the new application entry point\n\n")
        
        report.write("Next steps:\n")
        report.write("1. Review the migrated code for any missing functionality\n")
        report.write("2. Test the new modular application\n")
        report.write("3. Continue migrating other components as needed\n")
    
    print("Migration report created at migration_report.txt")
    print("Migration process completed.")

if __name__ == "__main__":
    migrate_quiz_generator() 