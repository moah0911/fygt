"""
Integration module to connect modular Edumate components with the original streamlit_app.py
"""

import sys
import os
import importlib.util

def import_function_from_file(file_path, function_name):
    """
    Dynamically import a function from a Python file
    
    Args:
        file_path: Path to the Python file
        function_name: Name of the function to import
        
    Returns:
        The imported function
    """
    try:
        # Generate a unique module name
        module_name = f"dynamic_import_{function_name}"
        
        # Load the module specification
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not find file: {file_path}")
        
        # Create the module
        module = importlib.util.module_from_spec(spec)
        
        # Execute the module in its own namespace
        spec.loader.exec_module(module)
        
        # Get the function from the module
        if hasattr(module, function_name):
            return getattr(module, function_name)
        else:
            raise AttributeError(f"Function '{function_name}' not found in {file_path}")
            
    except Exception as e:
        print(f"Error importing {function_name} from {file_path}: {e}")
        return None

def get_original_functions():
    """
    Get references to important functions from the original streamlit_app.py
    """
    streamlit_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'streamlit_app.py'))
    
    functions = {
        # Core functionality
        'load_data': import_function_from_file(streamlit_app_path, 'load_data'),
        'save_data': import_function_from_file(streamlit_app_path, 'save_data'),
        'register_user': import_function_from_file(streamlit_app_path, 'register_user'),
        'login_user': import_function_from_file(streamlit_app_path, 'login_user'),
        
        # Course management
        'create_course': import_function_from_file(streamlit_app_path, 'create_course'),
        'get_teacher_courses': import_function_from_file(streamlit_app_path, 'get_teacher_courses'),
        'get_student_courses': import_function_from_file(streamlit_app_path, 'get_student_courses'),
        'enroll_student': import_function_from_file(streamlit_app_path, 'enroll_student'),
        
        # Assignment management
        'create_assignment': import_function_from_file(streamlit_app_path, 'create_assignment'),
        'get_course_assignments': import_function_from_file(streamlit_app_path, 'get_course_assignments'),
        'delete_assignment': import_function_from_file(streamlit_app_path, 'delete_assignment'),
        'submit_assignment': import_function_from_file(streamlit_app_path, 'submit_assignment'),
        'grade_submission': import_function_from_file(streamlit_app_path, 'grade_submission'),
        
        # AI functionality
        'analyze_with_gemini': import_function_from_file(streamlit_app_path, 'analyze_with_gemini'),
        'analyze_image_with_gemini': import_function_from_file(streamlit_app_path, 'analyze_image_with_gemini'),
        'analyze_pdf_with_gemini': import_function_from_file(streamlit_app_path, 'analyze_pdf_with_gemini'),
        'auto_grade_submission': import_function_from_file(streamlit_app_path, 'auto_grade_submission'),
        'generate_ai_feedback': import_function_from_file(streamlit_app_path, 'generate_ai_feedback'),
    }
    
    return functions

# Functions from the original app
original_functions = get_original_functions()

# Make the original functions available for import
load_data = original_functions.get('load_data')
save_data = original_functions.get('save_data')
register_user = original_functions.get('register_user')
login_user = original_functions.get('login_user')
create_course = original_functions.get('create_course')
get_teacher_courses = original_functions.get('get_teacher_courses')
get_student_courses = original_functions.get('get_student_courses')
enroll_student = original_functions.get('enroll_student')
create_assignment = original_functions.get('create_assignment')
get_course_assignments = original_functions.get('get_course_assignments')
delete_assignment = original_functions.get('delete_assignment')
submit_assignment = original_functions.get('submit_assignment')
grade_submission = original_functions.get('grade_submission')
analyze_with_gemini = original_functions.get('analyze_with_gemini')
analyze_image_with_gemini = original_functions.get('analyze_image_with_gemini')
analyze_pdf_with_gemini = original_functions.get('analyze_pdf_with_gemini')
auto_grade_submission = original_functions.get('auto_grade_submission')
generate_ai_feedback = original_functions.get('generate_ai_feedback') 