"""
Script to connect our modular components to the original streamlit_app.py
This ensures that our new AI-powered components are properly integrated
with the existing application while maintaining its core functionality.
"""

import os
import sys
import traceback

# Add this directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    """Connect our modular components to the main streamlit_app.py"""
    print("Connecting modular components to main application...")
    
    # Check if required files and directories exist
    if not os.path.exists('streamlit_app.py'):
        print("Error: streamlit_app.py not found in current directory")
        return False
        
    if not os.path.exists('edumate/pages/test_creator.py'):
        print("Error: edumate/pages/test_creator.py not found")
        print("Make sure all modular components are created before running this script")
        return False
    
    # Try importing our component to verify it works
    try:
        print("Checking if modular components can be imported...")
        sys.path.insert(0, os.path.abspath('.'))
        from edumate.pages.test_creator import show_test_creator
        print("Successfully imported modular component")
    except Exception as e:
        print(f"Error importing modular component: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False
    
    # Create backup of original file if needed
    source_file = "streamlit_app.py"
    backup_file = "streamlit_app.py.bak"
    
    if not os.path.exists(backup_file):
        print(f"Creating backup of {source_file}...")
        try:
            with open(source_file, 'r', encoding='utf-8') as src_file:
                content = src_file.read()
            
            with open(backup_file, 'w', encoding='utf-8') as bak_file:
                bak_file.write(content)
                
            print(f"Backup created at {backup_file}")
        except Exception as e:
            print(f"Error creating backup: {e}")
            traceback.print_exc()
            return False
    
    # Now read the file to patch
    try:
        print("Reading streamlit_app.py for patching...")
        with open(source_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        print(f"File read successfully, {len(lines)} lines")
        
        # Find the show_test_creator function
        test_creator_start = None
        test_creator_end = None
        
        for i, line in enumerate(lines):
            if "def show_test_creator():" in line:
                test_creator_start = i
                print(f"Found function definition at line {i}: {line.strip()}")
            elif test_creator_start is not None and line.startswith("def ") and "def show_test_creator" not in line:
                test_creator_end = i
                print(f"Found next function at line {i}: {line.strip()}")
                break
        
        if test_creator_start is None:
            print("Error: Could not find 'def show_test_creator():' in the file")
            return False
            
        if test_creator_end is None:
            print("Warning: Could not find end of function. Using end of file.")
            test_creator_end = len(lines)
        
        print(f"Found show_test_creator function: lines {test_creator_start} to {test_creator_end}")
            
        # Create the new code to inject
        new_code = [
            "def show_test_creator():\n",
            "    \"\"\"Display the test creation interface for teachers.\"\"\"\n",
            "    # Import required modules\n",
            "    import streamlit as st\n",
            "    import sys\n",
            "    import os\n",
            "    \n",
            "    # Import the modular test creator component\n",
            "    try:\n",
            "        from edumate.pages.test_creator import show_test_creator as modular_test_creator\n",
            "        \n",
            "        # Call the modular implementation\n",
            "        modular_test_creator()\n",
            "    except Exception as e:\n",
            "        st.error(f\"Error loading test creator: {e}\")\n",
            "        st.warning(\"Falling back to original test creator. Please check console for errors.\")\n",
            "        \n",
            "        # Original implementation can go here as fallback\n",
            "        st.header(\"Create Test\")\n",
            "        st.info(\"The enhanced AI test creator is currently unavailable.\")\n",
            "\n"
        ]
        
        # Replace the original function with our connector
        modified_lines = lines[:test_creator_start] + new_code + lines[test_creator_end:]
        
        print(f"Patching file: Replacing original function with connector code ({len(new_code)} lines)")
        
        # Write the modified file
        with open(source_file, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)
            
        print(f"Successfully patched {source_file} to use modular components")
        print("The application now uses our AI-powered test creator components!")
        
        return True
            
    except Exception as e:
        print(f"Error patching file: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n--- Edumate Component Connector ---\n")
    success = main()
    if success:
        print("\nConnector script completed successfully!")
        print("Run 'streamlit run streamlit_app.py' to use the enhanced application")
    else:
        print("\nConnector script encountered errors")
        print("Please check the error messages above") 