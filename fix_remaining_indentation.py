#!/usr/bin/env python

def fix_remaining_indentation():
    """Fix the remaining indentation issues in the file."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # The section after the rebuilt part still has indentation issues
        # Find the line with Learning Objectives
        learning_obj_index = -1
        for i in range(1510, 1530):
            if i < len(lines) and 'st.write("### Learning Objectives")' in lines[i]:
                learning_obj_index = i
                break
        
        if learning_obj_index == -1:
            print("Could not find Learning Objectives section")
            return False
        
        # We need to normalize the indentation in this section
        # Determine the correct base indentation level from the surrounding code
        correct_indent = None
        
        # Look at previous few lines to determine correct indentation
        for i in range(learning_obj_index - 5, learning_obj_index):
            if i >= 0 and lines[i].strip() and not lines[i].strip().startswith('#'):
                base_indent = len(lines[i]) - len(lines[i].lstrip())
                if "else:" not in lines[i]:  # Don't use else line as reference
                    correct_indent = base_indent
                    break
        
        if correct_indent is None:
            # Default to 24 spaces based on inspection
            correct_indent = 24
            print(f"Using default indentation of {correct_indent} spaces")
        else:
            print(f"Using detected indentation of {correct_indent} spaces")
        
        # Fix indentation from Learning Objectives section to around line 1580
        end_index = min(learning_obj_index + 70, len(lines))
        for i in range(learning_obj_index, end_index):
            line = lines[i].rstrip()
            
            if not line:  # Skip empty lines
                continue
                
            stripped_line = line.strip()
            if stripped_line.startswith('#'):  # Preserve comment indentation
                continue
            
            # Apply consistent indentation
            if stripped_line.startswith("st.write("):
                # Main section headers and content
                lines[i] = " " * correct_indent + stripped_line + "\n"
            elif stripped_line.startswith("for "):
                # For loops should be at same level as content
                lines[i] = " " * correct_indent + stripped_line + "\n"
            else:
                # Indented block content inside loops should be indented more
                lines[i] = " " * (correct_indent + 4) + stripped_line + "\n"
        
        # Write the corrected content back
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Fixed remaining indentation issues")
        return True
    except Exception as e:
        print(f"Error fixing remaining indentation: {str(e)}")
        return False

if __name__ == "__main__":
    fix_remaining_indentation() 