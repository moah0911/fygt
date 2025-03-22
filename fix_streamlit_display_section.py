#!/usr/bin/env python

def fix_streamlit_display_section():
    """Fix the indentation in the streamlit display section."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find the beginning of the problematic section
        topic_intro_index = -1
        for i in range(1500, 1520):
            if i < len(lines) and "topic_intro" in lines[i]:
                topic_intro_index = i
                break
        
        if topic_intro_index == -1:
            print("Could not find topic_intro line")
            return False
        
        # Determine correct indentation from the indentation of the topic_intro line
        topic_intro_indent = len(lines[topic_intro_index]) - len(lines[topic_intro_index].lstrip())
        # The streamlit displays should be at the same indentation level
        display_indent = topic_intro_indent
        
        # Fix indentation of all display lines after topic_intro
        start_index = topic_intro_index + 1
        end_index = min(start_index + 50, len(lines))
        
        # Process all lines in the problematic section
        for i in range(start_index, end_index):
            line = lines[i]
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                continue
                
            # Handle comments (keep their indentation level consistent)
            if stripped_line.startswith("#"):
                lines[i] = " " * display_indent + stripped_line + "\n"
                continue
                
            # Handle Streamlit function calls
            if "st.write" in stripped_line:
                lines[i] = " " * display_indent + stripped_line + "\n"
            # Handle for loops
            elif stripped_line.startswith("for "):
                lines[i] = " " * display_indent + stripped_line + "\n"
            # Content inside loops needs extra indentation
            elif any(x in stripped_line for x in ["f\"{i}", "f\"- "]):
                lines[i] = " " * (display_indent + 4) + stripped_line + "\n"
        
        # Write the corrected content back
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Fixed indentation in streamlit display section")
        return True
    except Exception as e:
        print(f"Error fixing streamlit display section: {str(e)}")
        return False

if __name__ == "__main__":
    fix_streamlit_display_section() 