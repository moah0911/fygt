def fix_fstring_error():
    # Create a backup first
    import shutil
    shutil.copy('streamlit_app.py', 'streamlit_app.py.fstring_backup')
    print("Created backup as streamlit_app.py.fstring_backup")
    
    # Read the file content
    with open('streamlit_app.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Fix the unterminated f-string at line 5813
    # The problematic lines are around 5813 where an f-string is split across lines
    for i in range(5810, 5815):
        if i < len(lines):
            # Look for the start of the problematic f-string
            if 'with st.expander(f"' in lines[i] and 'Risk Score:' in lines[i]:
                # Check if the next line contains the continuation
                if i+1 < len(lines) and '{student_data.get' in lines[i+1]:
                    # Get the content of both lines
                    line1 = lines[i].strip()
                    line2 = lines[i+1].strip()
                    
                    # Remove the f-string opening from line1
                    content1 = line1.split('f"')[1].rstrip('"')
                    
                    # Get the continuation from line2
                    content2 = line2.rstrip('"):')
                    
                    # Create a fixed combined line with proper indentation
                    indent = ' ' * (len(lines[i]) - len(lines[i].lstrip()))
                    fixed_line = f'{indent}with st.expander(f"{content1} {content2}"):\n'
                    
                    # Replace the problematic lines
                    lines[i] = fixed_line
                    lines[i+1] = ''  # Empty the second line
                    
                    print(f"Fixed unterminated f-string at lines {i+1}-{i+2}")
                    break
    
    # Write the fixed lines back
    with open('streamlit_app.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("F-string error fixed successfully.")

if __name__ == "__main__":
    fix_fstring_error() 