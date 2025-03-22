def fix_indentation():
    # Read the file
    with open('streamlit_app.py', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create a backup
    with open('streamlit_app.py.bak2', 'w', encoding='utf-8') as backup:
        backup.write(content)
    
    # Split content into lines for processing
    lines = content.split("\n")
    
    # Check and fix the pattern around line 5399
    if len(lines) >= 5405:
        # Fix the with col2 statement
        if lines[5398].strip().startswith("with col2:"):
            # Ensure with col2 has the same indentation as with col1
            prefix = " " * 12  # 12 spaces for proper indentation
            lines[5398] = prefix + "with col2:"
            
            # Make sure the lines inside with col2: are properly indented
            if 5399 < len(lines) and lines[5399].strip().startswith("completion_rate"):
                lines[5399] = prefix + "    " + lines[5399].strip()
            
            if 5400 < len(lines) and lines[5400].strip().startswith("st.metric"):
                lines[5400] = prefix + "    " + lines[5400].strip()
    
            print("Fixed with col2: block indentation")
    
    # Write the fixed content back to the file
    with open('streamlit_app.py', 'w', encoding='utf-8') as file:
        file.write("\n".join(lines))
    
    print("With statement indentation fix completed.")

if __name__ == "__main__":
    fix_indentation() 