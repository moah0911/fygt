#!/usr/bin/env python
import re

def fix_inline_with_statements(file_path):
    """Fix instances where 'with' statements are on the same line as other code."""
    print(f"Fixing inline with statements in {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Pattern to match 'with' statements on the same line as other code
        # For example: 'col1, col2 = st.columns([10, 1])            with col1:'
        pattern = r'(.*?)(\s+with\s+.*?:)'
        
        # Replace inline with statements with properly formatted ones
        # This splits the line into two lines: one with the preceding code, and another with the 'with' statement
        fixed_content = re.sub(pattern, r'\1\n\2', content)
        
        # Additional fix for cases like 'with col1: <other code>'
        pattern2 = r'(with\s+.*?:)(\s+.*)'
        fixed_content = re.sub(pattern2, r'\1\n\2', fixed_content)
        
        # Write the corrected content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("Inline with statements fixed successfully")
        return True
    except Exception as e:
        print(f"Error fixing inline with statements: {str(e)}")
        return False

if __name__ == "__main__":
    fix_inline_with_statements('streamlit_app.py') 