import re

def fix_syntax_errors(file_path):
    """Fix all instances of the 'statement else:' syntax error in a Python file."""
    print(f"Opening file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Fix the pattern: any non-whitespace ending with whitespace+else:
        # This will fix statements like "return x            else:"
        content = re.sub(r'(\S.+?)(\s+else:)', r'\1\n\2', content)
        
        # Write back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed syntax errors in {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing syntax errors: {str(e)}")
        return False

if __name__ == "__main__":
    fix_syntax_errors('streamlit_app.py') 