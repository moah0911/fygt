import re

def fix_indentation():
    # Read the file
    with open('streamlit_app.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Create a backup
    with open('streamlit_app.py.bak', 'w', encoding='utf-8') as backup:
        backup.writelines(lines)
    
    # Fix the problematic lines around line 5395
    fixed_lines = lines[:]
    
    # Specifically fixing lines 5395-5401 (0-indexed would be 5394-5400)
    if len(fixed_lines) >= 5402:
        # Check if there's an issue with these specific lines
        line_5395 = fixed_lines[5394] if len(fixed_lines) > 5394 else ""
        line_5396 = fixed_lines[5395] if len(fixed_lines) > 5395 else ""
        
        # Check if there's extra indentation
        if line_5395.startswith('                with col1:'):
            fixed_lines[5394] = '            with col1:\n'
            print("Fixed line 5395")
        
        if line_5396.startswith('                avg_score'):
            fixed_lines[5395] = '                avg_score = metrics.get(\'average_score\', 0)\n'
            print("Fixed line 5396")
        
        # Check and fix the next few lines with similar pattern
        for i in range(5396, 5402):
            if i < len(fixed_lines):
                # Get the current line
                current_line = fixed_lines[i]
                
                # Remove extra indentation if present
                if current_line.startswith('                '):
                    # Keep the line content but fix indentation
                    fixed_lines[i] = '                ' + current_line.lstrip()
                    print(f"Fixed line {i+1}")
    
    # Write the fixed content back to the file
    with open('streamlit_app.py', 'w', encoding='utf-8') as file:
        file.writelines(fixed_lines)
    
    print("Indentation fix completed.")

if __name__ == "__main__":
    fix_indentation() 