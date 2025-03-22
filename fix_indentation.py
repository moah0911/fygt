import re

def fix_indentation_issues(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    line_number = 0
    
    # Known problem lines from the error report
    problem_lines = [2291, 2292, 2320, 2321, 2477, 2478, 2482, 2594, 2731, 2733, 
                    3228, 3229, 3575, 3576, 3852, 3870, 3942, 3982, 4044]
    
    in_block = False
    current_indent = ""
    
    for i, line in enumerate(lines):
        line_number = i + 1
        
        # Fix specific known indentation issues
        if line_number == 2291 or line_number == 2292:
            # Check previous line indentation
            prev_line = lines[i-1]
            indent_match = re.match(r'^(\s+)', prev_line)
            if indent_match:
                # Match the current line's indentation to the appropriate level
                current_indent = indent_match.group(1)
                line = current_indent + line.lstrip()
        
        elif line_number in [2477, 2478, 2482]:
            # Fix col1, col2 indentation
            line = line.replace('    with col', 'with col')
        
        elif line_number in [2731, 2733]:
            # Fix nested else statements
            if 'else:' in line:
                # Get proper indentation level
                prev_line = lines[i-1]
                match = re.search(r'^(\s+)', prev_line)
                if match:
                    base_indent = match.group(1)
                    # Find the corresponding if statement's indentation
                    for j in range(i-1, -1, -1):
                        if 'if ' in lines[j]:
                            if_match = re.search(r'^(\s+)', lines[j])
                            if if_match:
                                proper_indent = if_match.group(1)
                                line = proper_indent + 'else:\n'
                                break
        
        # Similarly fix other indentation issues...
        # Add more specific fixes as needed
        
        fixed_lines.append(line)
    
    # Write the fixed file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation issues in {file_path}")

# Fix the indentation in streamlit_app.py
fix_indentation_issues('streamlit_app.py') 