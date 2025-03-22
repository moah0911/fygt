#!/usr/bin/env python

def fix_specific_indent_issue(file_path):
    """Fix the specific indentation issue around line 1491 in the file."""
    print(f"Fixing indent issue in {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # The issue is around line 1491 where an else is misaligned
        # Let's find the section and fix it
        
        # Look for the pattern of the else line
        else_line_index = -1
        for i in range(1485, 1495):  # Search around line 1491
            if i < len(lines) and "else:" in lines[i] and len(lines[i].strip()) <= 6:
                else_line_index = i
                break
        
        if else_line_index == -1:
            print("Could not find the problematic else line")
            return False
        
        # Find the if statement that this else belongs to
        # Based on the structure, it appears to be part of a larger if-elif chain
        
        # Let's find if there's an "elif" statement before the else
        elif_line_index = -1
        for i in range(else_line_index - 20, else_line_index):
            if i >= 0 and "elif" in lines[i]:
                elif_line_index = i
                break
        
        if elif_line_index != -1:
            # Get the indentation level of the elif
            elif_indent = len(lines[elif_line_index]) - len(lines[elif_line_index].lstrip())
            
            # Adjust the else line to match the elif indentation
            lines[else_line_index] = " " * elif_indent + "else:\n"
            
            # Now we need to fix the indentation of the block following the else
            indent_for_block = elif_indent + 4  # Standard Python indentation is 4 spaces
            
            # Process the next lines, adjusting indentation
            i = else_line_index + 1
            while i < len(lines):
                line = lines[i]
                if line.strip() == "":
                    i += 1
                    continue
                    
                # Check if this line is part of the else block
                current_indent = len(line) - len(line.lstrip())
                
                # If indentation is clearly misaligned (likely from the else block)
                if current_indent > 20:  # Assuming the misaligned indentation is large
                    # Adjust to the proper indentation level
                    lines[i] = " " * indent_for_block + line.lstrip()
                    i += 1
                else:
                    # We've reached the end of the else block
                    break
        
        # Write the corrected content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Indentation issue fixed successfully")
        return True
    except Exception as e:
        print(f"Error fixing indentation: {str(e)}")
        return False

if __name__ == "__main__":
    fix_specific_indent_issue('streamlit_app.py') 