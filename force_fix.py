#!/usr/bin/env python

def force_fix():
    """Force fix the indentation issue by directly modifying specific lines."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Based on the error message, line 1491 has an 'else:' statement
        # and line 1492 needs to be properly indented
        
        # First find the else line
        else_line_index = -1
        for i in range(1488, 1495):
            if i < len(lines) and "else:" in lines[i] and len(lines[i].strip()) <= 6:
                else_line_index = i
                break
        
        if else_line_index == -1:
            print("Could not find the else line")
            return False
        
        print(f"Found else at line {else_line_index + 1}: {lines[else_line_index].strip()}")
        
        # Get the indentation level of the else
        else_indent = len(lines[else_line_index]) - len(lines[else_line_index].lstrip())
        correct_block_indent = else_indent + 4  # Standard indentation is 4 spaces
        
        # Check and fix the indentation of the next line (the objectives line)
        next_line_index = else_line_index + 1
        if next_line_index < len(lines) and "objectives" in lines[next_line_index]:
            current_indent = len(lines[next_line_index]) - len(lines[next_line_index].lstrip())
            
            if current_indent != correct_block_indent:
                # Fix the indentation
                lines[next_line_index] = " " * correct_block_indent + lines[next_line_index].lstrip()
                print(f"Fixed indentation of line {next_line_index + 1}")
        
            # Now fix the indentation of the list items and subsequent blocks
            # This will be a bit crude - we'll just properly indent blocks we find
            for i in range(next_line_index + 1, min(next_line_index + 50, len(lines))):
                line = lines[i]
                if not line.strip():
                    continue
                
                # Deeper indentation needed for list items inside the block
                if "]" in line and line.strip() == "]":
                    # End of list - should be at block level
                    lines[i] = " " * correct_block_indent + line.lstrip()
                elif "[" in line or "f\"" in line:
                    # List items/string literals - should be indented further
                    lines[i] = " " * (correct_block_indent + 4) + line.lstrip()
                elif "materials" in line or "activities" in line or "assessment" in line or "topic_intro" in line:
                    # These are block-level variables similar to objectives
                    lines[i] = " " * correct_block_indent + line.lstrip()
        
        # Write the fixed content
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Force fix applied")
        return True
    except Exception as e:
        print(f"Error applying force fix: {str(e)}")
        return False

if __name__ == "__main__":
    force_fix() 