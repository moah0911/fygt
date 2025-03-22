#!/usr/bin/env python

def fix_all_blocks_indentation():
    """Fix indentation for an entire section of the file."""
    try:
        # First find the problematic else statement
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find the else line
        else_line_index = -1
        for i in range(1485, 1495):
            if i < len(lines) and "else:" in lines[i].strip():
                else_line_index = i
                print(f"Found else at line {else_line_index + 1}")
                break
        
        if else_line_index == -1:
            print("Could not find the else line")
            return False
        
        # Get the indentation level for the else line
        else_indent = len(lines[else_line_index]) - len(lines[else_line_index].lstrip())
        block_indent = else_indent + 4  # Standard indentation inside else block
        list_indent = block_indent + 4  # Indentation for list items
        
        # Now scan forward to properly indent all content in the else block
        current_line = else_line_index + 1
        in_else_block = True
        
        while in_else_block and current_line < len(lines):
            line = lines[current_line].rstrip()
            
            # Skip empty lines
            if not line:
                current_line += 1
                continue
            
            # Determine current indentation
            current_indent = len(line) - len(line.lstrip())
            stripped_line = line.strip()
            
            # Check if we're exiting the else block (new code block at same or lower indent than else)
            if current_indent <= else_indent and current_line > else_line_index + 1 and stripped_line and not stripped_line.startswith("#"):
                print(f"Exiting else block at line {current_line + 1}")
                in_else_block = False
                break
            
            # Process based on line content
            if stripped_line.endswith("= ["):  # Start of a list declaration
                # This should be at block indent level (else + 4)
                lines[current_line] = " " * block_indent + line.lstrip()
                print(f"Fixed list declaration at line {current_line + 1}")
            elif stripped_line == "]":  # End of a list
                # Also at block indent level
                lines[current_line] = " " * block_indent + line.lstrip()
                print(f"Fixed list ending at line {current_line + 1}")
            elif stripped_line.startswith("f\""):  # List item
                # Deeper indentation for list items
                lines[current_line] = " " * list_indent + line.lstrip()
                print(f"Fixed list item at line {current_line + 1}")
            elif any(keyword in stripped_line for keyword in ["objectives", "materials", "activities", "assessment", "topic_intro"]):
                # These are block variables - should be at block indent
                lines[current_line] = " " * block_indent + line.lstrip()
                print(f"Fixed block variable at line {current_line + 1}")
            else:
                # Any other content in the else block
                lines[current_line] = " " * block_indent + line.lstrip()
                print(f"Fixed miscellaneous line at {current_line + 1}")
            
            current_line += 1
        
        # Write the fixed content
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Fixed indentation for the entire else block")
        return True
    except Exception as e:
        print(f"Error fixing indentation: {str(e)}")
        return False

if __name__ == "__main__":
    fix_all_blocks_indentation() 