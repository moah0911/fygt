#!/usr/bin/env python

def fix_indentation_in_section():
    """Fix indentation for a larger section of the file to ensure consistency."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Process a larger section of the file starting from the topic_intro line
        start_line = -1
        for i in range(1500, 1520):
            if i < len(lines) and "topic_intro" in lines[i]:
                start_line = i
                break
        
        if start_line == -1:
            print("Could not find topic_intro line")
            return False
        
        # Get the base indentation level
        base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        print(f"Base indentation level: {base_indent} spaces")
        
        # Fix all indentation from this point until we exit the block
        current_line = start_line + 1
        end_line = min(current_line + 100, len(lines))
        
        # Keep track of nesting level
        nesting_level = 0
        
        while current_line < end_line:
            line = lines[current_line].rstrip()
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                current_line += 1
                continue
            
            # Check for block exit conditions
            if stripped_line.startswith("if ai_") and nesting_level == 0:
                # Main conditional block for resources
                indent_level = base_indent
                nesting_level = 1
            elif stripped_line.startswith("resources =") and nesting_level == 1:
                # Variable assignment inside conditional
                indent_level = base_indent + 4
                nesting_level = 2
            elif stripped_line.startswith("{") and nesting_level == 2:
                # Dictionary items inside list
                indent_level = base_indent + 8
            elif stripped_line.startswith("}") and nesting_level == 2:
                # End of dictionary item
                indent_level = base_indent + 8
            elif stripped_line == "]" and nesting_level == 2:
                # End of resources list
                indent_level = base_indent + 4
                nesting_level = 1
            elif stripped_line.startswith("online_courses =") and nesting_level == 1:
                # New variable at same level as resources
                indent_level = base_indent + 4
            elif stripped_line.startswith("st.write("):
                # Streamlit function calls
                indent_level = base_indent
            elif stripped_line.startswith("for ") and nesting_level == 0:
                # Main for loops
                indent_level = base_indent
                nesting_level = 1
            elif nesting_level == 1 and stripped_line.startswith("st.write("):
                # Inside a for loop
                indent_level = base_indent + 4
            elif stripped_line.startswith("#"):
                # Comments should have the same indentation as surrounding code
                indent_level = base_indent
            else:
                # Default indentation based on nesting
                indent_level = base_indent + (4 * nesting_level)
            
            # Apply the calculated indentation
            lines[current_line] = " " * indent_level + stripped_line + "\n"
            print(f"Fixed line {current_line + 1} with indent {indent_level}")
            
            # Check for end of nested blocks
            if stripped_line.endswith(":") and "elif" not in stripped_line and "else" not in stripped_line:
                nesting_level += 1
            elif stripped_line == "]" or stripped_line.endswith(";"):
                if nesting_level > 0:
                    nesting_level -= 1
            
            current_line += 1
        
        # Write the corrected content back
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Fixed indentation in the entire section")
        return True
    except Exception as e:
        print(f"Error fixing indentation: {str(e)}")
        return False

if __name__ == "__main__":
    fix_indentation_in_section() 