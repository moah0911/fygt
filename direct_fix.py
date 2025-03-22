#!/usr/bin/env python

def direct_fix():
    """Directly fix the specific issue at line 1491 based on visual inspection."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # The issue is that there's an else at line 1491 that's misaligned
        # Looking at the code before it, there should be a matching if statement
        # But the else is improperly indented
        
        # Based on the context, this else appears to be at the wrong indentation level
        # Let's find the line first to confirm
        for i in range(1485, 1495):
            if i < len(lines) and lines[i].strip() == "else:":
                # Found the else line, now find the structure it belongs to
                # Assuming it belongs to an if statement that's some indentation level before
                
                # Let's check the previous lines for an if statement pattern
                # and find the main if-elif chain this else might belong to
                for j in range(i-1, max(0, i-100), -1):
                    if "if ai_subject" in lines[j]:
                        # Found a potential matching if
                        indent_level = len(lines[j]) - len(lines[j].lstrip())
                        print(f"Found potential matching 'if' at line {j+1} with indent {indent_level}")
                        
                        # Now we need to fix the else and its block
                        # First, adjust the else line to match the if's indentation
                        lines[i] = " " * indent_level + "else:\n"
                        
                        # Then adjust all the lines in the else block to have proper indentation
                        current_index = i + 1
                        while current_index < len(lines):
                            line = lines[current_index]
                            if not line.strip():
                                current_index += 1
                                continue
                            
                            # Check if this is still part of the else block
                            current_indent = len(line) - len(line.lstrip())
                            
                            # If indentation is wrong (too deep), adjust it
                            if current_indent > indent_level + 20:  # Arbitrary threshold
                                # Set it to proper indentation for inside an else block
                                lines[current_index] = " " * (indent_level + 4) + line.lstrip()
                                current_index += 1
                            else:
                                # We've likely reached the end of the else block
                                break
                        
                        break
                
                break
        
        # Write the fixed file
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Direct fix applied successfully")
        return True
    except Exception as e:
        print(f"Error applying direct fix: {str(e)}")
        return False

if __name__ == "__main__":
    direct_fix() 