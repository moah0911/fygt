#!/usr/bin/env python

def fix_all_indentation_errors(file_path):
    """Fix all indentation errors in the file by directly editing problematic sections."""
    print(f"Fixing indentation errors in {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Fix line 1280 issue (around line 1277-1280)
        if len(lines) >= 1280:
            # Fix specific block at line 1277-1280
            line_1276 = lines[1275] if 1275 < len(lines) else ""
            line_1277 = lines[1276] if 1276 < len(lines) else ""
            line_1278 = lines[1277] if 1277 < len(lines) else ""
            line_1279 = lines[1278] if 1278 < len(lines) else ""
            line_1280 = lines[1279] if 1279 < len(lines) else ""
            
            if "else:" in line_1277 and len(line_1277.strip()) == 5:
                print(f"Found indentation issue at line 1277: {line_1277.strip()}")
                # Replace the problematic lines
                new_lines = []
                new_lines.append(line_1276)  # Keep the line before as is
                
                # Fix the else clause to be properly indented
                if "if not lesson_title:" in line_1276:
                    # This means it should be indented to match the if
                    indent_level = line_1276.index("if")
                    new_lines.append(" " * indent_level + "                else:\n")
                else:
                    # Use a reasonable indentation based on context
                    new_lines.append("                                else:\n")
                
                # Keep the content of the else block, ensuring proper indentation
                if "st.success" in line_1279:
                    success_indent = line_1279.index("st.success")
                    new_lines.append(" " * success_indent + line_1279.strip() + "\n")
                else:
                    new_lines.append(line_1279)
                
                # Remove the duplicate else
                if "else:" in line_1280:
                    # Skip this line as it's a duplicate else
                    pass
                else:
                    new_lines.append(line_1280)
                
                # Replace the lines in the original file
                lines[1275:1280] = new_lines
        
        # Write the corrected content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Indentation errors fixed successfully")
        return True
    except Exception as e:
        print(f"Error fixing indentation: {str(e)}")
        return False

if __name__ == "__main__":
    fix_all_indentation_errors('streamlit_app.py') 