def fix_indentation():
    # Read the file
    with open('streamlit_app.py', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Create a backup
    with open('streamlit_app.py.bak_5648', 'w', encoding='utf-8') as backup:
        backup.writelines(lines)
    
    # The specific issue is at line 5648 - an indentation error after an if statement
    # Let's directly fix that specific area
    if len(lines) >= 5650:
        # Look at line 5647 (if statement) and 5648 (the line with the indentation error)
        if_line = lines[5647] if len(lines) > 5647 else ""
        block_line = lines[5648] if len(lines) > 5648 else ""
        
        # Check if we found the right lines
        if "if skill_df_data" in if_line and "skill_df_data" in block_line:
            print(f"Found the problematic lines:")
            print(f"Line 5648: {if_line.strip()}")
            print(f"Line 5649: {block_line.strip()}")
            
            # Fix indentation - properly indent line 5648
            current_indent = len(if_line) - len(if_line.lstrip())
            proper_indent = current_indent + 4
            
            # Create the fixed lines with proper indentation
            if "if skill_df_data:" in if_line:
                fixed_indent = " " * proper_indent
                content = block_line.strip()
                fixed_line = f"{fixed_indent}{content}\n"
                
                # Replace the original problematic line
                lines[5648] = fixed_line
                print(f"Fixed line 5649 with proper indentation")
                
                # Check next line as well - it might need similar fixing
                if len(lines) > 5649:
                    next_content = lines[5649].strip()
                    if next_content:
                        lines[5649] = f"{fixed_indent}{next_content}\n"
                        print(f"Fixed line 5650 with proper indentation")
    
    # Write the fixed content back to the file
    with open('streamlit_app.py', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    
    print("Indentation fix completed.")

if __name__ == "__main__":
    fix_indentation() 