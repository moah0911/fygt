#!/usr/bin/env python

def fix_materials_indent():
    """Fix the specific indentation issue with the materials section."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Look for the line with "materials =" with wrong indentation
        materials_line_index = -1
        for i in range(1495, 1500):
            if i < len(lines) and "materials = [" in lines[i]:
                materials_line_index = i
                print(f"Found materials at line {materials_line_index + 1}: {lines[materials_line_index].rstrip()}")
                break
        
        if materials_line_index != -1:
            # Find the correct indentation from the objectives line
            objectives_line_index = -1
            for i in range(materials_line_index - 10, materials_line_index):
                if i >= 0 and "objectives = [" in lines[i]:
                    objectives_line_index = i
                    print(f"Found objectives at line {objectives_line_index + 1}: {lines[objectives_line_index].rstrip()}")
                    break
            
            if objectives_line_index != -1:
                # Get the indentation level of objectives
                objectives_indent = len(lines[objectives_line_index]) - len(lines[objectives_line_index].lstrip())
                
                # Fix materials indentation to match objectives
                lines[materials_line_index] = " " * objectives_indent + lines[materials_line_index].lstrip()
                
                # Check and fix indentation of lines in the materials list
                for i in range(materials_line_index + 1, materials_line_index + 10):
                    if i >= len(lines):
                        break
                    
                    line = lines[i]
                    if line.strip() == "]":
                        # End of list - should be at same indent as materials
                        lines[i] = " " * objectives_indent + line.lstrip()
                        print(f"Fixed closing bracket at line {i+1}")
                        break
                    elif "f\"" in line:
                        # List items - should be indented further
                        lines[i] = " " * (objectives_indent + 4) + line.lstrip()
                        print(f"Fixed list item at line {i+1}")
                
                # Also check and fix activities after materials
                activities_line_index = -1
                for i in range(materials_line_index + 10, materials_line_index + 20):
                    if i < len(lines) and "activities = [" in lines[i]:
                        activities_line_index = i
                        print(f"Found activities at line {activities_line_index + 1}")
                        
                        # Fix activities indentation to match objectives
                        current_indent = len(lines[i]) - len(lines[i].lstrip())
                        if current_indent != objectives_indent:
                            lines[i] = " " * objectives_indent + lines[i].lstrip()
                            print(f"Fixed activities indent at line {i+1}")
                        
                        # Fix items in activities list
                        for j in range(i + 1, i + 10):
                            if j >= len(lines):
                                break
                                
                            line = lines[j]
                            if line.strip() == "]":
                                # End of list
                                lines[j] = " " * objectives_indent + line.lstrip()
                                break
                            elif "f\"" in line:
                                # List item
                                lines[j] = " " * (objectives_indent + 4) + line.lstrip()
                        
                        break
        
        # Write the fixed file
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Fixed materials indentation")
        return True
    except Exception as e:
        print(f"Error fixing materials indentation: {str(e)}")
        return False

if __name__ == "__main__":
    fix_materials_indent() 