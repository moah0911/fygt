#!/usr/bin/env python

def fix_for_loops():
    """Fix the indentation in the for loops."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Process all for loops in the file starting around line 1516
        start_index = 1510
        end_index = min(start_index + 100, len(lines))
        
        i = start_index
        while i < end_index:
            line = lines[i].rstrip()
            stripped_line = line.strip()
            
            # Check if this is a for loop
            if stripped_line.startswith("for ") and i + 1 < len(lines):
                # Get the indentation level of the for loop
                for_indent = len(line) - len(line.lstrip())
                
                # The next line should be indented more than the for loop
                next_line = lines[i + 1].rstrip()
                next_stripped = next_line.strip()
                
                if next_stripped and not next_line.startswith(" " * (for_indent + 4)):
                    # Fix the indentation of the next line
                    lines[i + 1] = " " * (for_indent + 4) + next_stripped + "\n"
                    print(f"Fixed indentation for line {i + 2}")
            
            i += 1
        
        # Write the corrected content back
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Fixed indentation in for loops")
        return True
    except Exception as e:
        print(f"Error fixing for loops: {str(e)}")
        return False

if __name__ == "__main__":
    fix_for_loops() 