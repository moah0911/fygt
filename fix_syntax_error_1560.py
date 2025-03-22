#!/usr/bin/env python

def fix_syntax_error_1560():
    """Fix the specific syntax error at line 1560."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find the problematic line around 1560
        target_line = -1
        for i in range(1555, 1565):
            if i < len(lines) and "PhET Interactive Simulations" in lines[i] and "st.write" in lines[i]:
                target_line = i
                break
        
        if target_line == -1:
            print("Could not find the problematic line")
            return False
        
        # This line incorrectly has two statements jammed together
        print(f"Found problematic line {target_line + 1}: {lines[target_line].strip()}")
        
        # Split the line into two separate statements
        line = lines[target_line]
        if "st.write" in line:
            # Split at the st.write
            parts = line.split("st.write")
            if len(parts) >= 2:
                # First part should end with a closing brace and comma
                if not parts[0].rstrip().endswith("},"):
                    parts[0] = parts[0].rstrip() + "},"
                
                # Second part should start with st.write
                corrected_lines = [
                    parts[0] + "\n",
                    "                            ]\n",  # Close the resources list
                    "                            \n",   # Add an empty line
                    "                            st.write" + parts[1] + "\n"  # Add the st.write statement
                ]
                
                # Replace the problematic line with corrected lines
                lines[target_line:target_line+1] = corrected_lines
                
                # Write the corrected content back
                with open('streamlit_app.py', 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                print("Fixed syntax error at line 1560")
                return True
            else:
                print("Could not split the line properly")
                return False
        
        return False
    except Exception as e:
        print(f"Error fixing syntax error: {str(e)}")
        return False

if __name__ == "__main__":
    fix_syntax_error_1560() 