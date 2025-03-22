import re

def fix_specific_error(file_path, line_num):
    """Fix the duplicate else clause issue at a specific line."""
    print(f"Opening file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find the problematic lines
        start_idx = line_num - 30 if line_num - 30 > 0 else 0
        end_idx = line_num + 10 if line_num + 10 < len(lines) else len(lines)
        
        # Extract the block to examine
        block = ''.join(lines[start_idx:end_idx])
        
        # Look for the pattern of double else clauses
        # Find if-else blocks with inappropriate indentation
        if '            else:' in block and '                # Create the course' in block:
            # Fix the indentation of the second else clause
            corrected_block = block.replace(
                '            else:\n                # Create the course',
                '                else:\n                    # Create the course'
            )
            # Fix any additional indentation issues
            corrected_block = corrected_block.replace('            else:\n                st.error', 
                                                      '                    else:\n                        st.error')
            
            # Replace the block in the original lines
            lines[start_idx:end_idx] = corrected_block.splitlines(True)
        
        # Write back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print(f"Fixed specific error at line {line_num} in {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing specific error: {str(e)}")
        return False

if __name__ == "__main__":
    fix_specific_error('streamlit_app.py', 1123) 