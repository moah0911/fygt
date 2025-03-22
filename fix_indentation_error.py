#!/usr/bin/env python
import re

def fix_indentation_error(file_path, line_num):
    """Fix indentation error at the specified line in the file."""
    print(f"Fixing indentation error at line {line_num} in {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find the block with the indentation issue
        start_idx = max(0, line_num - 15)
        end_idx = min(len(lines), line_num + 15)
        
        # Extract the problematic block
        problematic_block = ''.join(lines[start_idx:end_idx])
        
        # Fix the indentation - specifically for the line with 'else:' that's misaligned
        fixed_block = re.sub(
            r'(\s+)if enrolled_count > 0:\n(\s+)completion_rate = \(len\(submissions\) / enrolled_count\) \* 100\n\s+else:',
            r'\1if enrolled_count > 0:\n\2completion_rate = (len(submissions) / enrolled_count) * 100\n\2else:',
            problematic_block
        )
        
        # Apply the fix
        lines[start_idx:end_idx] = fixed_block.splitlines(True)
        
        # Write the corrected content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Indentation error fixed successfully")
        return True
    except Exception as e:
        print(f"Error fixing indentation: {str(e)}")
        return False

if __name__ == "__main__":
    fix_indentation_error('streamlit_app.py', 1162) 