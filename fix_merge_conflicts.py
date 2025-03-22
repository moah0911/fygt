"""
Script to automatically fix merge conflicts in the streamlit_app.py file.

This script identifies merge conflict markers and resolves them
by keeping the HEAD version (our changes) and removing conflict markers.
"""

import os
import re
import sys
import traceback

def fix_merge_conflicts(filename):
    """
    Fix merge conflicts in the given file
    
    Args:
        filename: Path to the file to fix
        
    Returns:
        Number of conflicts resolved
    """
    print(f"Checking {filename} for merge conflicts...")
    
    # Create backup of the file
    backup_file = f"{filename}.merge_backup"
    if not os.path.exists(backup_file):
        print(f"Creating backup of {filename} to {backup_file}")
        try:
            with open(filename, 'r', encoding='utf-8') as src_file:
                content = src_file.read()
            
            with open(backup_file, 'w', encoding='utf-8') as bak_file:
                bak_file.write(content)
                
            print(f"Backup created at {backup_file}")
        except Exception as e:
            print(f"Error creating backup: {e}")
            return 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Find all merge conflict markers
        conflict_pattern = r'<<<<<<< HEAD(.*?)=======(.*?)>>>>>>>'
        conflicts = re.findall(conflict_pattern, content, re.DOTALL)
        num_conflicts = len(conflicts)
        
        if num_conflicts == 0:
            print("No merge conflicts found.")
            return 0
        
        print(f"Found {num_conflicts} merge conflicts.")
        
        # Replace conflicts with the HEAD version
        conflict_marker_pattern = r'<<<<<<< HEAD(.*?)=======(.*?)>>>>>>> [a-z0-9]+'
        resolved_content = re.sub(conflict_marker_pattern, r'\1', content, flags=re.DOTALL)
        
        # Fix any indentation issues that might have been introduced
        lines = resolved_content.split('\n')
        fixed_lines = []
        last_indent = 0
        in_function = False
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                fixed_lines.append('')
                continue
                
            # Check if this is a new function definition
            if line.strip().startswith('def '):
                in_function = True
                last_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
                continue
                
            # Check indentation level
            if in_function and line[0:1].isalpha() and not line.strip().startswith('def '):
                # Likely this should be indented
                indent = ' ' * (last_indent + 4)
                fixed_lines.append(indent + line.strip())
            else:
                fixed_lines.append(line)
                
            # Update last indent level
            if line.strip():
                last_indent = len(line) - len(line.lstrip())
        
        # Write the fixed content back to the file
        with open(filename, 'w', encoding='utf-8') as file:
            file.write('\n'.join(fixed_lines))
        
        print(f"Fixed {num_conflicts} merge conflicts.")
        print(f"Original file backed up to {backup_file}")
        
        return num_conflicts
        
    except Exception as e:
        print(f"Error fixing merge conflicts: {e}")
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    file_to_fix = "streamlit_app.py"
    if len(sys.argv) > 1:
        file_to_fix = sys.argv[1]
    
    print(f"Automatic Merge Conflict Resolver for {file_to_fix}")
    print("-" * 50)
    
    fixed = fix_merge_conflicts(file_to_fix)
    
    if fixed > 0:
        print(f"\nSuccessfully fixed {fixed} merge conflicts.")
        print("Please check the file to ensure it was fixed correctly.")
        print("You can now run the application with: streamlit run streamlit_app.py")
    else:
        print("\nNo merge conflicts fixed.")
        print("There might be other issues with the file that need manual fixing.") 