#!/usr/bin/env python

def rebuild_garbled_section():
    """Completely rebuild the garbled section in the file."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find the start of the problematic section (around line 1490)
        start_index = -1
        end_index = -1
        
        for i in range(1485, 1495):
            if i < len(lines) and "else:" in lines[i].strip():
                start_index = i
                break
        
        if start_index == -1:
            print("Could not find the starting point")
            return False
        
        # Find end of the garbled section
        for i in range(start_index + 1, start_index + 100):
            if i < len(lines) and 'st.write("### Learning Objectives")' in lines[i]:
                end_index = i
                break
        
        if end_index == -1:
            print("Could not find the end point")
            end_index = start_index + 45  # Estimate based on inspection
        
        # Extract indentation level of the else statement
        else_indent = len(lines[start_index]) - len(lines[start_index].lstrip())
        block_indent = else_indent + 4  # Standard indentation inside else block
        list_indent = block_indent + 4  # Indentation for list items
        
        # Create a proper replacement block
        replacement_block = []
        replacement_block.append(lines[start_index])  # Keep the original else: line
        
        # Add properly indented objectives
        replacement_block.append(" " * block_indent + "objectives = [\n")
        replacement_block.append(" " * list_indent + f"f\"Explain the fundamental concepts and principles of {{ai_topic}}\",\n")
        replacement_block.append(" " * list_indent + f"f\"Apply knowledge of {{ai_topic}} to relevant problems or situations\",\n")
        replacement_block.append(" " * list_indent + f"f\"Analyze and evaluate information related to {{ai_topic}}\"\n")
        replacement_block.append(" " * block_indent + "]\n")
        
        # Add properly indented materials
        replacement_block.append(" " * block_indent + "materials = [\n")
        replacement_block.append(" " * list_indent + f"f\"Textbooks or digital resources covering {{ai_topic}}\",\n")
        replacement_block.append(" " * list_indent + f"f\"Worksheets or handouts with {{ai_topic}} exercises\",\n")
        replacement_block.append(" " * list_indent + f"f\"Visual aids or presentation slides illustrating {{ai_topic}}\"\n")
        replacement_block.append(" " * block_indent + "]\n")
        
        # Add properly indented activities
        replacement_block.append(" " * block_indent + "activities = [\n")
        replacement_block.append(" " * list_indent + f"f\"Begin with an engaging hook related to {{ai_topic}}\",\n")
        replacement_block.append(" " * list_indent + f"f\"Present key concepts of {{ai_topic}} with relevant examples\",\n")
        replacement_block.append(" " * list_indent + f"f\"Guide students through practice activities related to {{ai_topic}}\",\n")
        replacement_block.append(" " * list_indent + f"f\"Have students apply their understanding of {{ai_topic}} independently\"\n")
        replacement_block.append(" " * block_indent + "]\n")
        
        # Add properly indented assessment
        replacement_block.append(" " * block_indent + "assessment = [\n")
        replacement_block.append(" " * list_indent + f"f\"Ask questions throughout the lesson to check understanding of {{ai_topic}}\",\n")
        replacement_block.append(" " * list_indent + f"f\"Have students complete a brief quiz or exit ticket about {{ai_topic}}\",\n")
        replacement_block.append(" " * list_indent + f"f\"Assign homework that reinforces learning about {{ai_topic}}\"\n")
        replacement_block.append(" " * block_indent + "]\n")
        
        # Add properly indented topic_intro
        replacement_block.append(" " * block_indent + f"topic_intro = f\"Begin by connecting {{ai_topic}} to students' prior knowledge or experiences. Ask what they already know about {{ai_topic}} or related concepts. You might use a brief demonstration, video clip, or real-world example to show why {{ai_topic}} is important and relevant to their lives.\"\n")
        
        # Replace the problematic section with the fixed one
        lines[start_index:end_index] = replacement_block
        
        # Write the corrected content back
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("Successfully rebuilt the garbled section")
        return True
    except Exception as e:
        print(f"Error rebuilding section: {str(e)}")
        return False

if __name__ == "__main__":
    rebuild_garbled_section() 