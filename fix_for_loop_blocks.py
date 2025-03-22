#!/usr/bin/env python

def fix_for_loop_blocks():
    """Fix indentation of for loops and their blocks."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Identify the section containing learning objectives and materials
        lesson_section = content[content.find("# Display learning objectives"):content.find("# Add suggested next steps")]
        
        # Create corrected indentation for this section
        corrected_section = """                            # Display learning objectives
                            st.write("### Learning Objectives")
                            st.write("By the end of this lesson, students will be able to:")
                            for i, objective in enumerate(objectives, 1):
                                st.write(f"{i}. {objective}")
                        
                            # Display materials needed
                            st.write("### Materials Needed")
                            for material in materials:
                                st.write(f"- {material}")
                        
                            # Display lesson structure
                            st.write("### Lesson Structure")
                        
                            st.write("**Introduction (10 minutes)**")
                            st.write(topic_intro)
                        
                            st.write("**Main Activity (25 minutes)**")
                            for activity in activities:
                                st.write(f"- {activity}")
                        
                            st.write("**Conclusion (10 minutes)**")
                            st.write(f"- Have students summarize the key points about {ai_topic} in their own words")
                            st.write("- Address any questions or misconceptions that arose during the lesson")
                            st.write(f"- Preview how {ai_topic} connects to upcoming content")
                        
                            # Display assessment
                            st.write("### Assessment")
                            for assess in assessment:
                                st.write(f"- {assess}")"""
        
        # Replace the problematic section
        new_content = content.replace(lesson_section, corrected_section)
        
        # Write the corrected content back
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("Fixed for loop blocks")
        return True
    except Exception as e:
        print(f"Error fixing for loop blocks: {str(e)}")
        return False

if __name__ == "__main__":
    fix_for_loop_blocks() 