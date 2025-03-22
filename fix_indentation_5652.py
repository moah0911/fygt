def fix_indentation():
    # Read the file
    with open('streamlit_app.py', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Create a backup
    with open('streamlit_app.py.bak3', 'w', encoding='utf-8') as backup:
        backup.write(content)
    
    # Split content into lines for processing
    lines = content.split("\n")
    
    # Fix the region around line 5652
    if len(lines) >= 5660:
        # The indentation is severely broken around this area. Let's rewrite this section
        
        # Expected structure based on context:
        # This appears to be closing the "student comparison" section and moving to "longitudinal analysis"
        corrected_lines = [
            "                            if skill_df_data:",
            "                                skill_df = pd.DataFrame(skill_df_data)",
            "                                st.dataframe(skill_df)",
            "                        else:",
            "                            st.warning(\"Unable to generate student comparison. Please ensure selected students have submitted assignments.\")",
            "                else:",
            "                    st.info(\"Please select at least one student to compare.\")",
            "",
            "    with tab3:",
            "",
            "        st.header(\"Longitudinal Analysis\")"
        ]
        
        # Replace lines 5646-5656 with corrected lines
        # (using 0-indexed, so line 5647 is at index 5646)
        for i in range(len(corrected_lines)):
            idx = 5646 + i
            if idx < len(lines):
                lines[idx] = corrected_lines[i]
                print(f"Fixed line {idx + 1}")
    
    # Write the fixed content back to the file
    with open('streamlit_app.py', 'w', encoding='utf-8') as file:
        file.write("\n".join(lines))
    
    print("Indentation fix completed for lines around 5652.")

if __name__ == "__main__":
    fix_indentation() 