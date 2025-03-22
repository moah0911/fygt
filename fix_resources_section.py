#!/usr/bin/env python

def fix_resources_section():
    """Fix indentation in the resources section."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Find the resources section
        start_marker = "# Add suggested next steps and resources"
        end_marker = "st.write("  # This will find the next st.write after the resources section
        
        # Extract the section
        start_idx = content.find(start_marker)
        if start_idx == -1:
            print("Could not find the resources section start")
            return False
            
        # Find the next st.write after the resources section
        resources_section = content[start_idx:]
        end_idx = resources_section.find(end_marker, 500)  # Look for the next st.write after some reasonable length
        
        if end_idx == -1:
            # If no clear endpoint, go for a fixed length
            end_idx = 2000
        
        resources_section = content[start_idx:start_idx + end_idx]
        
        # Create the properly indented replacement
        base_indent = "                            "  # Based on our previous fixes
        
        # Build the replacement with proper indentation
        replacement = f"""{base_indent}# Add suggested next steps and resources
{base_indent}st.write("### Next Steps and Resources")

{base_indent}# Generate relevant resources based on subject and topic
{base_indent}if ai_subject.lower() in ["math", "mathematics"]:
{base_indent}    resources = [
{base_indent}        {{"name": f"Khan Academy: {{ai_topic}}", "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={{ai_topic.replace(' ', '+')}}"}},
{base_indent}        {{"name": f"Desmos Activities for {{ai_topic}}", "url": f"https://teacher.desmos.com/search?q={{ai_topic.replace(' ', '+')}}"}},
{base_indent}        {{"name": "NCTM Illuminations", "url": "https://illuminations.nctm.org/"}},
{base_indent}        {{"name": f"GeoGebra Materials for {{ai_topic}}", "url": f"https://www.geogebra.org/search/{{ai_topic.replace(' ', '%20')}}"}},
{base_indent}    ]
{base_indent}    online_courses = [
{base_indent}        {{"name": f"Coursera - {{ai_topic}} Courses", "url": f"https://www.coursera.org/search?query={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}        {{"name": f"Khan Academy - {{ai_topic}}", "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={{ai_topic.replace(' ', '+')}}"}},
{base_indent}    ]
{base_indent}elif ai_subject.lower() in ["science", "biology", "chemistry", "physics"]:
{base_indent}    resources = [
{base_indent}        {{"name": f"PhET Interactive Simulations for {{ai_topic}}", "url": f"https://phet.colorado.edu/en/simulations/filter?sort=alpha&view=grid&q={{ai_topic.replace(' ', '+')}}"}}"""
        
        # Replace the problematic section
        new_content = content.replace(resources_section, replacement)
        
        # Write the corrected content back
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("Fixed resources section indentation")
        return True
    except Exception as e:
        print(f"Error fixing resources section: {str(e)}")
        return False

if __name__ == "__main__":
    fix_resources_section() 