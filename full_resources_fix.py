#!/usr/bin/env python

def full_resources_fix():
    """Complete fix for the resources section by replacing it entirely."""
    try:
        with open('streamlit_app.py', 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Find where the assessment section ends
        assessment_section = '                            for assess in assessment:'
        after_assessment_idx = content.find(assessment_section)
        if after_assessment_idx == -1:
            print("Could not find assessment section")
            return False
            
        # Find where to continue after resource section
        marker_after_resources = "st.write('**Your personalized lesson plan has been generated!**"
        after_resources_idx = content.find(marker_after_resources)
        if after_resources_idx == -1:
            print("Could not find section after resources")
            return False
        
        # Keep everything before the resources and after resources
        before_resources = content[:after_assessment_idx + 100]  # Add some padding to ensure we get past the assessment section
        after_resources = content[after_resources_idx:]
        
        # Get the common indentation
        base_indent = "                            "
        
        # Create the complete resources section with proper indentation
        resources_section = f"""
{base_indent}# Add suggested next steps and resources
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
{base_indent}        {{"name": f"PhET Interactive Simulations for {{ai_topic}}", "url": f"https://phet.colorado.edu/en/simulations/filter?sort=alpha&view=grid&q={{ai_topic.replace(' ', '+')}}"}},
{base_indent}        {{"name": f"HHMI BioInteractive: {{ai_topic}}", "url": f"https://www.biointeractive.org/search?search={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}        {{"name": "NASA Education Resources", "url": "https://www.nasa.gov/education/resources/"}},
{base_indent}        {{"name": f"National Science Foundation: {{ai_topic}}", "url": f"https://www.nsf.gov/search.jsp?query={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}    ]
{base_indent}    online_courses = [
{base_indent}        {{"name": f"EdX - {{ai_topic}} Courses", "url": f"https://www.edx.org/search?q={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}        {{"name": f"Coursera - {{ai_topic}} Courses", "url": f"https://www.coursera.org/search?query={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}    ]
{base_indent}elif ai_subject.lower() in ["english", "language arts", "literature"]:
{base_indent}    resources = [
{base_indent}        {{"name": f"CommonLit: {{ai_topic}}", "url": f"https://www.commonlit.org/en/search?query={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}        {{"name": f"Project Gutenberg: {{ai_topic}}", "url": f"https://www.gutenberg.org/ebooks/search/?query={{ai_topic.replace(' ', '+')}}"}},
{base_indent}        {{"name": "ReadWriteThink", "url": "http://www.readwritethink.org/"}},
{base_indent}        {{"name": f"Poetry Foundation: {{ai_topic}}", "url": f"https://www.poetryfoundation.org/search?query={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}    ]
{base_indent}    online_courses = [
{base_indent}        {{"name": f"EdX - {{ai_topic}} Courses", "url": f"https://www.edx.org/search?q={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}        {{"name": f"Coursera - {{ai_topic}} Courses", "url": f"https://www.coursera.org/search?query={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}    ]
{base_indent}else:
{base_indent}    resources = [
{base_indent}        {{"name": f"Google Scholar: {{ai_topic}}", "url": f"https://scholar.google.com/scholar?q={{ai_topic.replace(' ', '+')}}"}},
{base_indent}        {{"name": f"Khan Academy: {{ai_topic}}", "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={{ai_topic.replace(' ', '+')}}"}},
{base_indent}    ]
{base_indent}    online_courses = [
{base_indent}        {{"name": f"Coursera - {{ai_topic}} Courses", "url": f"https://www.coursera.org/search?query={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}        {{"name": f"EdX - {{ai_topic}} Courses", "url": f"https://www.edx.org/search?q={{ai_topic.replace(' ', '%20')}}"}},
{base_indent}    ]

{base_indent}st.write("**Suggested Next Topics:**")
{base_indent}if ai_subject.lower() in ["math", "mathematics"]:
{base_indent}    next_topics = [
{base_indent}        f"Advanced applications of {ai_topic}",
{base_indent}        f"Historical development of {ai_topic}",
{base_indent}        f"{ai_topic} in real-world contexts"
{base_indent}    ]
{base_indent}elif ai_subject.lower() in ["science", "biology", "chemistry", "physics"]:
{base_indent}    next_topics = [
{base_indent}        f"Current research in {ai_topic}",
{base_indent}        f"Ethical considerations related to {ai_topic}",
{base_indent}        f"Technological applications of {ai_topic}"
{base_indent}    ]
{base_indent}elif ai_subject.lower() in ["english", "language arts", "literature"]:
{base_indent}    next_topics = [
{base_indent}        f"Comparative analysis of {ai_topic}",
{base_indent}        f"Critical perspectives on {ai_topic}",
{base_indent}        f"Creative projects inspired by {ai_topic}"
{base_indent}    ]
{base_indent}elif ai_subject.lower() in ["history", "social studies"]:
{base_indent}    next_topics = [
{base_indent}        f"Modern implications of {ai_topic}",
{base_indent}        f"Diverse perspectives on {ai_topic}",
{base_indent}        f"Primary source analysis related to {ai_topic}"
{base_indent}    ]
{base_indent}else:
{base_indent}    next_topics = [
{base_indent}        f"Advanced study of {ai_topic}",
{base_indent}        f"Interdisciplinary connections to {ai_topic}",
{base_indent}        f"Project-based learning with {ai_topic}"
{base_indent}    ]

{base_indent}for topic in next_topics:
{base_indent}    st.write(f"- {topic}")

{base_indent}st.write("**Recommended Resources:**")
{base_indent}for resource in resources:
{base_indent}    st.write(f"- [{resource['name']}]({resource['url']})")

{base_indent}st.write("**Online Courses:**")
{base_indent}for course in online_courses:
{base_indent}    st.write(f"- [{course['name']}]({course['url']})")
"""
        
        # Find the exact cut point in the assessment section
        assessment_end_marker = 'st.write(f"- {assess}")'
        assessment_end_idx = before_resources.rfind(assessment_end_marker)
        if assessment_end_idx != -1:
            before_resources = before_resources[:assessment_end_idx + len(assessment_end_marker)]
        
        # Combine everything
        new_content = before_resources + resources_section + after_resources
        
        # Write the corrected content back
        with open('streamlit_app.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("Completely replaced resources section")
        return True
    except Exception as e:
        print(f"Error replacing resources section: {str(e)}")
        return False

if __name__ == "__main__":
    full_resources_fix() 