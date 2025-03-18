"""Feedback service for personalized student feedback."""
from edumate.services.gemini_service import GeminiService
import datetime
from typing import List, Dict, Any, Optional


class FeedbackService:
    """Service for generating personalized feedback for students."""
    
    def __init__(self):
        """Initialize the feedback service."""
        self.gemini_service = GeminiService()
    
    def generate_feedback(self, submission, student_history=None, tone="constructive", language=None):
        """Generate personalized feedback for a submission with context awareness.
        
        Args:
            submission: The submission to generate feedback for
            student_history: Previous submissions by the student (optional)
            tone: The tone of the feedback (constructive, encouraging, critical, etc.)
            language: The language to generate feedback in (defaults to English)
        """
        if not submission or not submission.assignment:
            return "Unable to generate feedback for this submission."
        
        # Build context about the student if history is provided
        student_context = ""
        if student_history:
            patterns = self.analyze_student_patterns(student_history)
            student_context = f"""
            Student submission history context:
            - Average score: {patterns.get('average_score', 'N/A')}
            - Consistent strengths: {', '.join(patterns.get('strengths', []))}
            - Areas needing improvement: {', '.join(patterns.get('weaknesses', []))}
            - Learning velocity: {patterns.get('learning_velocity', 'N/A')}
            """
        
        # Use Gemini to generate personalized feedback with context
        prompt = f"""
        Generate personalized, actionable feedback for a student submission.
        
        Assignment: {submission.assignment.title}
        Score: {submission.score} out of {submission.assignment.points}
        
        {student_context}
        
        Tone should be: {tone}
        
        Feedback should include:
        1. Specific strengths of this submission
        2. Clear areas for improvement with actionable suggestions
        3. Connection to learning objectives
        4. Encouragement and growth mindset language
        5. Next steps for the student
        
        Make the feedback concise but impactful, highlighting 2-3 key points.
        """
        
        feedback = self.gemini_service.generate_text(prompt)
        
        # Translate feedback if needed
        if language and language.lower() != "english":
            feedback = self.translate_feedback(feedback, language)
        
        return feedback
    
    def generate_improvement_plan(self, student, course):
        """Generate an improvement plan for a student in a course."""
        if not student or not course:
            return "Unable to generate improvement plan."
        
        # Get all graded submissions for the student in the course
        submissions = self._get_student_submissions(student, course)
        
        if not submissions:
            return "No graded submissions available to generate an improvement plan."
        
        # Analyze submissions to identify patterns
        strengths, weaknesses = self._analyze_submissions(submissions)
        
        # Generate improvement plan
        prompt = f"""
        Generate a personalized improvement plan for {student.get_full_name()} in the course {course.name}.
        
        Student's strengths:
        {self._format_list(strengths)}
        
        Areas needing improvement:
        {self._format_list(weaknesses)}
        
        Create a detailed, actionable improvement plan that:
        1. Acknowledges and builds on the student's strengths
        2. Addresses each area needing improvement with specific strategies
        3. Includes recommended resources for each area
        4. Provides a timeline with milestones
        5. Suggests ways to measure progress
        
        Format the plan in a clear, encouraging way that motivates the student to take action.
        """
        
        improvement_plan = self.gemini_service.generate_text(prompt)
        
        return improvement_plan
    
    def analyze_student_patterns(self, submissions: List[Any]) -> Dict[str, Any]:
        """Analyze a student's submission history to identify patterns.
        
        Args:
            submissions: List of student's previous submissions
            
        Returns:
            Dictionary with pattern analysis including strengths, weaknesses, and trends
        """
        if not submissions:
            return {}
            
        # Extract scores and dates
        scores = []
        dates = []
        feedback_points = []
        
        for sub in submissions:
            if hasattr(sub, 'score') and sub.score is not None:
                scores.append(sub.score)
                
            if hasattr(sub, 'submitted_at'):
                dates.append(sub.submitted_at)
                
            # Extract key points from feedback
            if hasattr(sub, 'feedback') and sub.feedback:
                feedback_points.extend(self._extract_feedback_points(sub.feedback))
        
        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Calculate learning velocity (trend in scores over time)
        learning_velocity = "stable"
        if len(scores) > 2 and len(dates) > 2:
            # Convert dates to days since first submission
            try:
                if isinstance(dates[0], str):
                    base_date = datetime.datetime.fromisoformat(dates[0].replace('Z', '+00:00'))
                    days = [(datetime.datetime.fromisoformat(d.replace('Z', '+00:00')) - base_date).days for d in dates]
                else:
                    base_date = dates[0]
                    days = [(d - base_date).days for d in dates]
                    
                # Simple linear regression to find trend
                if len(days) > 1:
                    import numpy as np
                    slope, _ = np.polyfit(days, scores, 1)
                    
                    if slope > 0.5:
                        learning_velocity = "rapidly improving"
                    elif slope > 0.1:
                        learning_velocity = "improving"
                    elif slope < -0.5:
                        learning_velocity = "declining"
                    elif slope < -0.1:
                        learning_velocity = "slightly declining"
                    else:
                        learning_velocity = "stable"
            except Exception as e:
                learning_velocity = "unable to determine"
                print(f"Error calculating learning velocity: {e}")
        
        # Identify common strengths and weaknesses from feedback
        strengths = []
        weaknesses = []
        
        # Count occurrences of feedback points
        point_counts = {}
        for point in feedback_points:
            if point.get('type') == 'strength':
                point_counts[point.get('text', '')] = point_counts.get(point.get('text', ''), 0) + 1
                
            if point.get('type') == 'weakness':
                point_counts[point.get('text', '')] = point_counts.get(point.get('text', ''), 0) + 1
        
        # Find most common points
        for point, count in sorted(point_counts.items(), key=lambda x: x[1], reverse=True):
            if count > len(submissions) * 0.3:  # At least 30% of submissions
                if any(p.get('text') == point and p.get('type') == 'strength' for p in feedback_points):
                    strengths.append(point)
                if any(p.get('text') == point and p.get('type') == 'weakness' for p in feedback_points):
                    weaknesses.append(point)
        
        return {
            'average_score': avg_score,
            'strengths': strengths[:3],  # Top 3 strengths
            'weaknesses': weaknesses[:3],  # Top 3 weaknesses
            'learning_velocity': learning_velocity,
            'submission_count': len(submissions)
        }
    
    def generate_feedback_with_template(self, submission, template_id, placeholders=None):
        """Generate feedback using a predefined template.
        
        Args:
            submission: The submission to generate feedback for
            template_id: The ID of the template to use
            placeholders: Dictionary of placeholders to fill in the template
            
        Returns:
            Rendered feedback from template
        """
        # Get the template
        template = self._get_feedback_template(template_id)
        if not template:
            return self.generate_feedback(submission)  # Fallback to regular feedback
            
        # Prepare placeholders
        placeholders = placeholders or {}
        default_placeholders = {
            "student_name": getattr(submission.student, "name", "Student"),
            "assignment_title": getattr(submission.assignment, "title", "Assignment"),
            "score": getattr(submission, "score", "N/A"),
            "max_score": getattr(submission.assignment, "points", 100),
            "due_date": getattr(submission.assignment, "due_date", "N/A"),
            "current_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        
        # Merge placeholders
        all_placeholders = {**default_placeholders, **placeholders}
        
        # Render template
        feedback = template.get("content", "")
        for key, value in all_placeholders.items():
            feedback = feedback.replace(f"{{{{{key}}}}}", str(value))
            
        return feedback
    
    def translate_feedback(self, feedback, target_language):
        """Translate feedback to another language.
        
        Args:
            feedback: The feedback text to translate
            target_language: The language to translate to
            
        Returns:
            Translated feedback
        """
        prompt = f"""
        Translate the following feedback to {target_language}. 
        Maintain the tone, meaning, and educational context:
        
        {feedback}
        """
        
        translated_feedback = self.gemini_service.generate_text(prompt)
        return translated_feedback
    
    def check_feedback_tone(self, feedback):
        """Check if feedback has appropriate tone and suggest improvements.
        
        Args:
            feedback: The feedback text to check
            
        Returns:
            Dictionary with tone analysis and suggestions
        """
        prompt = """
        Analyze the tone of this teacher feedback and suggest improvements:
        
        """ + feedback + """
        
        Evaluate for:
        1. Growth mindset language
        2. Constructive criticism
        3. Specificity
        4. Encouragement
        5. Clarity
        6. Cultural sensitivity
        7. Appropriate formality
        
        Return:
        - Tone score (1-10)
        - Strengths
        - Suggested improvements
        - Revised version if score is below 7
        """
        
        analysis = self.gemini_service.generate_text(prompt)
        return analysis
    
    def generate_comparative_feedback(self, submission, previous_submissions):
        """Generate feedback comparing current submission to previous ones."""
        if not submission or not previous_submissions:
            return "Unable to generate comparative feedback."
        
        # Extract relevant information
        current_score = submission.score or 0
        previous_scores = [s.score or 0 for s in previous_submissions]
        avg_previous_score = sum(previous_scores) / len(previous_scores) if previous_scores else 0
        
        # Determine if student is improving
        improving = current_score > avg_previous_score
        
        # Generate comparative feedback
        prompt = f"""
        Generate feedback comparing a student's current submission to their previous ones.
        
        Current submission score: {current_score}
        Average score on previous submissions: {avg_previous_score}
        
        The student is {'improving' if improving else 'not showing improvement'} based on scores.
        
        Provide encouraging, constructive feedback that:
        1. Acknowledges progress or identifies regression
        2. Highlights specific improvements or persistent issues
        3. Suggests next steps for continued improvement
        4. Uses growth mindset language
        5. Is specific and actionable
        
        Keep the feedback concise (3-5 sentences) but impactful.
        """
        
        feedback = self.gemini_service.generate_text(prompt)
        
        return feedback
    
    def _get_student_submissions(self, student, course):
        """Get all graded submissions for a student in a course."""
        if not hasattr(student, 'submissions'):
            return []
        
        # Filter submissions by course and graded status
        return [sub for sub in student.submissions 
                if sub.assignment.course_id == course.id and sub.is_graded]
    
    def _analyze_submissions(self, submissions):
        """Analyze submissions to identify strengths and weaknesses."""
        if not submissions:
            return [], []
        
        # Extract feedback from submissions
        feedback_texts = [sub.feedback for sub in submissions if sub.feedback]
        
        if not feedback_texts:
            return [], []
        
        # Use Gemini to analyze feedback and identify patterns
        prompt = f"""
        Analyze the following feedback from multiple assignments and identify the student's strengths and weaknesses.
        
        Feedback:
        {' '.join(feedback_texts)}
        
        Identify:
        1. Top 3 consistent strengths
        2. Top 3 areas needing improvement
        
        Format your response as two lists: "Strengths:" followed by bullet points, and "Weaknesses:" followed by bullet points.
        """
        
        analysis = self.gemini_service.generate_text(prompt)
        
        # Parse strengths and weaknesses
        strengths = []
        weaknesses = []
        
        try:
            # Extract strengths section
            if "Strengths:" in analysis:
                strengths_section = analysis.split("Strengths:")[1].split("Weaknesses:")[0]
                strengths = [s.strip().strip('- ') for s in strengths_section.strip().split('\n') if s.strip()]
            
            # Extract weaknesses section
            if "Weaknesses:" in analysis:
                weaknesses_section = analysis.split("Weaknesses:")[1]
                weaknesses = [w.strip().strip('- ') for w in weaknesses_section.strip().split('\n') if w.strip()]
        except Exception:
            # Fallback if parsing fails
            return [], []
        
        return strengths, weaknesses
    
    def _get_student_learning_style(self, student):
        """Get a student's learning style preferences."""
        return getattr(student, 'learning_style', ['visual', 'text'])
    
    def _format_list(self, items):
        """Format a list of items as a bullet-point string."""
        if not items:
            return "None identified."
        
        return '\n'.join([f'- {item}' for item in items])
    
    def _extract_feedback_points(self, feedback):
        """Extract key points from feedback text."""
        if not feedback:
            return []
            
        # Use Gemini to extract key points
        prompt = f"""
        Extract key points from this feedback:
        
        {feedback}
        
        For each point, identify if it's a strength or a weakness.
        Format as JSON with 'points' array containing objects with 'text' and 'type' fields.
        """
        
        try:
            result = self.gemini_service.generate_text(prompt)
            import json
            parsed = json.loads(result)
            return parsed.get('points', [])
        except Exception:
            # Fallback to simple extraction
            points = []
            
            # Simple heuristic - look for sentences with positive/negative terms
            positive_terms = ["excellent", "good", "great", "well done", "strength", "strong"]
            negative_terms = ["improve", "needs", "work on", "missing", "lacks", "weak"]
            
            sentences = feedback.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(term in sentence.lower() for term in positive_terms):
                    points.append({"text": sentence, "type": "strength"})
                elif any(term in sentence.lower() for term in negative_terms):
                    points.append({"text": sentence, "type": "weakness"})
                    
            return points
            
    def _get_feedback_template(self, template_id):
        """Get a feedback template by ID."""
        templates = {
            "general": {
                "id": "general",
                "name": "General Feedback",
                "content": """
                Dear {{student_name}},
                
                I've reviewed your submission for {{assignment_title}}. You received a score of {{score}}/{{max_score}}.
                
                Strengths: {{strengths}}
                
                Areas for improvement: {{weaknesses}}
                
                Next steps: {{next_steps}}
                
                Please let me know if you have any questions.
                
                Best regards,
                Your Teacher
                """
            },
            "encouragement": {
                "id": "encouragement",
                "name": "Encouragement",
                "content": """
                Hi {{student_name}},
                
                I want to commend your effort on {{assignment_title}}! Your score was {{score}}/{{max_score}}.
                
                I particularly appreciated: {{strengths}}
                
                To further strengthen your work, consider: {{weaknesses}}
                
                Remember, each assignment is a stepping stone to mastery. I'm here to support your growth!
                
                Keep up the great work,
                Your Teacher
                """
            },
            "improvement": {
                "id": "improvement",
                "name": "Focus on Improvement",
                "content": """
                Hello {{student_name}},
                
                I've reviewed your work on {{assignment_title}} ({{score}}/{{max_score}}).
                
                While there are some good elements such as {{strengths}}, there are key areas that need your attention.
                
                Please focus on:
                {{weaknesses}}
                
                I recommend these specific next steps:
                {{next_steps}}
                
                Let's discuss this during office hours. I'm confident you can make progress!
                
                Regards,
                Your Teacher
                """
            }
        }
        
        return templates.get(template_id) 