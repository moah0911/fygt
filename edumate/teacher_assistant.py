"""Teacher Assistant module for AI-powered grading and feedback."""
import logging
from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path

from edumate.services.ai_grading import AIGradingService
from edumate.services.feedback_service import FeedbackService
from edumate.services.plagiarism_service import PlagiarismService

logger = logging.getLogger(__name__)

class TeacherAssistant:
    """AI-powered assistant for teachers to automate grading and feedback."""
    
    def __init__(self):
        """Initialize the teacher assistant with required services."""
        self.ai_grading = AIGradingService()
        self.feedback_service = FeedbackService()
        self.plagiarism_service = PlagiarismService() if 'PlagiarismService' in globals() else None
        
        # Initialize submission history tracking
        self.history_dir = Path("data/student_history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
    def grade_assignment(self, 
                         submission_id: str,
                         assignment_text: str, 
                         assignment_type: str = 'essay', 
                         rubric: Optional[Dict] = None,
                         max_points: int = 100,
                         student_id: Optional[str] = None) -> Dict[str, Any]:
        """Grade an assignment using AI and provide detailed feedback.
        
        Args:
            submission_id: Unique identifier for this submission
            assignment_text: The content to grade
            assignment_type: Type of assignment (essay, code, quiz, etc.)
            rubric: Grading criteria (optional)
            max_points: Maximum points for the assignment
            student_id: Student identifier for history tracking (optional)
            
        Returns:
            Dictionary with grade, feedback, and analysis
        """
        logger.info(f"Grading {assignment_type} assignment for submission {submission_id}")
        
        # If no rubric provided, use default for assignment type
        if not rubric:
            rubric = self._get_default_rubric(assignment_type)
            
        # Get student history if student_id provided
        student_history = None
        if student_id:
            student_history = self._get_student_history(student_id)
            
        # Grade the assignment
        grading_result = self.ai_grading.grade_assignment(
            submission_id=submission_id,
            content=assignment_text,
            assignment_type=assignment_type,
            rubric=rubric,
            max_points=max_points,
            student_history=student_history
        )
        
        # Generate personalized feedback
        feedback = self._generate_enhanced_feedback(
            grading_result, 
            assignment_type,
            student_history
        )
        
        # Add feedback to the result
        grading_result['enhanced_feedback'] = feedback
        
        # Update student history if student_id provided
        if student_id:
            self._update_student_history(student_id, grading_result)
            
        return grading_result
    
    def grade_bulk_assignments(self, 
                              assignments: List[Dict],
                              assignment_type: str,
                              rubric: Optional[Dict] = None) -> List[Dict]:
        """Grade multiple assignments in bulk.
        
        Args:
            assignments: List of assignment dictionaries with keys:
                         - submission_id, content, student_id (optional)
            assignment_type: Type of assignment (essay, code, etc.)
            rubric: Grading criteria (optional)
            
        Returns:
            List of grading results
        """
        results = []
        for assignment in assignments:
            result = self.grade_assignment(
                submission_id=assignment['submission_id'],
                assignment_text=assignment['content'],
                assignment_type=assignment_type,
                rubric=rubric,
                student_id=assignment.get('student_id')
            )
            results.append(result)
            
        # Perform comparative analysis
        class_insights = self._analyze_class_performance(results)
        
        # Add class insights to each result
        for result in results:
            result['class_insights'] = class_insights
            
        return results
    
    def check_plagiarism(self, content: str, compare_against: List[str] = None) -> Dict:
        """Check content for plagiarism.
        
        Args:
            content: The content to check
            compare_against: List of other content to compare against (optional)
            
        Returns:
            Dictionary with plagiarism score and details
        """
        if not self.plagiarism_service:
            logger.warning("Plagiarism service not available")
            return {"score": 0, "message": "Plagiarism check not available"}
            
        return self.plagiarism_service.check_plagiarism(content, compare_against)
    
    def generate_test_questions(self, topic: str, difficulty: str = 'medium', 
                               count: int = 5) -> List[Dict]:
        """Generate test questions on a given topic.
        
        Args:
            topic: The subject to generate questions about
            difficulty: Difficulty level (easy, medium, hard)
            count: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        # This would call the AI service to generate questions
        # For now, return a placeholder
        return [{"question": f"Sample question {i} about {topic}", 
                "answer": "Sample answer"} for i in range(count)]
    
    def _generate_enhanced_feedback(self, 
                                   grading_result: Dict, 
                                   assignment_type: str,
                                   student_history: Optional[List] = None) -> str:
        """Generate comprehensive, personalized feedback.
        
        Args:
            grading_result: The grading result dictionary
            assignment_type: Type of assignment
            student_history: Previous submissions by this student (optional)
            
        Returns:
            Formatted feedback string
        """
        # Base feedback components
        strengths = grading_result.get('strengths', [])
        weaknesses = grading_result.get('weaknesses', [])
        suggestions = grading_result.get('improvement_suggestions', [])
        score = grading_result.get('total_score', 0)
        
        # Format differs by assignment type
        if assignment_type == 'code':
            return self._format_code_feedback(grading_result)
        elif assignment_type == 'quiz':
            return self._format_quiz_feedback(grading_result)
        else:
            # Default feedback format
            feedback = [
                f"Your score: {score:.1f}",
                "\nStrengths:",
            ]
            
            for strength in strengths:
                feedback.append(f"- {strength}")
                
            feedback.append("\nAreas for improvement:")
            for weakness in weaknesses:
                feedback.append(f"- {weakness}")
                
            if suggestions:
                feedback.append("\nSuggestions:")
                for suggestion in suggestions:
                    feedback.append(f"- {suggestion}")
                    
            # Add personalized note based on history if available
            if student_history:
                feedback.append("\n" + self._generate_progress_note(grading_result, student_history))
                
            return "\n".join(feedback)
    
    def _format_code_feedback(self, grading_result: Dict) -> str:
        """Format feedback specifically for code assignments."""
        feedback = [
            f"Code Assessment - Score: {grading_result.get('total_score', 0):.1f}",
            "\nCode Quality Analysis:"
        ]
        
        # Add criterion scores
        if 'criterion_scores' in grading_result:
            for criterion, data in grading_result['criterion_scores'].items():
                feedback.append(f"\n{criterion}: {data.get('score', 0):.1f}")
                if 'feedback' in data:
                    feedback.append(f"  {data['feedback']}")
                    
        # Add code-specific metrics
        if 'code_metrics' in grading_result:
            metrics = grading_result['code_metrics']
            feedback.append("\nMetrics:")
            for metric, value in metrics.items():
                feedback.append(f"- {metric}: {value}")
                
        # Add suggestions
        if 'improvement_suggestions' in grading_result:
            feedback.append("\nSuggestions for improvement:")
            for suggestion in grading_result['improvement_suggestions']:
                feedback.append(f"- {suggestion}")
                
        return "\n".join(feedback)
    
    def _format_quiz_feedback(self, grading_result: Dict) -> str:
        """Format feedback specifically for quiz assignments."""
        feedback = [
            f"Quiz Results - Score: {grading_result.get('total_score', 0):.1f}/{grading_result.get('total_possible', 100)}",
            f"Correct answers: {grading_result.get('correct_count', 0)} out of {grading_result.get('total_questions', 0)}"
        ]
        
        # Add question-by-question feedback
        if 'question_results' in grading_result:
            feedback.append("\nQuestion Analysis:")
            for question in grading_result['question_results']:
                status = "✓" if question.get('is_correct', False) else "✗"
                feedback.append(f"\nQ{question.get('question', 'N/A')}: {status}")
                if not question.get('is_correct', False) and 'explanation' in question:
                    feedback.append(f"  Correct answer: {question.get('correct_answer', 'N/A')}")
                    feedback.append(f"  Explanation: {question.get('explanation', '')}")
                    
        return "\n".join(feedback)
    
    def _get_student_history(self, student_id: str) -> List[Dict]:
        """Retrieve historical submissions for a student.
        
        Args:
            student_id: The student identifier
            
        Returns:
            List of previous submission results
        """
        history_file = self.history_dir / f"{student_id}.json"
        if not history_file.exists():
            return []
            
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading student history: {e}")
            return []
    
    def _update_student_history(self, student_id: str, result: Dict):
        """Update student history with new submission result.
        
        Args:
            student_id: The student identifier
            result: The grading result to add to history
        """
        history = self._get_student_history(student_id)
        
        # Add only essential data to avoid large files
        history_entry = {
            'submission_id': result.get('submission_id', 'unknown'),
            'timestamp': result.get('timestamp', ''),
            'assignment_type': result.get('assignment_type', ''),
            'score': result.get('total_score', 0),
            'strengths': result.get('strengths', []),
            'weaknesses': result.get('weaknesses', [])
        }
        
        history.append(history_entry)
        
        # Keep only recent history (last 20 submissions)
        if len(history) > 20:
            history = history[-20:]
            
        try:
            with open(self.history_dir / f"{student_id}.json", 'w') as f:
                json.dump(history, f)
        except Exception as e:
            logger.error(f"Error saving student history: {e}")
    
    def _generate_progress_note(self, current_result: Dict, history: List[Dict]) -> str:
        """Generate a personalized note about student progress.
        
        Args:
            current_result: Current grading result
            history: Previous submissions
            
        Returns:
            Progress note string
        """
        if not history:
            return "This is your first submission. Keep up the good work!"
            
        # Calculate average previous score
        prev_scores = [entry.get('score', 0) for entry in history]
        avg_score = sum(prev_scores) / len(prev_scores) if prev_scores else 0
        current_score = current_result.get('total_score', 0)
        
        # Determine if improving
        if current_score > avg_score + 5:
            return f"Great improvement! Your score of {current_score:.1f} is above your average of {avg_score:.1f}."
        elif current_score < avg_score - 5:
            return f"Your score of {current_score:.1f} is below your average of {avg_score:.1f}. Keep working on the suggestions provided."
            else:
            return f"You're maintaining consistent performance around {avg_score:.1f}. Focus on the suggestions to improve further."
    
    def _analyze_class_performance(self, results: List[Dict]) -> Dict:
        """Analyze performance across multiple submissions.
        
        Args:
            results: List of grading results
            
        Returns:
            Dictionary with class-wide insights
        """
        if not results:
            return {}
            
        # Extract scores
        scores = [r.get('total_score', 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Collect all strengths and weaknesses
        all_strengths = []
        all_weaknesses = []
        
        for result in results:
            all_strengths.extend(result.get('strengths', []))
            all_weaknesses.extend(result.get('weaknesses', []))
            
        # Count occurrences
        strength_counts = {}
        for strength in all_strengths:
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
            
        weakness_counts = {}
        for weakness in all_weaknesses:
            weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
            
        # Find most common
        common_strengths = sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        common_weaknesses = sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'average_score': avg_score,
            'submission_count': len(results),
            'common_strengths': [s[0] for s in common_strengths],
            'common_weaknesses': [w[0] for w in common_weaknesses],
            'score_distribution': {
                'below_60': sum(1 for s in scores if s < 60),
                '60_to_70': sum(1 for s in scores if 60 <= s < 70),
                '70_to_80': sum(1 for s in scores if 70 <= s < 80),
                '80_to_90': sum(1 for s in scores if 80 <= s < 90),
                'above_90': sum(1 for s in scores if s >= 90)
            }
        }
    
    def _get_default_rubric(self, assignment_type: str) -> Dict:
        """Get default rubric for an assignment type.
        
        Args:
            assignment_type: Type of assignment
            
        Returns:
            Default rubric dictionary
        """
        default_rubrics = {
            'essay': {
                'Content': {'points': 40, 'description': 'Quality and relevance of ideas'},
                'Organization': {'points': 20, 'description': 'Logical flow and structure'},
                'Language': {'points': 20, 'description': 'Grammar, vocabulary, and style'},
                'Evidence': {'points': 20, 'description': 'Use of supporting examples and citations'}
            },
            'code': {
                'Correctness': {'points': 40, 'description': 'Code executes correctly'},
                'Style': {'points': 20, 'description': 'Code follows best practices'},
                'Documentation': {'points': 20, 'description': 'Code is well documented'},
                'Efficiency': {'points': 20, 'description': 'Code is optimized and efficient'}
            },
            'quiz': {
                'Accuracy': {'points': 100, 'description': 'Correctness of answers'}
            },
            'short_answer': {
                'Accuracy': {'points': 50, 'description': 'Factual correctness'},
                'Completeness': {'points': 30, 'description': 'Addresses all aspects of the question'},
                'Clarity': {'points': 20, 'description': 'Clear and concise expression'}
            },
            'project': {
                'Requirements': {'points': 30, 'description': 'Met all project requirements'},
                'Implementation': {'points': 30, 'description': 'Quality of implementation'},
                'Creativity': {'points': 20, 'description': 'Creative and innovative approach'},
                'Presentation': {'points': 20, 'description': 'Clear presentation and documentation'}
            }
        }
        
        return default_rubrics.get(assignment_type, {
            'Overall Quality': {'points': 100, 'description': 'Overall quality of submission'}
        })
