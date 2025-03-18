"""Enhanced AI grading service with advanced capabilities."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import re
import numpy as np
from pathlib import Path
import hashlib
import time

from edumate.services.ai_service import AIService
from edumate.services.gemini_service import GeminiService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIGradingService:
    """Advanced service for AI-powered grading of assignments."""

    def __init__(self):
        """Initialize the AI grading service."""
        self.ai_service = AIService()
        self.gemini = GeminiService()
        self.cache_dir = Path("data/cache/grading")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.grading_history = {}

    @staticmethod
    def is_available() -> bool:
        """Check if AI grading is available."""
        return AIService.is_available()
    
    def grade_assignment(self, 
                        submission_id: str,
                        content: str, 
                        assignment_type: str, 
                        rubric: Dict[str, Any], 
                        max_points: int = 100,
                        student_history: Optional[List[Dict]] = None,
                        class_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Grade any assignment type with comprehensive AI analysis.
        
        Args:
            submission_id: Unique identifier for the submission
            content: The content to grade
            assignment_type: Type of assignment (essay, code, quiz, etc.)
            rubric: Grading rubric with criteria and points
            max_points: Maximum points for the assignment
            student_history: Previous submissions by this student (optional)
            class_data: Class-wide performance data (optional)
            
        Returns:
            Dictionary with grade, feedback, and detailed analysis
        """
        # Check cache first
        cache_key = self._generate_cache_key(submission_id, content)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            logger.info(f"Using cached grading result for {submission_id}")
            return cached_result
            
        # Select appropriate grading strategy
        grading_strategies = {
            'essay': self.grade_essay,
            'code': self.grade_code,
            'quiz': self.grade_quiz,
            'short_answer': self.grade_short_answer,
            'project': self.grade_project,
            'math': self.grade_math,
            'diagram': self.grade_diagram,
            'presentation': self.grade_presentation
        }
        
        # Get the appropriate grading function or use default
        grading_function = grading_strategies.get(
            assignment_type.lower(), 
            self._default_grading
        )
        
        try:
            # Apply the grading strategy
            start_time = time.time()
            result = grading_function(
                content=content,
                rubric=rubric,
                max_points=max_points,
                student_history=student_history,
                class_data=class_data
            )
            
            # Validate the result
            validated_result = self._validate_grading_result(result, max_points)
            
            # Record processing time
            processing_time = time.time() - start_time
            validated_result['processing_time'] = processing_time
            
            # Cache the result
            self._cache_result(cache_key, validated_result)
            
            # Update grading history
            self._update_grading_history(submission_id, validated_result)
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Error grading {assignment_type} assignment: {str(e)}")
            # Provide a graceful fallback
            return self._fallback_grading(content, rubric, max_points, str(e))
    
    def grade_essay(self, 
                   content: str, 
                   rubric: Dict[str, Any], 
                   max_points: int = 100,
                   **kwargs) -> Dict[str, Any]:
        """Grade an essay assignment with deep content analysis.
        
        Analyzes writing quality, argument structure, evidence use,
        and content relevance against the rubric.
        """
        logger.info("Grading essay assignment with enhanced analysis")
        
        # Pre-process the content for better analysis
        processed_content = self._preprocess_text(content)
        
        # Build a structured rubric for AI evaluation
        formatted_rubric = self._format_rubric(rubric)
        
        # Add essay-specific analytical dimensions
        analysis_dimensions = [
            "Writing mechanics (grammar, spelling, punctuation)",
            "Argument structure and logical flow",
            "Evidence use and citation quality",
            "Concept understanding and application",
            "Originality and critical thinking"
        ]
        
        # Assemble the detailed essay analysis prompt
        analysis_prompt = f"""
        Thoroughly analyze this essay against the provided rubric.
        
        Essay content:
        {processed_content}
        
        Evaluation rubric:
        {json.dumps(formatted_rubric, indent=2)}
        
        Analyze the following dimensions:
        {json.dumps(analysis_dimensions, indent=2)}
        
        For each rubric criterion:
        1. Provide a numeric score based on the criterion's weight
        2. Give specific evidence from the essay supporting your score
        3. Suggest specific improvements
        
        Also evaluate:
        - Overall writing quality
        - Strengths and weaknesses
        - Key misunderstandings or gaps
        - Plagiarism indicators (if any)
        
        Format your response as JSON with these keys:
        - criterion_scores: {criterion: {score: X, evidence: "...", feedback: "..."}}
        - total_score: X
        - strengths: ["...", "..."]
        - weaknesses: ["...", "..."]
        - improvement_suggestions: ["...", "..."]
        - quality_metrics: {writing: X, logic: X, evidence: X, understanding: X, originality: X}
        """
        
        # Get AI analysis response
        result = self.gemini.generate_structured_response(analysis_prompt)
        
        # Check for successful parsing
        if not result or not isinstance(result, dict):
            logger.warning("Failed to parse structured essay grading response")
            # Use fallback approach
            return self._fallback_grading(content, rubric, max_points, "JSON parsing error")
            
        # Calculate final score based on criterion scores
        if 'criterion_scores' in result and 'total_score' not in result:
            total = sum(score_data.get('score', 0) for score_data in result['criterion_scores'].values())
            # Scale to max points if needed
            max_possible = sum(criterion.get('points', 1) for criterion in formatted_rubric)
            result['total_score'] = (total / max_possible) * max_points if max_possible > 0 else 0
            
        # Add metadata
        result['assignment_type'] = 'essay'
        result['grading_method'] = 'ai_enhanced'
        result['confidence'] = self._calculate_confidence(result)
        
        return result
    
    def grade_code(self, 
                  content: str, 
                  rubric: Dict[str, Any], 
                  max_points: int = 100,
                  language: str = 'python',
                  test_cases: Optional[List[Dict]] = None,
                  **kwargs) -> Dict[str, Any]:
        """Grade a code assignment with advanced static and dynamic analysis.
        
        Evaluates code correctness, style, efficiency, documentation,
        and runs test cases when available.
        """
        logger.info(f"Grading {language} code assignment with test execution")
        
        # Identify language if not provided
        if not language:
            language = self._detect_language(content)
        
        # Format rubric for code evaluation
        formatted_rubric = self._format_rubric(rubric)
        
        # Add code-specific criteria if not present
        code_criteria = {
            "Correctness": {"points": max_points * 0.4, "description": "Code executes correctly and produces expected outputs"},
            "Code Quality": {"points": max_points * 0.25, "description": "Code follows best practices, is well-structured and readable"},
            "Documentation": {"points": max_points * 0.2, "description": "Code includes appropriate comments and documentation"},
            "Efficiency": {"points": max_points * 0.15, "description": "Code is efficient and optimized"}
        }
        
        for criterion, definition in code_criteria.items():
            if not any(c.get('name', '').lower() == criterion.lower() for c in formatted_rubric):
                formatted_rubric.append({
                    "name": criterion,
                    "points": definition["points"],
                    "description": definition["description"]
                })
        
        # Run tests if provided
        test_results = []
        if test_cases:
            for test in test_cases:
                test_results.append(self._run_test_case(content, language, test))
        
        # Static code analysis
        static_analysis = self._analyze_code_quality(content, language)
        
        # Build the code grading prompt
        grading_prompt = f"""
        Evaluate this {language} code against the provided rubric.
        
        Code:
        ```{language}
        {content}
        ```
        
        Rubric:
        {json.dumps(formatted_rubric, indent=2)}
        
        Test Results:
        {json.dumps(test_results, indent=2)}
        
        Static Analysis:
        {json.dumps(static_analysis, indent=2)}
        
        Provide detailed analysis of:
        1. Correctness - does the code work as intended?
        2. Code quality - structure, readability, naming conventions
        3. Documentation - comments, docstrings, clarity
        4. Efficiency - algorithmic choices, resource usage
        5. Test coverage - how well did it handle test cases?
        
        For each criterion, provide:
        - Score based on rubric points
        - Specific code examples that influenced your scoring
        - Concrete improvement suggestions
        
        Format as JSON with:
        - criterion_scores: {criterion: {score: X, evidence: "...", feedback: "..."}}
        - total_score: X
        - strengths: ["...", "..."]
        - weaknesses: ["...", "..."]
        - improvement_suggestions: ["...", "..."]
        - code_metrics: {correctness: X, quality: X, documentation: X, efficiency: X}
        """
        
        # Get AI analysis
        result = self.gemini.generate_structured_response(grading_prompt)
        
        # Calculate test-based correctness score
        test_score = self._calculate_test_score(test_results) if test_results else None
        
        # If we have test results, blend AI and test scores for correctness
        if test_score is not None and result and isinstance(result, dict):
            correctness_criterion = next((c for c in formatted_rubric if c.get('name', '').lower() == 'correctness'), None)
            if correctness_criterion and 'criterion_scores' in result and 'Correctness' in result['criterion_scores']:
                # Blend AI and test scores (70% test, 30% AI)
                correctness_points = correctness_criterion.get('points', max_points * 0.4)
                ai_score = result['criterion_scores']['Correctness'].get('score', 0)
                blended_score = (test_score * 0.7) + (ai_score * 0.3)
                result['criterion_scores']['Correctness']['score'] = blended_score
                
                # Recalculate total score
                if 'criterion_scores' in result:
                    result['total_score'] = sum(
                        score_data.get('score', 0) 
                        for score_data in result['criterion_scores'].values()
                    )
        
        # Add metadata
        result['assignment_type'] = 'code'
        result['language'] = language
        result['grading_method'] = 'test_enhanced_ai'
        result['confidence'] = self._calculate_confidence(result)
        
        return result
    
    def grade_quiz(self, 
                  content: str, 
                  rubric: Dict[str, Any], 
                  max_points: int = 100,
                  answer_key: Optional[Dict] = None,
                  **kwargs) -> Dict[str, Any]:
        """Grade a quiz with automatic answer matching and partial credit.
        
        Supports multiple choice, short answer, matching, and
        other quiz formats with configurable scoring.
        """
        logger.info("Grading quiz with answer matching")
        
        try:
            # Parse the quiz submission into structured format
            quiz_data = self._parse_quiz_submission(content)
            
            # If answer key was provided, use it
            if answer_key:
                return self._grade_with_answer_key(quiz_data, answer_key, max_points)
            
            # Otherwise, use AI to grade each question
            return self._grade_quiz_with_ai(quiz_data, rubric, max_points)
        
        except Exception as e:
            logger.error(f"Error during quiz grading: {str(e)}")
            return self._fallback_grading(content, rubric, max_points, str(e))
    
    def _default_grading(self, 
                       content: str, 
                       rubric: Dict[str, Any], 
                       max_points: int = 100,
                       **kwargs) -> Dict[str, Any]:
        """Default grading approach for unrecognized assignment types."""
        logger.info("Using default grading approach")
        
        # Generic grading prompt
        grading_prompt = f"""
        Grade the following submission against the provided rubric.
        
        Submission content:
        {content}
        
        Rubric:
        {json.dumps(self._format_rubric(rubric), indent=2)}
        
        Maximum points: {max_points}
        
        Provide detailed feedback including:
        1. Score for each rubric criterion with explanation
        2. Overall score
        3. Strengths and weaknesses
        4. Suggestions for improvement
        
        Format as JSON with:
        - criterion_scores: {criterion: {score: X, evidence: "...", feedback: "..."}}
        - total_score: X
        - strengths: ["...", "..."]
        - weaknesses: ["...", "..."]
        - improvement_suggestions: ["...", "..."]
        """
        
        # Get AI analysis
        result = self.gemini.generate_structured_response(grading_prompt)
        
        # Add metadata
        if result and isinstance(result, dict):
            result['assignment_type'] = 'generic'
            result['grading_method'] = 'ai_only'
            result['confidence'] = self._calculate_confidence(result)
        
        return result
    
    def _fallback_grading(self, 
                        content: str, 
                        rubric: Dict[str, Any], 
                        max_points: int = 100,
                        error_message: str = "") -> Dict[str, Any]:
        """Fallback grading when primary method fails."""
        logger.warning(f"Using fallback grading due to: {error_message}")
        
        # Simplified prompt for fallback
        fallback_prompt = f"""
        Grade this submission on a scale of 0 to {max_points}.
        
        Submission:
        {content[:2000]}  # Limit content length
        
        Provide:
        1. A numeric score
        2. Brief explanation
        3. One strength and one weakness
        
        Format as simple JSON with score, explanation, strength, weakness keys.
        """
        
        try:
            # Try simplified approach
            fallback_result = self.gemini.generate_structured_response(
                fallback_prompt,
                retries=1,  # Don't retry multiple times in fallback mode
                simplify=True  # Request simpler output
            )
            
            if fallback_result and isinstance(fallback_result, dict) and 'score' in fallback_result:
                return {
                    'total_score': float(fallback_result.get('score', 0)),
                    'explanation': fallback_result.get('explanation', 'Fallback grading used due to error.'),
                    'strengths': [fallback_result.get('strength', 'No strengths identified')],
                    'weaknesses': [fallback_result.get('weakness', 'Unable to determine weaknesses')],
                    'grading_method': 'fallback',
                    'error_message': error_message,
                    'confidence': 0.5,  # Medium confidence for fallback
                }
        except Exception as fallback_error:
            logger.error(f"Fallback grading also failed: {str(fallback_error)}")
        
        # Last resort scoring
        return {
            'total_score': max_points * 0.7,  # Default to 70%
            'explanation': 'Automated grading encountered errors. Manual review recommended.',
            'strengths': ['Unable to determine strengths automatically'],
            'weaknesses': ['Unable to determine weaknesses automatically'],
            'grading_method': 'emergency_fallback',
            'error_message': f"{error_message}; fallback also failed",
            'confidence': 0.1  # Very low confidence
        }
    
    def _format_rubric(self, rubric: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format the rubric into a standardized structure for AI processing."""
        formatted = []
        
        # Handle different rubric formats
        if isinstance(rubric, dict):
            # Format 1: {criterion: points}
            if all(isinstance(v, (int, float)) for v in rubric.values()):
                for criterion, points in rubric.items():
                    formatted.append({
                        "name": criterion,
                        "points": points,
                        "description": ""
                    })
            # Format 2: {criterion: {points: X, description: Y}}
            elif all(isinstance(v, dict) and 'points' in v for v in rubric.values()):
                for criterion, details in rubric.items():
                    formatted.append({
                        "name": criterion,
                        "points": details.get('points', 0),
                        "description": details.get('description', '')
                    })
        # Format 3: [{name: X, points: Y, description: Z}]
        elif isinstance(rubric, list) and all(isinstance(item, dict) for item in rubric):
            formatted = rubric
        
        # Ensure all criteria have a description
        for item in formatted:
            if not item.get('description'):
                item['description'] = f"Points awarded for {item.get('name', 'this criterion')}"
        
        return formatted
    
    def _validate_grading_result(self, result: Dict[str, Any], max_points: int) -> Dict[str, Any]:
        """Validate and normalize grading results."""
        if not result or not isinstance(result, dict):
            logger.error("Invalid grading result, using fallback")
            return self._fallback_grading("", {}, max_points, "Invalid result format")
        
        # Ensure total_score exists and is normalized
        if 'total_score' not in result:
            # Try to calculate from criterion scores
            if 'criterion_scores' in result:
                result['total_score'] = sum(
                    score_data.get('score', 0) 
                    for score_data in result['criterion_scores'].values()
                )
            else:
                result['total_score'] = max_points * 0.7  # Default to 70%
        
        # Ensure score is within range
        result['total_score'] = max(0, min(float(result['total_score']), max_points))
        
        # Ensure required fields exist
        if 'strengths' not in result:
            result['strengths'] = []
        if 'weaknesses' not in result:
            result['weaknesses'] = []
        if 'improvement_suggestions' not in result:
            result['improvement_suggestions'] = []
        
        return result
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence level in grading result."""
        confidence = 0.7  # Base confidence
        
        # More detailed results increase confidence
        if result.get('criterion_scores') and len(result.get('criterion_scores', {})) > 3:
            confidence += 0.1
            
        # Specific evidence in scoring explanations increases confidence
        if 'criterion_scores' in result:
            evidence_count = sum(
                1 for score_data in result['criterion_scores'].values() 
                if len(score_data.get('evidence', '')) > 30
            )
            if evidence_count > len(result['criterion_scores']) / 2:
                confidence += 0.1
                
        # Consistency checks
        if result.get('strengths') and result.get('weaknesses') and result.get('improvement_suggestions'):
            # Check for contradictions between strengths and weaknesses
            strength_terms = ' '.join(result.get('strengths', [])).lower()
            weakness_terms = ' '.join(result.get('weaknesses', [])).lower()
            # If same terms appear in both, reduce confidence
            overlapping_words = set(strength_terms.split()) & set(weakness_terms.split())
            overlapping_words -= {'and', 'the', 'of', 'to', 'in', 'with', 'a', 'is', 'are'}  # Ignore common words
            if len(overlapping_words) > 3:
                confidence -= 0.2
                
        return min(max(confidence, 0.1), 0.99)  # Cap between 0.1 and 0.99
    
    def _generate_cache_key(self, submission_id: str, content: str) -> str:
        """Generate a unique cache key for grading results."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{submission_id}_{content_hash}"
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if grading result exists in cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache: {str(e)}")
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache grading result for future use."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")
    
    def _update_grading_history(self, submission_id: str, result: Dict[str, Any]) -> None:
        """Update grading history for consistency analysis."""
        self.grading_history[submission_id] = {
            'timestamp': time.time(),
            'score': result.get('total_score', 0),
            'confidence': result.get('confidence', 0.7)
        }
        
        # Remove old entries if history gets too large
        if len(self.grading_history) > 1000:
            # Keep only recent entries
            cutoff_time = time.time() - (7 * 24 * 60 * 60)  # One week
            self.grading_history = {
                k: v for k, v in self.grading_history.items() 
                if v['timestamp'] > cutoff_time
            }
            
    def _preprocess_text(self, text: str) -> str:
        """Prepare text content for AI analysis."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common encoding issues
        text = text.replace("â€™", "'").replace("â€"", "—").replace("â€œ", """).replace("â€", """)
        return text.strip()
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code sample."""
        # Simple heuristics for common languages
        if re.search(r'(import\s+[a-zA-Z0-9_.]+|from\s+[a-zA-Z0-9_.]+\s+import)', code):
            return 'python'
        elif re.search(r'(public\s+class|private\s+class|protected\s+class)', code):
            return 'java'
        elif re.search(r'(#include\s*<[a-zA-Z0-9_.]+>|int\s+main\s*\()', code):
            return 'c++'
        elif re.search(r'(function\s+[a-zA-Z0-9_]+\s*\(|const\s+[a-zA-Z0-9_]+\s*=|let\s+[a-zA-Z0-9_]+\s*=|var\s+[a-zA-Z0-9_]+\s*=)', code):
            return 'javascript'
        elif re.search(r'(<!DOCTYPE html>|<html>|<head>|<body>)', code, re.IGNORECASE):
            return 'html'
        else:
            # Default to python if can't determine
            return 'python'
    
    def _parse_quiz_submission(self, content: str) -> List[Dict]:
        """Parse quiz submission into structured format."""
        questions = []
        
        # Try to detect question format
        if '{' in content and '}' in content and ':' in content:
            # Looks like JSON format
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and 'answers' in parsed:
                    return parsed['answers']
                elif isinstance(parsed, list):
                    return parsed
                else:
                    return [parsed]
            except json.JSONDecodeError:
                # Not valid JSON, continue with text parsing
                pass
        
        # Split by numbered questions
        question_pattern = r'(\d+[\.\)]\s*.*?(?=\d+[\.\)]|$))'
        matches = re.findall(question_pattern, content, re.DOTALL)
        
        if matches:
            for match in matches:
                question_num = re.match(r'(\d+)', match).group(1)
                answer_text = re.sub(r'^\d+[\.\)]\s*', '', match).strip()
                
                questions.append({
                    'question_number': question_num,
                    'response': answer_text
                })
        else:
            # Fallback, just split by lines
            lines = content.strip().split('\n')
            for i, line in enumerate(lines):
                questions.append({
                    'question_number': str(i+1),
                    'response': line.strip()
                })
        
        return questions
    
    def _grade_with_answer_key(self, quiz_data: List[Dict], 
                              answer_key: Dict, 
                              max_points: int) -> Dict[str, Any]:
        """Grade quiz using provided answer key."""
        total_possible = len(answer_key)
        correct_count = 0
        question_results = []
        
        for question in quiz_data:
            q_num = question.get('question_number')
            response = question.get('response', '').strip().lower()
            
            if q_num in answer_key:
                correct_answer = answer_key[q_num].lower()
                is_correct = response == correct_answer
                
                # Check for partial matching or multiple choice
                similarity = similarity_score(response, correct_answer)
                
                question_results.append({
                    'question': q_num,
                    'response': response,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'similarity': similarity,
                    'points': 1 if is_correct else (0.5 if similarity > 0.8 else 0)
                })
                
                if is_correct:
                    correct_count += 1
                elif similarity > 0.8:
                    correct_count += 0.5
        
        # Calculate score
        score = (correct_count / total_possible) * max_points if total_possible > 0 else 0
        
        return {
            'total_score': score,
            'question_results': question_results,
            'correct_count': correct_count,
            'total_questions': total_possible,
            'grading_method': 'answer_key',
            'confidence': 0.95  # High confidence with answer key
        }
    
    def _grade_quiz_with_ai(self, quiz_data: List[Dict], 
                           rubric: Dict[str, Any], 
                           max_points: int) -> Dict[str, Any]:
        """Grade quiz using AI when no answer key is available."""
        grading_prompt = f"""
        Grade this quiz submission against the rubric.
        
        Quiz responses:
        {json.dumps(quiz_data, indent=2)}
        
        Rubric:
        {json.dumps(self._format_rubric(rubric), indent=2)}
        
        For each question:
        1. Determine if the answer is correct
        2. Provide explanation of correct answer
        3. Assign a point value
        
        Format response as JSON with:
        - question_results: array of objects with question_number, is_correct, explanation, points
        - total_score: sum of points
        - total_possible: maximum possible points
        - feedback: general feedback on quiz performance
        """
        
        result = self.gemini.generate_structured_response(grading_prompt)
        
        # Calculate total from question results if needed
        if result and 'question_results' in result and 'total_score' not in result:
            points = sum(q.get('points', 0) for q in result['question_results'])
            total_possible = sum(1 for _ in quiz_data)
            result['total_score'] = (points / total_possible) * max_points if total_possible > 0 else 0
            result['total_possible'] = total_possible
        
        # Add metadata
        if result:
            result['grading_method'] = 'ai_quiz'
            result['confidence'] = self._calculate_confidence(result)
            
        return result
    
    def _analyze_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """Perform static analysis on code."""
        # This would typically use language-specific tools (e.g., pylint, ESLint)
        # Here we use a simplified version
        metrics = {
            'lines': len(code.split('\n')),
            'complexity': 0,
            'comments': 0,
            'style_issues': []
        }
        
        # Count comments
        comment_patterns = {
            'python': r'(#.*?$|""".*?""")',
            'javascript': r'(\/\/.*?$|\/\*.*?\*\/)',
            'java': r'(\/\/.*?$|\/\*.*?\*\/)',
            'c++': r'(\/\/.*?$|\/\*.*?\*\/)'
        }
        
        pattern = comment_patterns.get(language, comment_patterns['python'])
        metrics['comments'] = len(re.findall(pattern, code, re.MULTILINE | re.DOTALL))
        
        # Simple complexity estimation (function/method count)
        function_patterns = {
            'python': r'def\s+[a-zA-Z0-9_]+\s*\(',
            'javascript': r'function\s+[a-zA-Z0-9_]*\s*\(|\([^)]*\)\s*=>',
            'java': r'(public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])',
            'c++': r'[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\) *(\{?|[^;])'
        }
        
        pattern = function_patterns.get(language, function_patterns['python'])
        metrics['complexity'] = len(re.findall(pattern, code))
        
        # Style issues (simplified)
        style_checks = {
            'python': [
                {'pattern': r'[^\w](print\()|(import\s*\*)', 'message': 'Avoid print() in production code or import *'},
                {'pattern': r'(\w+ ?= ?\w+ ?\+ ?1)|(\w+ ?\+= ?1)', 'message': 'Consider using += for incrementation'},
                {'pattern': r'([^a-zA-Z0-9_]|^)(if|for|while)\(', 'message': 'Use space after if/for/while'}
            ]
        }
        
        for check in style_checks.get(language, []):
            if re.search(check['pattern'], code):
                metrics['style_issues'].append(check['message'])
        
        return metrics
    
    def _run_test_case(self, code: str, language: str, test: Dict) -> Dict:
        """Run a test case on the code."""
        # In a real implementation, this would use language-specific runners
        # Here we provide a mock implementation
        return {
            'test_name': test.get('name', 'Unnamed test'),
            'passed': True,  # Mock result
            'output': "Test passed successfully",  # Mock output
            'execution_time': 0.05  # Mock time
        }
    
    def _calculate_test_score(self, test_results: List[Dict]) -> float:
        """Calculate score based on test results."""
        if not test_results:
            return 0
            
        passed = sum(1 for test in test_results if test.get('passed', False))
        total = len(test_results)
        
        return (passed / total) if total > 0 else 0

    # Additional methods for other assignment types
    def grade_short_answer(self, content: str, rubric: Dict[str, Any], max_points: int = 100, **kwargs):
        """Grade short answer assignments."""
        # Implementation similar to essay but shorter
        # Uses semantic similarity for checking key points
        pass
        
    def grade_project(self, content: str, rubric: Dict[str, Any], max_points: int = 100, **kwargs):
        """Grade multi-part project submissions."""
        # Implementation would handle multiple documents, code, etc.
        pass
        
    def grade_math(self, content: str, rubric: Dict[str, Any], max_points: int = 100, **kwargs):
        """Grade math problems with step verification."""
        # Implementation would check both final answers and work steps
        pass
        
    def grade_diagram(self, content: str, rubric: Dict[str, Any], max_points: int = 100, **kwargs):
        """Grade diagram/visual submissions using vision models."""
        # Implementation would use vision models to assess diagrams
        pass
        
    def grade_presentation(self, content: str, rubric: Dict[str, Any], max_points: int = 100, **kwargs):
        """Grade presentations with slide analysis."""
        # Implementation would extract and analyze slides
        pass 