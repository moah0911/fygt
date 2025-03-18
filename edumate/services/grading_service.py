"""Grading service for automated assignment grading."""
import re
import json
from edumate.services.gemini_service import GeminiService
from edumate.utils.text_utils import extract_text_from_file, similarity_score
from edumate.utils.code_utils import run_code, check_code_style


class GradingService:
    """Service for automated grading of assignments."""
    
    def __init__(self):
        """Initialize the grading service."""
        self.gemini_service = GeminiService()
    
    def grade_submission(self, submission):
        """Grade a submission based on its assignment type."""
        if not submission or not submission.assignment:
            return None
        
        assignment = submission.assignment
        assignment_type = assignment.assignment_type
        
        # Get content from file if needed
        content = submission.content or ""
        if submission.file_path:
            file_content = extract_text_from_file(submission.file_path)
            if file_content:
                content = file_content
        
        # Grade based on assignment type
        if assignment_type == 'essay':
            return self.grade_essay(submission, content)
        elif assignment_type == 'code':
            return self.grade_code(submission, content)
        elif assignment_type == 'quiz':
            return self.grade_quiz(submission, content)
        elif assignment_type == 'short_answer':
            return self.grade_short_answer(submission, content)
        elif assignment_type == 'project':
            return self.grade_project(submission, content)
        else:
            return None
    
    def grade_essay(self, submission, content):
        """Grade an essay submission."""
        assignment = submission.assignment
        rubric_text = self._get_rubric_text(assignment)
        
        # Enhanced plagiarism detection
        plagiarism_score = 0
        plagiarism_details = {}
        
        try:
            # First check if PlagiarismService is available
            from edumate.services.plagiarism_service import PlagiarismService
            plagiarism_service = PlagiarismService()
            
            # Get other submissions and external reference sources
            other_submissions = [s.content for s in assignment.submissions 
                                if s.id != submission.id and s.content]
            
            # Use dedicated plagiarism service for more accurate detection
            plagiarism_result = plagiarism_service.check_plagiarism(content, other_submissions)
            
            # Extract detailed plagiarism information
            plagiarism_score = plagiarism_result.get("score", 0)
            plagiarism_details = {
                "matches": plagiarism_result.get("matches", []),
                "sources": [m.get("source") for m in plagiarism_result.get("matches", [])],
                "original": plagiarism_result.get("original", True),
                "summary": plagiarism_result.get("summary", "")
            }
            
        except ImportError:
            # Fall back to simpler plagiarism check with Gemini
            if other_submissions:
                plagiarism_result = self.gemini_service.check_plagiarism(content, other_submissions)
                plagiarism_score = self._extract_plagiarism_score(plagiarism_result)
                plagiarism_details = {"summary": plagiarism_result}
        
        # Grade the essay
        grading_result = self.gemini_service.grade_essay(content, rubric_text, assignment.points)
        
        # Extract score and feedback
        score = self._extract_score(grading_result, assignment.points)
        
        # Apply more nuanced plagiarism penalty based on severity
        if plagiarism_score > 0.7:  # Severe plagiarism (>70%)
            penalty = score * 0.8  # 80% penalty
            score = max(0, score - penalty)
            grading_result += "\n\nWARNING: Severe plagiarism detected. Score significantly reduced."
        elif plagiarism_score > 0.5:  # Significant plagiarism (50-70%)
            penalty = score * 0.6  # 60% penalty
            score = max(0, score - penalty)
            grading_result += "\n\nWARNING: Significant plagiarism detected. Score substantially reduced."
        elif plagiarism_score > 0.3:  # Moderate plagiarism (30-50%)
            penalty = score * 0.4  # 40% penalty
            score = max(0, score - penalty)
            grading_result += "\n\nWARNING: Moderate plagiarism detected. Score reduced."
        elif plagiarism_score > 0.1:  # Minor plagiarism (10-30%)
            penalty = score * 0.2  # 20% penalty
            score = max(0, score - penalty)
            grading_result += "\n\nNote: Some similarity to other sources detected. Minor score reduction applied."
        
        # Add plagiarism details to feedback
        if plagiarism_score > 0.1:
            grading_result += f"\n\nPLAGIARISM ANALYSIS:\n"
            grading_result += f"Similarity score: {plagiarism_score:.2f} (0-1 scale)\n"
            
            if "summary" in plagiarism_details:
                grading_result += f"Analysis: {plagiarism_details['summary']}\n"
                
            if "sources" in plagiarism_details and plagiarism_details["sources"]:
                grading_result += f"Similar sources: {', '.join(plagiarism_details['sources'][:3])}\n"
                
            if "matches" in plagiarism_details and plagiarism_details["matches"]:
                grading_result += "\nMatched content examples:\n"
                for i, match in enumerate(plagiarism_details["matches"][:2]):  # Show top 2 matches
                    if "matched_text" in match:
                        grading_result += f"{i+1}. \"{match.get('matched_text', '')}...\"\n"
        
        # Update submission
        submission.score = score
        submission.feedback = grading_result
        submission.plagiarism_score = plagiarism_score
        submission.plagiarism_details = plagiarism_details  # Store detailed information
        submission.is_graded = True
        
        # Add criterion scores if rubric exists
        if assignment.rubric and assignment.rubric.criteria:
            self._add_criterion_scores(submission, grading_result)
        
        return submission
    
    def grade_code(self, submission, content):
        """Grade a code submission with enhanced static analysis."""
        assignment = submission.assignment
        rubric_text = self._get_rubric_text(assignment)
        
        # Determine language from file extension or assignment instructions
        language = self._determine_code_language(submission)
        
        # Run the code if possible
        run_result = None
        if language:
            try:
                run_result = run_code(content, language)
            except Exception as e:
                run_result = {
                    "success": False,
                    "error": f"Error executing code: {str(e)}",
                    "output": ""
                }
        
        # Enhanced static analysis
        code_analysis = self._analyze_code(content, language)
        
        # Prepare requirements and test cases
        requirements = assignment.instructions or "No specific requirements provided."
        test_cases = self._extract_test_cases(assignment.instructions)
        
        # Add test results if available
        test_results = {}
        if test_cases and run_result and run_result.get("success", False):
            test_results = self._run_test_cases(content, language, test_cases)
        
        # Grade the code with AI
        detailed_prompt = f"""
        Grade this {language} code submission based on the following rubric:
        {rubric_text}
        
        Code:
        ```{language}
        {content}
        ```
        
        Requirements:
        {requirements}
        
        Static Analysis Results:
        - Complexity: {code_analysis.get('complexity', 'Unknown')}
        - Maintainability: {code_analysis.get('maintainability', 'Unknown')}
        - Security Issues: {', '.join(code_analysis.get('security_issues', ['None detected']))}
        - Style Conformance: {code_analysis.get('style_score', 0)}/10
        - Performance Issues: {', '.join(code_analysis.get('performance_issues', ['None detected']))}
        
        Execution Success: {run_result.get('success', False) if run_result else 'Not executed'}
        
        Test Results: {test_results.get('summary', 'No tests executed')}
        
        Provide a comprehensive evaluation covering:
        1. Code correctness and functionality
        2. Code quality and organization
        3. Algorithm efficiency
        4. Documentation and readability
        5. Error handling
        
        Include specific code snippets that could be improved.
        
        Format with SCORE: X/{assignment.points} at the beginning.
        """
        
        grading_result = self.gemini_service.generate_text(detailed_prompt)
        
        # Extract score and feedback
        score = self._extract_score(grading_result, assignment.points)
        
        # Apply score adjustments based on analysis
        if code_analysis.get('critical_issues', False):
            # Reduce score for critical security issues
            score = max(0, score * 0.7)  # 30% reduction
            grading_result += "\n\nNOTE: Score reduced due to critical security or performance issues."
        
        # Add execution results to feedback
        feedback = grading_result + "\n\n"
        
        feedback += f"CODE ANALYSIS SUMMARY:\n"
        feedback += f"- Complexity: {code_analysis.get('complexity', 'Unknown')}\n"
        feedback += f"- Maintainability: {code_analysis.get('maintainability', 'Unknown')}\n"
        feedback += f"- Style Score: {code_analysis.get('style_score', 0)}/10\n"
        
        if code_analysis.get('security_issues'):
            feedback += f"- Security Issues: {', '.join(code_analysis.get('security_issues'))}\n"
        
        if code_analysis.get('performance_issues'):
            feedback += f"- Performance Issues: {', '.join(code_analysis.get('performance_issues'))}\n"
            
        if run_result:
            feedback += f"\nCODE EXECUTION RESULTS:\n"
            feedback += f"Success: {run_result.get('success', False)}\n"
            feedback += f"Output: {run_result.get('output', '')}\n"
            
            if run_result.get('error'):
                feedback += f"Errors: {run_result.get('error')}\n"
        
        if test_results:
            feedback += f"\nTEST CASE RESULTS:\n"
            feedback += f"Tests Passed: {test_results.get('passed', 0)}/{test_results.get('total', 0)}\n"
            
            if test_results.get('details'):
                feedback += "Details:\n"
                for test, result in test_results.get('details', {}).items():
                    status = "✓" if result.get('passed', False) else "✗"
                    feedback += f"- Test '{test}': {status}\n"
                    if not result.get('passed', False) and result.get('error'):
                        feedback += f"  Error: {result.get('error')}\n"
        
        # Update submission
        submission.score = score
        submission.feedback = feedback
        submission.code_analysis = code_analysis  # Store detailed analysis
        submission.is_graded = True
        
        # Add criterion scores if rubric exists
        if assignment.rubric and assignment.rubric.criteria:
            self._add_criterion_scores(submission, grading_result)
        
        return submission
    
    def grade_quiz(self, submission, content):
        """Grade a quiz submission with enhanced answer matching and partial credit."""
        assignment = submission.assignment
        
        # Try different approaches to parsing quiz answers, ordered by reliability
        parsed_formats = [
            self._parse_quiz_answers_json(content),  # Try JSON format first (most reliable)
            self._parse_quiz_answers_structured(content),  # Try structured format
            self._parse_quiz_answers(content)  # Fall back to basic format
        ]
        
        # Select first successful parsing result
        student_answers = {}
        for parsed in parsed_formats:
            if parsed and len(parsed) > 0:
                student_answers = parsed
                break
                
        # Try the same approaches for correct answers from assignment instructions
        parsed_correct_formats = [
            self._parse_quiz_answers_json(assignment.instructions),
            self._parse_quiz_answers_structured(assignment.instructions),
            self._parse_quiz_answers(assignment.instructions)
        ]
        
        # Select first successful parsing result for correct answers
        correct_answers = {}
        for parsed in parsed_correct_formats:
            if parsed and len(parsed) > 0:
                correct_answers = parsed
                break
        
        if not student_answers or not correct_answers:
            # If parsing fails, use AI to grade the quiz
            grading_result = self.gemini_service.analyze_text(
                content,
                f"""
                Grade this quiz submission against the correct answers.
                
                Quiz questions and answers:
                {assignment.instructions}
                
                Student submission:
                {content}
                
                Provide a detailed breakdown of each question, whether the answer was correct 
                or incorrect, and the final score out of {assignment.points}.
                Format your response with SCORE: X at the beginning.
                """
            )
            score = self._extract_score(grading_result, assignment.points)
            
            # Update submission
            submission.score = score
            submission.feedback = grading_result
            submission.is_graded = True
            return submission
        
        # Initialize feedback and scoring
        feedback = "QUIZ GRADING RESULTS:\n\n"
        total_questions = len(correct_answers)
        total_points = 0
        possible_points = 0
        
        # Prepare detailed results for each question
        question_results = []
        
        # Grade each question
        for question_num, correct_answer in sorted(correct_answers.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            student_answer = student_answers.get(question_num, "")
            question_points = 1  # Default points per question
            
            # Check if the answer has associated points (e.g. "Question 1 (5 points)")
            if isinstance(correct_answer, dict) and 'points' in correct_answer:
                question_points = float(correct_answer['points'])
                correct_answer_value = correct_answer['answer']
            else:
                correct_answer_value = correct_answer
                
            possible_points += question_points
            
            # Calculate score based on answer format
            score, explanation = self._score_answer(student_answer, correct_answer_value)
            question_points_earned = score * question_points
            total_points += question_points_earned
            
            # Determine if answer was fully correct, partially correct, or incorrect
            if score >= 0.99:  # Fully correct
                status = "Correct ✓"
            elif score > 0:  # Partially correct
                status = f"Partially Correct ({score:.0%}) ⚠"
            else:  # Incorrect
                status = "Incorrect ✗"
                
            # Add to feedback
            feedback += f"Question {question_num}: {status}\n"
            if isinstance(student_answer, (list, dict)):
                feedback += f"Your answer: {json.dumps(student_answer)}\n"
            else:
                feedback += f"Your answer: {student_answer}\n"
                
            if isinstance(correct_answer_value, (list, dict)):
                feedback += f"Correct answer: {json.dumps(correct_answer_value)}\n"
            else:
                feedback += f"Correct answer: {correct_answer_value}\n"
                
            if explanation:
                feedback += f"Explanation: {explanation}\n"
                
            feedback += f"Points: {question_points_earned:.2f}/{question_points}\n\n"
            
            # Store detailed result for this question
            question_results.append({
                "question": question_num,
                "student_answer": student_answer,
                "correct_answer": correct_answer_value,
                "score": score,
                "points_earned": question_points_earned,
                "points_possible": question_points,
                "explanation": explanation
            })
        
        # Calculate final score
        if possible_points > 0:
            final_score = (total_points / possible_points) * assignment.points
        else:
            final_score = 0
            
        # Add summary to feedback
        feedback += f"\nFinal Score: {final_score:.2f}/{assignment.points} " + \
                   f"({total_points:.2f}/{possible_points} points earned)"
        
        # Update submission
        submission.score = final_score
        submission.feedback = feedback
        submission.question_results = question_results  # Store detailed results
        submission.is_graded = True
        
        return submission
    
    def grade_short_answer(self, submission, content):
        """Grade a short answer submission."""
        assignment = submission.assignment
        rubric_text = self._get_rubric_text(assignment)
        
        # Use AI to grade short answer
        grading_result = self.gemini_service.analyze_text(
            content,
            f"Grade this short answer response based on the following rubric:\n{rubric_text}\n\n"
            f"Provide a score out of {assignment.points} and detailed feedback."
        )
        
        # Extract score and feedback
        score = self._extract_score(grading_result, assignment.points)
        
        # Update submission
        submission.score = score
        submission.feedback = grading_result
        submission.is_graded = True
        
        # Add criterion scores if rubric exists
        if assignment.rubric and assignment.rubric.criteria:
            self._add_criterion_scores(submission, grading_result)
        
        return submission
    
    def grade_project(self, submission, content):
        """Grade a project submission with milestone tracking and comprehensive evaluation."""
        assignment = submission.assignment
        rubric_text = self._get_rubric_text(assignment)
        
        # Extract milestone data if available
        milestones = self._extract_milestones(assignment.instructions)
        milestone_completions = self._evaluate_milestones(content, milestones)
        
        # Check if this is an update to a previous submission
        previous_submissions = [s for s in assignment.submissions 
                              if s.student_id == submission.student_id and 
                              s.id != submission.id and
                              s.is_graded]
        
        progress_analysis = None
        if previous_submissions:
            # Get the most recent previous submission
            previous_submissions.sort(key=lambda s: s.submitted_at, reverse=True)
            previous = previous_submissions[0]
            
            # Analyze progress since last submission
            progress_analysis = self._analyze_project_progress(
                previous_content=previous.content,
                current_content=content,
                previous_milestone_completions=getattr(previous, 'milestone_completions', None),
                current_milestone_completions=milestone_completions
            )
        
        # Extract project components
        components = self._extract_project_components(content)
        
        # Construct a detailed prompt for AI grading
        detailed_prompt = f"""
        Grade this project submission based on the following rubric:
        {rubric_text}
        
        Project submission:
        ```
        {content[:2000]}  # Limit content length for large projects
        ```
        
        Project has {len(components)} identifiable components:
        {self._format_components(components)}
        
        {"Milestone completion:" if milestones else ""}
        {self._format_milestone_completion(milestone_completions) if milestones else ""}
        
        {"Progress since previous submission:" if progress_analysis else ""}
        {self._format_progress_analysis(progress_analysis) if progress_analysis else ""}
        
        Evaluate the project on:
        1. Completeness and functionality
        2. Design and architecture
        3. Code quality and best practices
        4. Documentation and clarity
        5. Creativity and innovation
        
        For each criterion in the rubric, provide:
        - A score out of the available points
        - Specific evidence supporting your evaluation
        - Constructive feedback for improvement
        
        Begin your response with SCORE: X/{assignment.points}
        
        Then provide a detailed evaluation organized by rubric criteria,
        followed by overall strengths and areas for improvement.
        """
        
        # Use AI to grade project
        grading_result = self.gemini_service.generate_text(detailed_prompt)
        
        # Extract score and feedback
        score = self._extract_score(grading_result, assignment.points)
        
        # Adjust score based on milestone completion if available
        if milestone_completions:
            milestone_score = sum(m.get('completion', 0) * m.get('weight', 1) 
                                 for m in milestone_completions.values())
            milestone_max = sum(m.get('weight', 1) for m in milestone_completions.values())
            
            if milestone_max > 0:
                # Calculate milestone completion percentage
                milestone_pct = milestone_score / milestone_max
                
                # If milestone completion is significantly different from AI score,
                # adjust the score to balance between the two
                ai_score_pct = score / assignment.points
                score_diff = abs(ai_score_pct - milestone_pct)
                
                if score_diff > 0.2:  # More than 20% difference
                    # Weighted average, favoring AI score (70% AI, 30% milestone)
                    adjusted_score = (ai_score_pct * 0.7 + milestone_pct * 0.3) * assignment.points
                    score = adjusted_score
                    grading_result += f"\n\nNote: Score adjusted based on milestone completion ({milestone_pct:.1%})."
        
        # Add progress feedback if available
        if progress_analysis:
            grading_result += "\n\nPROGRESS ANALYSIS:\n"
            
            if progress_analysis.get('improvement_percentage'):
                grading_result += f"- Progress since last submission: {progress_analysis['improvement_percentage']:.1%}\n"
                
            if progress_analysis.get('new_milestones'):
                grading_result += "- Newly completed milestones:\n"
                for milestone in progress_analysis['new_milestones']:
                    grading_result += f"  - {milestone}\n"
                    
            if progress_analysis.get('key_additions'):
                grading_result += "- Key additions since last submission:\n"
                for addition in progress_analysis['key_additions'][:3]:  # Top 3
                    grading_result += f"  - {addition}\n"
        
        # Add milestone details to feedback
        if milestone_completions:
            grading_result += "\n\nMILESTONE COMPLETION:\n"
            
            for milestone_id, milestone_data in milestone_completions.items():
                status = "✓" if milestone_data.get('completion', 0) >= 0.9 else (
                         "⚠" if milestone_data.get('completion', 0) >= 0.5 else "✗")
                grading_result += f"{status} {milestone_data.get('name', f'Milestone {milestone_id}')}: "
                grading_result += f"{milestone_data.get('completion', 0):.0%} complete\n"
                
                if milestone_data.get('feedback'):
                    grading_result += f"  {milestone_data['feedback']}\n"
        
        # Add component analysis to feedback
        if components:
            grading_result += "\n\nCOMPONENT ANALYSIS:\n"
            
            for component, details in components.items():
                quality = details.get('quality', 'N/A')
                grading_result += f"- {component}: {quality}\n"
                if details.get('issues'):
                    grading_result += f"  Issues: {details['issues']}\n"
                if details.get('suggestions'):
                    grading_result += f"  Suggestion: {details['suggestions']}\n"
        
        # Update submission
        submission.score = score
        submission.feedback = grading_result
        submission.is_graded = True
        submission.milestone_completions = milestone_completions
        submission.project_components = components
        
        # Add criterion scores if rubric exists
        if assignment.rubric and assignment.rubric.criteria:
            self._add_criterion_scores(submission, grading_result)
        
        return submission
    
    def _get_rubric_text(self, assignment):
        """Get rubric text from an assignment."""
        if not assignment.rubric:
            return "No rubric provided."
        
        rubric_text = f"Rubric: {assignment.rubric.name}\n\n"
        
        for criterion in assignment.rubric.criteria:
            rubric_text += f"- {criterion.name} ({criterion.max_score} points): {criterion.description}\n"
        
        return rubric_text
    
    def _extract_score(self, grading_result, max_score):
        """Extract numerical score from grading result text."""
        if not grading_result:
            return 0
        
        # Look for score in format "SCORE: X" or "Score: X/Y"
        score_match = re.search(r'SCORE:\s*(\d+(?:\.\d+)?)', grading_result, re.IGNORECASE)
        if score_match:
            try:
                return min(float(score_match.group(1)), max_score)
            except ValueError:
                pass
        
        # Look for score in format "X/Y"
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', grading_result)
        if score_match:
            try:
                score = float(score_match.group(1))
                out_of = float(score_match.group(2))
                return (score / out_of) * max_score if out_of > 0 else 0
            except ValueError:
                pass
        
        # If no score found, use AI to extract it
        score_prompt = f"Extract only the numerical score from this grading result. Respond with just the number:\n\n{grading_result}"
        score_text = self.gemini_service.generate_text(score_prompt)
        
        try:
            score = float(score_text.strip())
            return min(score, max_score)
        except ValueError:
            # Default to 70% if we can't extract a score
            return max_score * 0.7
    
    def _extract_plagiarism_score(self, plagiarism_result):
        """Extract plagiarism score from plagiarism check result."""
        if not plagiarism_result:
            return 0
        
        # Look for percentage in format "PLAGIARISM SCORE: X%" or "X% plagiarism"
        score_match = re.search(r'PLAGIARISM SCORE:\s*(\d+(?:\.\d+)?)', plagiarism_result, re.IGNORECASE)
        if score_match:
            try:
                return min(float(score_match.group(1)) / 100, 1.0)
            except ValueError:
                pass
        
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*plagiarism', plagiarism_result, re.IGNORECASE)
        if score_match:
            try:
                return min(float(score_match.group(1)) / 100, 1.0)
            except ValueError:
                pass
        
        # If no score found, use AI to extract it
        score_prompt = f"Extract only the plagiarism percentage from this result. Respond with just the number (0-100):\n\n{plagiarism_result}"
        score_text = self.gemini_service.generate_text(score_prompt)
        
        try:
            score = float(score_text.strip())
            return min(score / 100, 1.0)
        except ValueError:
            return 0
    
    def _add_criterion_scores(self, submission, grading_result):
        """Add criterion scores based on grading result."""
        from edumate.models.submission import CriterionScore
        
        assignment = submission.assignment
        if not assignment.rubric or not assignment.rubric.criteria:
            return
        
        # Clear existing criterion scores
        for cs in submission.criterion_scores:
            cs.delete()
        
        # For each criterion, try to extract a score from the grading result
        for criterion in assignment.rubric.criteria:
            # Look for the criterion name in the grading result
            pattern = rf'{re.escape(criterion.name)}.*?(\d+(?:\.\d+)?)\s*/\s*{criterion.max_score}'
            score_match = re.search(pattern, grading_result, re.IGNORECASE | re.DOTALL)
            
            if score_match:
                try:
                    score = min(float(score_match.group(1)), criterion.max_score)
                except ValueError:
                    score = criterion.max_score * 0.7  # Default to 70%
            else:
                # If no specific score found, allocate proportionally to overall score
                score = (submission.score / assignment.points) * criterion.max_score
            
            # Create criterion score
            criterion_score = CriterionScore(
                submission_id=submission.id,
                criterion_id=criterion.id,
                score=score
            )
            criterion_score.save()
    
    def _determine_code_language(self, submission):
        """Determine the programming language of a code submission."""
        if submission.file_path:
            ext = submission.file_path.split('.')[-1].lower()
            language_map = {
                'py': 'python',
                'java': 'java',
                'cpp': 'cpp',
                'c': 'c',
                'js': 'javascript',
                'html': 'html',
                'css': 'css',
                'php': 'php'
            }
            return language_map.get(ext)
        
        # Try to determine from assignment instructions
        if submission.assignment.instructions:
            instructions = submission.assignment.instructions.lower()
            for lang in ['python', 'java', 'cpp', 'c', 'javascript', 'html', 'css', 'php']:
                if lang in instructions:
                    return lang
        
        # Default to Python
        return 'python'
    
    def _extract_test_cases(self, instructions):
        """Extract test cases from assignment instructions."""
        if not instructions:
            return []
        
        test_cases = []
        
        # Look for test cases section
        test_section_match = re.search(r'test cases?:(.*?)(?:\n\n|$)', instructions, re.IGNORECASE | re.DOTALL)
        if test_section_match:
            test_section = test_section_match.group(1)
            # Extract individual test cases (lines starting with - or * or number)
            case_matches = re.findall(r'(?:^|\n)\s*(?:-|\*|\d+\.)\s*(.*?)(?:\n|$)', test_section)
            test_cases.extend(case_matches)
        
        return test_cases
    
    def _parse_quiz_answers(self, content):
        """Parse quiz answers from content."""
        if not content:
            return {}
        
        answers = {}
        
        # Look for answers in format "1. A" or "Question 1: A"
        answer_matches = re.findall(r'(?:Question\s*)?(\d+)[\.:]\s*([A-D])', content, re.IGNORECASE)
        for question_num, answer in answer_matches:
            answers[question_num] = answer
        
        return answers
    
    def _parse_quiz_answers_json(self, content):
        """Parse quiz answers in JSON format."""
        if not content:
            return {}
            
        # Look for JSON block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{(?:[^{}]|"[^"]*")*\}', content, re.DOTALL)
            
        if json_match:
            try:
                json_str = json_match.group(1) if hasattr(json_match, 'group') and json_match.group(1) else json_match.group(0)
                answers = json.loads(json_str)
                
                # Validate that it's a dict with question numbers as keys
                if isinstance(answers, dict):
                    # Convert numeric keys to strings if needed
                    return {str(k): v for k, v in answers.items()}
            except:
                pass
                
        return {}
        
    def _parse_quiz_answers_structured(self, content):
        """Parse quiz answers in a structured format like Markdown or formatted text."""
        if not content:
            return {}
            
        answers = {}
        
        # Match patterns like "1. Answer: B" or "Question 1: B"
        patterns = [
            r'(?:Question\s*)?(\d+)[\.:].*?(?:Answer|answer|ANSWER)[:\s]+([A-Za-z0-9].+?)(?=(?:\n\s*\d+[\.:])|\Z)',
            r'(?:Question\s*)?(\d+)[\.:].*?(?:\n\s*(?:Answer|answer|ANSWER)[:\s]+([A-Za-z0-9].+?)(?=\n|\Z))',
            r'(?:^|\n)(\d+)[\.:\)]\s*([A-Da-d])\s*(?:\n|\Z)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for question_num, answer in matches:
                # Clean up the answer
                clean_answer = answer.strip()
                
                # For multiple choice, extract just the letter
                mc_match = re.match(r'^([A-Da-d])[\.:\)]', clean_answer)
                if mc_match:
                    clean_answer = mc_match.group(1).upper()
                    
                answers[question_num] = clean_answer
                
        return answers
    
    def _score_answer(self, student_answer, correct_answer):
        """Score an answer with support for partial credit.
        
        Returns:
            tuple: (score as float 0-1, explanation text)
        """
        # Handle empty answers
        if not student_answer:
            return 0, "No answer provided"
            
        # Normalize answers to strings for comparison
        if isinstance(student_answer, (list, dict)) and isinstance(correct_answer, (list, dict)):
            # Handle complex answer types (lists, dicts)
            try:
                import json
                student_json = json.dumps(student_answer, sort_keys=True)
                correct_json = json.dumps(correct_answer, sort_keys=True)
                
                if student_json == correct_json:
                    return 1.0, "Exactly correct"
                else:
                    # For list answers, check overlap
                    if isinstance(student_answer, list) and isinstance(correct_answer, list):
                        common = set(str(x).lower() for x in student_answer) & set(str(y).lower() for y in correct_answer)
                        total = set(str(y).lower() for y in correct_answer)
                        
                        if not total:
                            return 0, "Invalid answer format"
                            
                        overlap = len(common) / len(total)
                        if overlap > 0:
                            return overlap, f"Partially correct ({len(common)}/{len(total)} items correct)"
                        else:
                            return 0, "No correct items"
                    
                    # For dict answers, compare keys and values
                    elif isinstance(student_answer, dict) and isinstance(correct_answer, dict):
                        # Count matching key-value pairs
                        matches = 0
                        for key, value in correct_answer.items():
                            if key in student_answer and str(student_answer[key]).lower() == str(value).lower():
                                matches += 1
                                
                        if not correct_answer:
                            return 0, "Invalid answer format"
                            
                        overlap = matches / len(correct_answer)
                        if overlap > 0:
                            return overlap, f"Partially correct ({matches}/{len(correct_answer)} items correct)"
                        else:
                            return 0, "No correct items"
                    
                    return 0, "Incorrect answer format"
            except Exception:
                # Fall back to string comparison if JSON comparison fails
                student_str = str(student_answer).lower()
                correct_str = str(correct_answer).lower()
                return 1.0 if student_str == correct_str else 0, "Format error, compared as strings"
        
        # For multiple choice, match letters exactly (case-insensitive)
        if isinstance(correct_answer, str) and len(correct_answer) == 1 and correct_answer.isalpha():
            student_str = str(student_answer).strip().upper()
            correct_str = correct_answer.strip().upper()
            return 1.0 if student_str == correct_str else 0, None
            
        # For numeric answers, allow small differences
        try:
            student_num = float(student_answer)
            correct_num = float(correct_answer)
            
            # Calculate percent difference
            if correct_num == 0:
                # Avoid division by zero
                return 1.0 if student_num == 0 else 0, None
                
            percent_diff = abs(student_num - correct_num) / abs(correct_num)
            
            if percent_diff < 0.01:  # Within 1%
                return 1.0, "Exact match"
            elif percent_diff < 0.05:  # Within 5%
                return 0.8, "Very close (within 5%)"
            elif percent_diff < 0.10:  # Within 10%
                return 0.5, "Close (within 10%)"
            else:
                return 0, "Incorrect numerical answer"
        except (ValueError, TypeError):
            # Not numeric, continue with string comparison
            pass
            
        # For text answers, use similarity
        student_str = str(student_answer).lower().strip()
        correct_str = str(correct_answer).lower().strip()
        
        # Exact match
        if student_str == correct_str:
            return 1.0, "Exactly correct"
            
        # Check for key words/phrases in longer answers
        if len(correct_str.split()) > 3:  # For longer answers
            # Extract key terms from correct answer
            import re
            key_terms = set(re.findall(r'\b\w{4,}\b', correct_str))  # Words with 4+ chars
            student_terms = set(re.findall(r'\b\w{4,}\b', student_str))
            
            if key_terms:
                matches = key_terms.intersection(student_terms)
                coverage = len(matches) / len(key_terms)
                
                if coverage >= 0.8:
                    return 0.9, "Contains most key concepts"
                elif coverage >= 0.6:
                    return 0.7, "Contains many key concepts"
                elif coverage >= 0.4:
                    return 0.5, "Contains some key concepts"
                elif coverage > 0:
                    return 0.2, "Contains few key concepts"
        
        # Use string similarity as a fallback
        try:
            similarity = similarity_score(student_str, correct_str)
            
            if similarity >= 0.9:
                return 0.9, "Very similar to correct answer"
            elif similarity >= 0.7:
                return 0.6, "Somewhat similar to correct answer"
            elif similarity >= 0.5:
                return 0.3, "Slightly similar to correct answer"
        except:
            pass
            
        return 0, "Incorrect"
    
    def _analyze_code(self, code, language):
        """Perform static analysis on code to identify quality, security, and performance issues."""
        # Default analysis result
        analysis = {
            "complexity": "Medium",
            "maintainability": "Good",
            "style_score": 7,
            "security_issues": [],
            "performance_issues": [],
            "critical_issues": False
        }
        
        # Basic code metrics
        line_count = len(code.split("\n"))
        analysis["line_count"] = line_count
        
        # Check for empty code
        if not code.strip():
            analysis["complexity"] = "N/A"
            analysis["maintainability"] = "Poor"
            analysis["style_score"] = 0
            analysis["security_issues"].append("Empty code submission")
            analysis["critical_issues"] = True
            return analysis
            
        # Use style analysis from code_utils
        style_result = None
        try:
            style_result = check_code_style(code, language)
            if style_result and "issues" in style_result:
                if len(style_result["issues"]) > 10:
                    analysis["style_score"] = 3  # Many style issues
                    analysis["maintainability"] = "Poor"
                elif len(style_result["issues"]) > 5:
                    analysis["style_score"] = 5  # Several style issues
                    analysis["maintainability"] = "Fair"
                elif len(style_result["issues"]) > 0:
                    analysis["style_score"] = 7  # Few style issues
                else:
                    analysis["style_score"] = 10  # No style issues
                    analysis["maintainability"] = "Excellent"
        except Exception:
            # Style checking failed, use default values
            pass
            
        # Language-specific analysis
        if language == "python":
            # Check for common Python security issues
            if "eval(" in code or "exec(" in code:
                analysis["security_issues"].append("Use of potentially unsafe eval() or exec()")
                analysis["critical_issues"] = True
                
            if "import os" in code and "os.system(" in code:
                analysis["security_issues"].append("Potentially unsafe system command execution")
                analysis["critical_issues"] = True
                
            # Check for performance issues
            if "+= 1" in code and "range" in code and "for" in code:
                analysis["performance_issues"].append("Possible inefficient loop implementation")
                
            if "while True:" in code and "break" not in code:
                analysis["performance_issues"].append("Infinite loop detected")
                analysis["critical_issues"] = True
                
        elif language in ["java", "c", "cpp"]:
            # Check for common C/C++/Java security issues
            if "strcpy" in code or "strcat" in code:
                analysis["security_issues"].append("Use of unsafe string functions")
                
            if "malloc" in code and "free" not in code:
                analysis["security_issues"].append("Potential memory leak")
                analysis["critical_issues"] = True
                
        elif language == "javascript":
            # Check for common JavaScript issues
            if "eval(" in code:
                analysis["security_issues"].append("Use of potentially unsafe eval()")
                analysis["critical_issues"] = True
                
            if "innerHTML =" in code:
                analysis["security_issues"].append("Potential XSS vulnerability with innerHTML")
                
        # Complexity analysis based on code structure
        if line_count > 200:
            analysis["complexity"] = "Very High"
        elif line_count > 100:
            analysis["complexity"] = "High"
        elif line_count > 50:
            analysis["complexity"] = "Medium"
        else:
            analysis["complexity"] = "Low"
            
        # Check for documentation
        comment_lines = 0
        for line in code.split("\n"):
            if language == "python" and ("#" in line or '"""' in line):
                comment_lines += 1
            elif language in ["java", "c", "cpp", "javascript"] and ("//" in line or "/*" in line):
                comment_lines += 1
                
        comment_ratio = comment_lines / max(1, line_count)
        if comment_ratio < 0.05:
            analysis["maintainability"] = "Poor"  # Very little documentation
            
        # Use AI to enhance the analysis if needed (more comprehensive checks)
        if line_count > 50:  # Only for non-trivial code
            try:
                ai_analysis_prompt = f"""
                Analyze this {language} code for quality, security, and performance issues.
                Respond with a JSON object containing these keys only:
                - complexity: string rating from "Low" to "Very High"
                - maintainability: string rating from "Poor" to "Excellent"
                - security_issues: array of strings (or empty array)
                - performance_issues: array of strings (or empty array)
                - critical_issues: boolean
                
                Code:
                ```{language}
                {code[:1000]}  # Limit for very large submissions
                ```
                
                Format your response as valid JSON only.
                """
                ai_result = self.gemini_service.generate_structured_response(ai_analysis_prompt)
                
                # Merge AI analysis with our basic analysis
                if isinstance(ai_result, dict):
                    if "complexity" in ai_result:
                        analysis["complexity"] = ai_result["complexity"]
                    if "maintainability" in ai_result:
                        analysis["maintainability"] = ai_result["maintainability"]
                    if "security_issues" in ai_result and ai_result["security_issues"]:
                        # Append new issues, avoid duplicates
                        new_issues = [issue for issue in ai_result["security_issues"] 
                                     if issue not in analysis["security_issues"]]
                        analysis["security_issues"].extend(new_issues)
                    if "performance_issues" in ai_result and ai_result["performance_issues"]:
                        new_issues = [issue for issue in ai_result["performance_issues"] 
                                     if issue not in analysis["performance_issues"]]
                        analysis["performance_issues"].extend(new_issues)
                    if "critical_issues" in ai_result:
                        analysis["critical_issues"] = analysis["critical_issues"] or ai_result["critical_issues"]
            except Exception:
                # AI analysis failed, continue with basic analysis
                pass
                
        return analysis
    
    def _run_test_cases(self, code, language, test_cases):
        """Run code against provided test cases and return results."""
        results = {
            "passed": 0,
            "total": len(test_cases),
            "details": {},
            "summary": "No tests executed"
        }
        
        if not test_cases:
            return results
            
        # Execute each test case
        for i, test_case in enumerate(test_cases):
            test_name = f"Test {i+1}"
            test_code = self._prepare_test_code(code, language, test_case)
            
            try:
                test_result = run_code(test_code, language)
                passed = test_result.get("success", False) and not test_result.get("error")
                
                if passed:
                    results["passed"] += 1
                    
                results["details"][test_name] = {
                    "passed": passed,
                    "output": test_result.get("output", ""),
                    "error": test_result.get("error", "")
                }
            except Exception as e:
                results["details"][test_name] = {
                    "passed": False,
                    "output": "",
                    "error": str(e)
                }
                
        # Generate summary
        if results["total"] > 0:
            pass_rate = (results["passed"] / results["total"]) * 100
            results["summary"] = f"{results['passed']}/{results['total']} tests passed ({pass_rate:.1f}%)"
            
        return results
        
    def _prepare_test_code(self, code, language, test_case):
        """Prepare code with test case for execution."""
        if language == "python":
            # Add test case as a Python assertion or function call
            if "assert" in test_case:
                return code + f"\n\n# Test case\n{test_case}"
            else:
                return code + f"\n\n# Test case\nprint({test_case})"
                
        elif language == "java":
            # For Java, we would need more sophisticated test preparation
            # This is a simplified version
            if code.contains("public static void main"):
                # Add output for the test case
                return code.replace("public static void main(String[] args) {", 
                                   f"public static void main(String[] args) {{\n    // Test case\n    System.out.println({test_case});\n")
            else:
                return code
                
        elif language == "javascript":
            # Add test case as console.log statement
            return code + f"\n\n// Test case\nconsole.log({test_case});"
            
        # For other languages, just append the test case
        return code + f"\n\n// Test case\n{test_case}" 

    def grade_submissions_batch(self, submissions, with_comparative=True):
        """Grade multiple submissions in batch for improved efficiency and comparative analysis.
        
        Args:
            submissions: List of submissions to grade
            with_comparative: Whether to include comparative analysis
            
        Returns:
            List of graded submissions with additional comparative insights
        """
        if not submissions:
            return []
            
        # Group submissions by assignment
        assignment_groups = {}
        for submission in submissions:
            if submission.assignment:
                assignment_id = submission.assignment.id
                if assignment_id not in assignment_groups:
                    assignment_groups[assignment_id] = []
                assignment_groups[assignment_id].append(submission)
        
        # Process each assignment group
        all_graded = []
        
        for assignment_id, group in assignment_groups.items():
            # Skip empty groups
            if not group:
                continue
                
            # Get assignment details
            assignment = group[0].assignment
            assignment_type = assignment.assignment_type
            
            # Grade each submission individually
            graded_group = []
            for submission in group:
                # Skip already graded submissions
                if submission.is_graded:
                    graded_group.append(submission)
                    continue
                
                # Grade the submission
                try:
                    graded_submission = self.grade_submission(submission)
                    if graded_submission:
                        graded_group.append(graded_submission)
                except Exception as e:
                    # Log error and continue with next submission
                    logger.error(f"Error grading submission {submission.id}: {str(e)}")
                    submission.feedback = f"Error during grading: {str(e)}"
                    graded_group.append(submission)
            
            # Add comparative analysis if requested and there are multiple submissions
            if with_comparative and len(graded_group) > 1:
                self._add_comparative_analysis(graded_group, assignment)
                
            # Add all graded submissions to result
            all_graded.extend(graded_group)
        
        return all_graded
        
    def _add_comparative_analysis(self, submissions, assignment):
        """Add comparative analysis to graded submissions."""
        if not submissions or len(submissions) < 2:
            return
            
        # Calculate statistics
        scores = [s.score for s in submissions if hasattr(s, 'score') and s.score is not None]
        if not scores:
            return
            
        # Basic statistics
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        # Calculate median
        sorted_scores = sorted(scores)
        mid = len(sorted_scores) // 2
        median_score = sorted_scores[mid] if len(sorted_scores) % 2 == 1 else (sorted_scores[mid-1] + sorted_scores[mid]) / 2
        
        # Calculate standard deviation
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Calculate score distribution
        total = len(scores)
        distribution = {
            'A': sum(1 for s in scores if s >= 0.9 * assignment.points) / total,
            'B': sum(1 for s in scores if 0.8 * assignment.points <= s < 0.9 * assignment.points) / total,
            'C': sum(1 for s in scores if 0.7 * assignment.points <= s < 0.8 * assignment.points) / total,
            'D': sum(1 for s in scores if s < 0.6 * assignment.points) / total
        }
        
        # Create comparative information
        comparative_info = {
            'avg_score': avg_score,
            'median_score': median_score,
            'min_score': min_score,
            'max_score': max_score,
            'std_dev': std_dev,
            'total_submissions': len(submissions),
            'score_distribution': distribution
        }
        
        # Assignment-specific analysis
        if assignment.assignment_type == 'code':
            # Add code-specific analysis
            code_issues = {}
            for submission in submissions:
                if hasattr(submission, 'code_analysis') and submission.code_analysis:
                    # Collect common issues
                    for issue in submission.code_analysis.get('security_issues', []):
                        code_issues[issue] = code_issues.get(issue, 0) + 1
                    for issue in submission.code_analysis.get('performance_issues', []):
                        code_issues[issue] = code_issues.get(issue, 0) + 1
            
            # Add most common issues
            if code_issues:
                sorted_issues = sorted(code_issues.items(), key=lambda x: x[1], reverse=True)
                comparative_info['common_code_issues'] = [
                    {'issue': issue, 'count': count}
                    for issue, count in sorted_issues[:5]  # Top 5 issues
                ]
        
        elif assignment.assignment_type == 'quiz':
            # Add quiz-specific analysis
            question_stats = {}
            for submission in submissions:
                if hasattr(submission, 'question_results') and submission.question_results:
                    for result in submission.question_results:
                        q_id = result.get('question')
                        if q_id not in question_stats:
                            question_stats[q_id] = {
                                'total': 0,
                                'correct': 0,
                                'partial': 0,
                                'incorrect': 0,
                                'avg_score': 0,
                                'scores': []
                            }
                        
                        stats = question_stats[q_id]
                        stats['total'] += 1
                        score = result.get('score', 0)
                        stats['scores'].append(score)
                        
                        if score >= 0.99:
                            stats['correct'] += 1
                        elif score > 0:
                            stats['partial'] += 1
                        else:
                            stats['incorrect'] += 1
            
            # Calculate averages for each question
            for q_id, stats in question_stats.items():
                if stats['scores']:
                    stats['avg_score'] = sum(stats['scores']) / len(stats['scores'])
                    # Remove raw scores to reduce size
                    del stats['scores']
            
            comparative_info['question_statistics'] = question_stats
            
            # Identify most difficult questions (lowest avg score)
            if question_stats:
                sorted_questions = sorted(question_stats.items(), 
                                         key=lambda x: x[1]['avg_score'])
                comparative_info['difficult_questions'] = [
                    {'question': q_id, 'avg_score': stats['avg_score']}
                    for q_id, stats in sorted_questions[:3]  # Top 3 most difficult
                ]
        
        elif assignment.assignment_type == 'essay':
            # Add essay-specific analysis
            plagiarism_stats = {
                'high': 0,
                'medium': 0,
                'low': 0,
                'none': 0
            }
            
            for submission in submissions:
                if hasattr(submission, 'plagiarism_score'):
                    score = submission.plagiarism_score
                    if score > 0.5:
                        plagiarism_stats['high'] += 1
                    elif score > 0.3:
                        plagiarism_stats['medium'] += 1
                    elif score > 0.1:
                        plagiarism_stats['low'] += 1
                    else:
                        plagiarism_stats['none'] += 1
            
            comparative_info['plagiarism_statistics'] = plagiarism_stats
            
            # Extract common themes using AI if there are enough submissions
            if len(submissions) >= 5:
                try:
                    # Extract a sample of contents
                    sample_content = "\n---\n".join([
                        s.content[:500] for s in submissions[:5] if s.content
                    ])
                    
                    # Use AI to identify common themes
                    theme_prompt = f"""
                    Analyze these excerpts from student essays on the assignment: "{assignment.title}".
                    Identify 3-5 common themes, arguments, or approaches that students are using.
                    For each theme, provide a brief description.
                    
                    Format your response as a JSON array of objects with "theme" and "description" keys.
                    
                    Excerpts:
                    {sample_content}
                    """
                    
                    theme_result = self.gemini_service.generate_structured_response(theme_prompt)
                    if theme_result and isinstance(theme_result, list):
                        comparative_info['common_themes'] = theme_result
                    elif isinstance(theme_result, dict) and 'themes' in theme_result:
                        comparative_info['common_themes'] = theme_result['themes']
                except Exception as e:
                    # Silently fail if AI analysis fails
                    pass
        
        # Add comparative information to each submission
        for submission in submissions:
            submission.comparative_analysis = comparative_info
            
            # Add personalized comparative feedback
            if hasattr(submission, 'score') and submission.score is not None:
                percentile = sum(1 for s in scores if s < submission.score) / len(scores)
                
                if percentile > 0.9:
                    comparative_text = f"Your score is in the top 10% of the class (rank: {int(percentile * len(scores) + 1)}/{len(scores)})."
                elif percentile > 0.75:
                    comparative_text = f"Your score is in the top 25% of the class (rank: {int(percentile * len(scores) + 1)}/{len(scores)})."
                elif percentile > 0.5:
                    comparative_text = f"Your score is above the class median (rank: {int(percentile * len(scores) + 1)}/{len(scores)})."
                elif percentile > 0.25:
                    comparative_text = f"Your score is below the class median (rank: {int(percentile * len(scores) + 1)}/{len(scores)})."
                else:
                    comparative_text = f"Your score is in the bottom 25% of the class (rank: {int(percentile * len(scores) + 1)}/{len(scores)})."
                
                submission.feedback += f"\n\nCOMPARATIVE ANALYSIS:\n"
                submission.feedback += f"Class average: {avg_score:.2f}/{assignment.points}\n"
                submission.feedback += f"Class median: {median_score:.2f}/{assignment.points}\n"
                submission.feedback += f"{comparative_text}\n"
                
                # Add specific suggestions based on comparative analysis
                if assignment.assignment_type == 'code' and hasattr(submission, 'code_analysis') and submission.code_analysis:
                    if 'common_code_issues' in comparative_info:
                        submission_issues = (
                            submission.code_analysis.get('security_issues', []) + 
                            submission.code_analysis.get('performance_issues', [])
                        )
                        common_issues = [i['issue'] for i in comparative_info['common_code_issues']]
                        
                        shared_issues = [issue for issue in submission_issues if issue in common_issues]
                        if shared_issues:
                            submission.feedback += "\nYour submission shares these common issues with other students:\n"
                            for issue in shared_issues:
                                submission.feedback += f"- {issue}\n"
                
                elif assignment.assignment_type == 'quiz' and 'difficult_questions' in comparative_info:
                    submission.feedback += "\nMost challenging questions for the class:\n"
                    for q_data in comparative_info['difficult_questions']:
                        q_id = q_data['question']
                        avg = q_data['avg_score']
                        submission.feedback += f"- Question {q_id}: {avg:.0%} class average\n"
        
        return comparative_info
        
    def _extract_milestones(self, instructions):
        """Extract milestone information from assignment instructions."""
        if not instructions:
            return {}
            
        milestones = {}
        
        # Look for a milestone section in the instructions
        milestone_section_pattern = r'(?:Milestones|Project Milestones|Deliverables):(.*?)(?:\n\n|$)'
        milestone_section_match = re.search(milestone_section_pattern, instructions, re.IGNORECASE | re.DOTALL)
        
        if not milestone_section_match:
            return {}
            
        milestone_section = milestone_section_match.group(1)
        
        # Extract individual milestones
        milestone_pattern = r'(?:^|\n)\s*(?:\d+\.|[-*])\s*(?:\[(\d+)%\])?\s*(.+?)(?=\n\s*(?:\d+\.|[-*])|\Z)'
        milestone_matches = re.findall(milestone_pattern, milestone_section)
        
        for i, (weight_str, description) in enumerate(milestone_matches):
            milestone_id = f"M{i+1}"
            
            # Parse weight if available
            weight = 1.0  # Default weight
            if weight_str:
                try:
                    weight = float(weight_str) / 100.0
                except ValueError:
                    pass
                    
            # Add to milestones dictionary
            milestones[milestone_id] = {
                'name': description.strip(),
                'weight': weight,
                'order': i+1
            }
            
        return milestones
        
    def _evaluate_milestones(self, content, milestones):
        """Evaluate the completion status of each milestone."""
        if not milestones or not content:
            return {}
            
        milestone_completions = {}
        
        # Use content to create completion estimates
        try:
            # Use AI to evaluate milestone completion
            milestones_text = "\n".join([f"{m_id}: {m['name']}" for m_id, m in milestones.items()])
            
            completion_prompt = f"""
            Evaluate the completion status of each milestone for this project submission.
            For each milestone, provide:
            1. A completion percentage (0-100%)
            2. Brief evidence supporting your evaluation
            3. Specific feedback or suggestion
            
            Project milestones:
            {milestones_text}
            
            Project submission:
            ```
            {content[:3000]}  # Limit content length
            ```
            
            Format your response as a JSON object where keys are milestone IDs and values are objects with:
            - completion: number between 0 and 1
            - evidence: string with evidence of completion
            - feedback: string with specific feedback
            
            Keep your response concise and focus only on evaluating milestone completion.
            """
            
            milestone_results = self.gemini_service.generate_structured_response(completion_prompt)
            
            if milestone_results and isinstance(milestone_results, dict):
                # Process the results
                for m_id, m_data in milestone_results.items():
                    if m_id in milestones:
                        # Combine with original milestone data
                        milestone_completions[m_id] = {
                            **milestones[m_id],  # Original milestone data
                            'completion': m_data.get('completion', 0),
                            'evidence': m_data.get('evidence', ''),
                            'feedback': m_data.get('feedback', '')
                        }
            
            # Fall back to heuristic evaluation if AI fails
            if not milestone_completions:
                raise Exception("AI milestone evaluation failed")
                
        except Exception:
            # Use heuristic evaluation as fallback
            for m_id, milestone in milestones.items():
                milestone_name = milestone['name'].lower()
                words = set(re.findall(r'\b\w+\b', milestone_name))
                
                # Count word matches in content
                match_count = sum(1 for word in words if word.lower() in content.lower())
                
                # Estimate completion based on keyword presence
                if words:
                    completion = min(1.0, match_count / len(words))
                else:
                    completion = 0.5  # Default if no keywords
                    
                milestone_completions[m_id] = {
                    **milestone,
                    'completion': completion,
                    'evidence': f"Based on keyword matching ({match_count}/{len(words)} keywords found)",
                    'feedback': "Automatic evaluation - review manually for accuracy"
                }
                
        return milestone_completions
        
    def _extract_project_components(self, content):
        """Extract and analyze project components from content."""
        components = {}
        
        # Try to identify common project components
        # This is a simplified version - would be more robust with language-specific parsing
        
        # Check for documentation
        readme_match = re.search(r'(?i)# readme|readme\.md|documentation', content)
        if readme_match:
            components['Documentation'] = {
                'present': True,
                'quality': self._estimate_component_quality(content, 'documentation')
            }
            
        # Check for tests
        test_match = re.search(r'(?i)test_|_test|unittest|pytest|describe\(|it\(', content)
        if test_match:
            components['Testing'] = {
                'present': True,
                'quality': self._estimate_component_quality(content, 'testing')
            }
            
        # Check for database
        db_match = re.search(r'(?i)database|db\.|sql|query|model\.|schema', content)
        if db_match:
            components['Database'] = {
                'present': True,
                'quality': self._estimate_component_quality(content, 'database')
            }
            
        # Check for UI/frontend
        ui_match = re.search(r'(?i)html|css|<div|component|view|render', content)
        if ui_match:
            components['UI/Frontend'] = {
                'present': True,
                'quality': self._estimate_component_quality(content, 'ui')
            }
            
        # Check for API/backend
        api_match = re.search(r'(?i)api|route|controller|endpoint|server', content)
        if api_match:
            components['API/Backend'] = {
                'present': True,
                'quality': self._estimate_component_quality(content, 'api')
            }
            
        # Add AI-based component analysis if we have some components
        if components and len(content) > 500:
            try:
                component_prompt = f"""
                Analyze this project submission and identify the main components present.
                For each component, evaluate its quality and provide 1-2 specific issues or suggestions.
                
                Project content excerpt:
                ```
                {content[:2500]}
                ```
                
                Format your response as a JSON object with component names as keys and objects with 
                "quality" (string), "issues" (string), and "suggestions" (string) as values.
                
                Focus only on clearly identifiable components.
                """
                
                ai_components = self.gemini_service.generate_structured_response(component_prompt)
                
                if ai_components and isinstance(ai_components, dict):
                    # Merge AI components with detected components
                    for comp_name, comp_data in ai_components.items():
                        if isinstance(comp_data, dict):
                            components[comp_name] = comp_data
            except Exception:
                # Continue with basic component detection if AI fails
                pass
                
        return components
        
    def _estimate_component_quality(self, content, component_type):
        """Estimate the quality of a project component based on heuristics."""
        if component_type == 'documentation':
            # Check for comprehensive documentation
            if re.search(r'(?i)installation|usage|api reference|example', content):
                return "Good"
            elif re.search(r'(?i)readme|how to|instructions', content):
                return "Adequate"
            else:
                return "Minimal"
                
        elif component_type == 'testing':
            # Check for test coverage
            test_count = len(re.findall(r'(?i)test_|it\(|assert', content))
            if test_count > 10:
                return "Comprehensive"
            elif test_count > 5:
                return "Adequate"
            else:
                return "Limited"
                
        elif component_type == 'database':
            # Check for database design
            if re.search(r'(?i)migration|relationship|foreign key|join', content):
                return "Well-designed"
            elif re.search(r'(?i)table|model|schema', content):
                return "Basic"
            else:
                return "Minimal"
                
        elif component_type == 'ui':
            # Check for UI sophistication
            if re.search(r'(?i)responsive|animation|accessibility|component', content):
                return "Sophisticated"
            elif re.search(r'(?i)css|style|layout', content):
                return "Basic"
            else:
                return "Minimal"
                
        elif component_type == 'api':
            # Check for API design
            if re.search(r'(?i)rest|authentication|validation|error handling', content):
                return "Well-designed"
            elif re.search(r'(?i)route|endpoint|controller', content):
                return "Basic"
            else:
                return "Minimal"
                
        return "Unknown"
        
    def _analyze_project_progress(self, previous_content, current_content, 
                                 previous_milestone_completions, current_milestone_completions):
        """Analyze progress between project submissions."""
        if not previous_content or not current_content:
            return {}
            
        # Calculate basic metrics
        prev_length = len(previous_content)
        curr_length = len(current_content)
        length_change = curr_length - prev_length
        
        # Initialize progress analysis
        progress = {
            'length_change': length_change,
            'length_change_percentage': length_change / max(1, prev_length),
            'improvement_percentage': 0.0,
            'key_additions': [],
            'new_milestones': []
        }
        
        # Compare milestone completions if available
        if previous_milestone_completions and current_milestone_completions:
            new_milestones = []
            
            for m_id, curr_data in current_milestone_completions.items():
                if m_id in previous_milestone_completions:
                    prev_completion = previous_milestone_completions[m_id].get('completion', 0)
                    curr_completion = curr_data.get('completion', 0)
                    
                    # Check if milestone was newly completed
                    if prev_completion < 0.9 and curr_completion >= 0.9:
                        new_milestones.append(curr_data.get('name', f'Milestone {m_id}'))
                
            progress['new_milestones'] = new_milestones
            
            # Calculate overall progress
            prev_total = sum(m.get('completion', 0) * m.get('weight', 1) 
                           for m in previous_milestone_completions.values())
            curr_total = sum(m.get('completion', 0) * m.get('weight', 1) 
                           for m in current_milestone_completions.values())
            max_total = sum(m.get('weight', 1) for m in current_milestone_completions.values())
            
            if max_total > 0:
                prev_pct = prev_total / max_total
                curr_pct = curr_total / max_total
                progress['improvement_percentage'] = max(0, curr_pct - prev_pct)
        
        # Try to identify key additions using AI
        try:
            diff_prompt = f"""
            Compare the previous and current versions of this project and identify 
            the key changes or additions. Focus on meaningful functional changes, 
            not just minor edits or formatting.
            
            Previous version excerpt:
            ```
            {previous_content[:1000]}
            ```
            
            Current version excerpt:
            ```
            {current_content[:1000]}
            ```
            
            Respond with a JSON array of strings, each describing one key addition or change.
            Limit to the 5 most significant changes.
            """
            
            additions = self.gemini_service.generate_structured_response(diff_prompt)
            
            if additions and isinstance(additions, list):
                progress['key_additions'] = additions
            elif isinstance(additions, dict) and 'changes' in additions:
                progress['key_additions'] = additions['changes']
        except Exception:
            # Fall back to basic change identification
            # Find lines that appear in new but not old content
            import difflib
            prev_lines = previous_content.split('\n')
            curr_lines = current_content.split('\n')
            
            # Use difflib to find added lines
            diff = difflib.unified_diff(prev_lines, curr_lines, n=0)
            
            # Extract added lines (those starting with '+')
            added = [line[1:] for line in diff if line.startswith('+') and not line.startswith('+++')]
            
            # Filter for meaningful lines (not just whitespace, comments, etc.)
            meaningful = [line for line in added 
                        if len(line.strip()) > 10 and not line.strip().startswith(('#', '//', '/*', '*'))]
            
            # Select a sample of meaningful additions
            if meaningful:
                progress['key_additions'] = meaningful[:5]  # Limit to 5 key additions
        
        return progress
        
    def _format_components(self, components):
        """Format component information for inclusion in prompts."""
        if not components:
            return "No specific components identified."
            
        result = ""
        for name, details in components.items():
            quality = details.get('quality', 'Unknown')
            result += f"- {name}: {quality}\n"
            
        return result
        
    def _format_milestone_completion(self, milestone_completions):
        """Format milestone completion information for inclusion in prompts."""
        if not milestone_completions:
            return "No milestone data available."
            
        result = ""
        for m_id, data in milestone_completions.items():
            name = data.get('name', f'Milestone {m_id}')
            completion = data.get('completion', 0)
            result += f"- {name}: {completion:.0%} complete\n"
            
        return result
        
    def _format_progress_analysis(self, progress_analysis):
        """Format progress analysis information for inclusion in prompts."""
        if not progress_analysis:
            return "No progress data available."
            
        result = ""
        
        if 'improvement_percentage' in progress_analysis:
            result += f"Progress: {progress_analysis['improvement_percentage']:.1%} improvement\n"
            
        if progress_analysis.get('new_milestones'):
            result += "Newly completed milestones:\n"
            for milestone in progress_analysis['new_milestones']:
                result += f"- {milestone}\n"
                
        if progress_analysis.get('key_additions'):
            result += "Key additions:\n"
            for addition in progress_analysis['key_additions'][:3]:
                result += f"- {addition}\n"
                
        return result 