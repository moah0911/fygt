"""Gemini service for AI-powered grading and feedback with enhanced capabilities."""
import os
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import google.generativeai as genai
try:
    import openai
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiService:
    """Enhanced service for interacting with the Gemini API."""
    
    def __init__(self, api_key=None, fallback_api_key=None):
        """Initialize the Gemini service.
        
        Args:
            api_key: Gemini API key
            fallback_api_key: OpenAI API key for fallback
        """
        # Try environment variables if not provided
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.openai_api_key = fallback_api_key or os.getenv('OPENAI_API_KEY')
        self.use_fallback = False
        
        # Set up caching
        self.cache_dir = Path("data/cache/gemini")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure Gemini
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                self.vision_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
                self.use_fallback = bool(self.openai_api_key)
        else:
            self.use_fallback = bool(self.openai_api_key)
            logger.warning("Gemini API key not found. Using OpenAI fallback if available.")
            
        # Configure OpenAI fallback
        if self.use_fallback and self.openai_api_key and openai:
            try:
                openai.api_key = self.openai_api_key
                logger.info("OpenAI fallback initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI fallback: {e}")
                self.use_fallback = False
    
    def generate_text(self, prompt: str, temperature: float = 0.7, 
                     max_tokens: int = 1024, retries: int = 2) -> str:
        """Generate text using Gemini or fallback to OpenAI.
        
        Args:
            prompt: The prompt to generate text from
            temperature: Creativity level (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            retries: Number of retry attempts
            
        Returns:
            Generated text response
        """
        # Check cache first
        cache_key = self._generate_cache_key(prompt, temperature, max_tokens)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
            
        # Try Gemini first
        if not self.use_fallback and self.api_key:
            for attempt in range(retries):
                try:
                    # Use Gemini
                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens
                        )
                    )
                    result = response.text
                    self._cache_result(cache_key, result)
                    return result
                except Exception as e:
                    logger.error(f"Error using Gemini API (attempt {attempt+1}/{retries}): {e}")
                    if attempt == retries - 1 and self.openai_api_key:
                        self.use_fallback = True
                        logger.info("Falling back to OpenAI.")
                    else:
                        time.sleep(1)  # Slight delay before retry
        
        # Use OpenAI as fallback
        if self.use_fallback and self.openai_api_key and openai:
            try:
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result = response.choices[0].message.content
                self._cache_result(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"Error using OpenAI fallback: {e}")
                return f"Error: {str(e)}"
        
        return "Error: No API keys available for text generation."
    
    def generate_structured_response(self, prompt: str, 
                                    retries: int = 2, 
                                    simplify: bool = False) -> Dict[str, Any]:
        """Generate a structured JSON response.
        
        Args:
            prompt: The prompt to generate from
            retries: Number of retry attempts
            simplify: Whether to simplify the prompt for fallback
            
        Returns:
            Parsed JSON response as dictionary
        """
        # Add explicit instruction for JSON format
        json_prompt = f"""
        {prompt}
        
        IMPORTANT: Format your entire response as valid JSON. 
        Do not include any explanatory text outside the JSON structure.
        Ensure all keys and values are properly quoted.
        """
        
        if simplify:
            json_prompt = f"""
            {prompt}
            
            Return a simple JSON response with no nesting more than 2 levels deep.
            """
            
        # Try to get a response and parse as JSON
        for attempt in range(retries):
            try:
                response_text = self.generate_text(json_prompt, temperature=0.1)
                
                # Extract JSON part
                json_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
                else:
                    # Try to parse the entire response as JSON
                    return json.loads(response_text)
            except Exception as e:
                logger.warning(f"Failed to parse JSON response (attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    # Last attempt, try with more explicit instructions
                    try:
                        simplified_prompt = f"""
                        {prompt}
                        
                        CRITICAL: Return ONLY a valid JSON object with no other text.
                        The response must be parseable by json.loads().
                        """
                        response_text = self.generate_text(simplified_prompt, temperature=0.1)
                        json_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group(0))
                    except Exception as final_e:
                        logger.error(f"Final JSON parsing attempt failed: {final_e}")
                        
        # If all attempts fail, return empty dict
        logger.error("All attempts to parse structured response failed")
        return {}
    
    def analyze_text(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyze text with Gemini and return structured insights.
        
        Args:
            text: Text to analyze
            prompt: Optional specific prompt (otherwise general analysis)
            
        Returns:
            Dictionary with analysis results
        """
        # Default analysis prompt
        if not prompt:
            prompt = f"""
            Analyze the following text and provide detailed insights:
            
            {text}
            
            Include analysis of:
            1. Main themes and topics
            2. Writing style and tone
            3. Clarity and structure
            4. Key strengths and weaknesses
            
            Format your response as JSON with these keys:
            - themes: array of main themes
            - tone: assessment of writing tone
            - clarity: rating from 1-10 with explanation
            - strengths: array of strengths
            - weaknesses: array of weaknesses
            - recommendations: array of improvement suggestions
            """
        else:
            prompt = f"{prompt}\n\n{text}"
            
        return self.generate_structured_response(prompt)
    
    def grade_with_rubric(self, content: str, rubric: List[Dict], 
                         max_score: int = 100) -> Dict[str, Any]:
        """Grade content against a specific rubric.
        
        Args:
            content: Content to grade
            rubric: List of rubric criteria with names and points
            max_score: Maximum possible score
            
        Returns:
            Dictionary with scores and feedback
        """
        # Format rubric for prompt
        rubric_text = "Grading Rubric:\n"
        for criterion in rubric:
            name = criterion.get('name', 'Unnamed Criterion')
            points = criterion.get('points', 0)
            description = criterion.get('description', '')
            rubric_text += f"- {name} ({points} points): {description}\n"
        
        # Create grading prompt
        prompt = f"""
        Grade the following content against the provided rubric.
        
        {rubric_text}
        
        Content to grade:
        {content[:12000]}  # Limit content length
        
        For each criterion in the rubric:
        1. Provide a score out of the available points
        2. Provide specific evidence from the content supporting your score
        3. Offer constructive feedback for improvement
        
        Also provide:
        - Overall strengths (2-3 points)
        - Areas for improvement (2-3 points)
        - Suggestions for development
        
        Format your response as JSON with:
        - criterion_scores: object with criterion names as keys, containing score, evidence, and feedback
        - total_score: sum of criterion scores (out of {max_score})
        - strengths: array of strength points
        - weaknesses: array of weakness points
        - improvement_suggestions: array of specific suggestions
        """
        
        return self.generate_structured_response(prompt)
    
    def review_code(self, code: str, language: str, 
                   requirements: Optional[str] = None) -> Dict[str, Any]:
        """Review code for quality, correctness, and style.
        
        Args:
            code: Code to review
            language: Programming language
            requirements: Code requirements (optional)
            
        Returns:
            Dictionary with code review results
        """
        # Create code review prompt
        prompt = f"""
        Review this {language} code for quality, correctness, and style.
        
        ```{language}
        {code}
        ```
        
        {f"Requirements:\n{requirements}\n" if requirements else ""}
        
        Analyze the code for:
        1. Correctness - does it work as intended?
        2. Code quality - structure, naming, readability
        3. Style - adherence to {language} best practices
        4. Potential bugs or edge cases
        5. Performance considerations
        6. Security issues
        
        Format your response as JSON with:
        - correctness: assessment of code correctness (1-10)
        - quality: assessment of code quality (1-10)
        - style: assessment of code style (1-10) 
        - issues: array of specific issues found
        - bugs: array of potential bugs or edge cases
        - suggestions: array of improvement suggestions
        - strengths: array of code strengths
        """
        
        return self.generate_structured_response(prompt)
    
    def detect_ai_content(self, text: str) -> Dict[str, Any]:
        """Detect whether content was likely AI-generated.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with AI detection results
        """
        prompt = f"""
        Analyze this text and determine if it was likely generated by AI:
        
        {text[:8000]}  # Limit content length
        
        Look for indicators such as:
        1. Repetitive patterns or phrases
        2. Lack of personal perspective
        3. Generic examples
        4. Uniform tone throughout
        5. Lack of nuance on complex topics
        
        Format your response as JSON with:
        - probability: number from 0.0 to 1.0 indicating likelihood of AI generation
        - confidence: confidence in the assessment (low, medium, high)
        - indicators: array of specific indicators found
        - human_elements: array of elements suggesting human authorship
        """
        
        return self.generate_structured_response(prompt)
    
    def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze an image using Gemini's vision capabilities.
        
        Args:
            image_path: Path to image file
            prompt: Specific analysis instructions
            
        Returns:
            Dictionary with image analysis results
        """
        if not self.api_key:
            return {"error": "Gemini API key required for image analysis"}
            
        try:
            # Load the image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                
            # Create multimodal prompt with image
            multimodal_prompt = [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
            
            # Generate image analysis
            response = self.vision_model.generate_content(multimodal_prompt)
            
            # Try to parse as JSON if possible
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                # Return as regular text if not JSON
                return {"analysis": response.text}
                
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}
    
    def check_plagiarism(self, content: str, reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Check content for plagiarism against provided references.
        
        Args:
            content: Content to check
            reference_texts: Reference texts to compare against
            
        Returns:
            Dictionary with plagiarism analysis
        """
        if not reference_texts:
            prompt = f"""
            Analyze this content for indicators of plagiarism or AI generation:
            
            {content[:8000]}  # Limit content length
            
            Look for:
            1. Unusual language patterns or inconsistencies
            2. Academic language that seems out of place
            3. Sentences that appear to be from different sources
            4. Generic phrasing typical of AI generation
            
            Format your response as JSON with:
            - plagiarism_indicators: array of suspicious elements
            - ai_generation_indicators: array of AI-like patterns
            - originality_score: estimated originality (0-100%)
            - analysis: overall assessment
            """
        else:
            # Format reference texts
            reference_formatted = []
            for i, ref in enumerate(reference_texts[:3]):  # Limit to 3 references
                reference_formatted.append(f"Reference {i+1}:\n{ref[:1000]}")
                
            references = "\n\n".join(reference_formatted)
            
            prompt = f"""
            Compare this content against the provided references for potential plagiarism:
            
            Content to check:
            {content[:5000]}  # Limit content length
            
            References to compare against:
            {references}
            
            Identify:
            1. Any specific passages that appear similar to references
            2. Overall similarity assessment
            3. Potential direct copying vs paraphrasing
            
            Format your response as JSON with:
            - similarity_score: estimated similarity (0.0 to 1.0)
            - similar_passages: array of potentially plagiarized passages with matching reference
            - assessment: overall plagiarism assessment
            """
            
        return self.generate_structured_response(prompt)
    
    def generate_feedback(self, submission: Dict[str, Any], tone: str = "constructive") -> str:
        """Generate personalized feedback for a submission.
        
        Args:
            submission: Submission data
            tone: Desired tone for feedback
            
        Returns:
            Formatted feedback string
        """
        # Extract key information from submission
        content = submission.get('content', '')[:5000]  # Limit content length
        score = submission.get('score', 0)
        max_score = submission.get('max_score', 100)
        feedback_points = submission.get('feedback_points', [])
        
        # Format existing feedback points
        feedback_formatted = ""
        if feedback_points:
            feedback_formatted = "Existing feedback points:\n" + "\n".join([f"- {point}" for point in feedback_points])
        
        prompt = f"""
        Generate personalized, {tone} feedback for this student submission.
        
        Submission excerpt:
        {content}
        
        Score: {score} out of {max_score}
        
        {feedback_formatted}
        
        The feedback should:
        1. Start with positive encouragement
        2. Highlight specific strengths
        3. Address areas for improvement constructively
        4. Provide specific, actionable suggestions
        5. End with encouraging next steps
        
        Use a {tone} tone throughout. Make the feedback personalized and specific to this submission.
        """
        
        return self.generate_text(prompt)
    
    def suggest_resources(self, topic: str, student_level: str = "intermediate",
                         resource_types: List[str] = None) -> List[Dict[str, str]]:
        """Suggest learning resources for a specific topic.
        
        Args:
            topic: Topic to find resources for
            student_level: Student knowledge level
            resource_types: Types of resources to suggest
            
        Returns:
            List of resource suggestions
        """
        resource_types_str = ", ".join(resource_types) if resource_types else "articles, videos, books, tutorials"
        
        prompt = f"""
        Suggest high-quality learning resources about {topic} for a {student_level} level student.
        
        Resource types to include: {resource_types_str}
        
        For each resource, provide:
        1. Title
        2. Type (video, article, book, etc.)
        3. Difficulty level
        4. Brief description
        5. URL or source (if possible)
        
        Format your response as a JSON array of resources, each with title, type, level, description, and url fields.
        """
        
        result = self.generate_structured_response(prompt)
        
        # Extract resources from result
        if isinstance(result, dict) and 'resources' in result:
            return result['resources']
        elif isinstance(result, list):
            return result
        else:
            return []
    
    def analyze_tone(self, text: str) -> Dict[str, Any]:
        """Analyze the tone of text for educational feedback appropriateness.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with tone analysis
        """
        prompt = f"""
        Analyze the tone of this educational feedback:
        
        {text}
        
        Evaluate it for:
        1. Constructiveness (Is it helpful and motivating?)
        2. Specificity (Is it detailed and relevant?)
        3. Balance (Does it address both strengths and weaknesses?)
        4. Growth mindset language (Does it focus on improvement?)
        5. Empathy (Is it considerate of the student's perspective?)
        
        Format your response as JSON with:
        - constructiveness: rating from 1-10
        - specificity: rating from 1-10
        - balance: rating from 1-10
        - growth_mindset: rating from 1-10
        - empathy: rating from 1-10
        - overall_score: average of all ratings
        - needs_review: boolean indicating if tone needs improvement
        - improvement_suggestions: array of suggestions if needed
        """
        
        return self.generate_structured_response(prompt)
    
    def rephrase_feedback(self, feedback: str) -> str:
        """Rephrase feedback to improve tone and effectiveness.
        
        Args:
            feedback: Original feedback text
            
        Returns:
            Improved feedback text
        """
        prompt = f"""
        Rephrase this educational feedback to improve its tone and effectiveness:
        
        {feedback}
        
        Make the feedback:
        1. More constructive and growth-oriented
        2. Balanced between strengths and areas for improvement
        3. Specific and actionable
        4. Empathetic and encouraging
        5. Clear and concise
        
        Maintain the same substantive content and specific points.
        """
        
        return self.generate_text(prompt)
    
    def advanced_analysis(self, prompt: str) -> Dict[str, Any]:
        """Perform advanced structured analysis with Gemini's most advanced capabilities.
        
        Args:
            prompt: Detailed analysis prompt
            
        Returns:
            Structured analysis results
        """
        # Use the most detailed model available
        if self.api_key:
            try:
                # Use Gemini Pro
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=2048,
                        top_p=0.95,
                        top_k=40
                    )
                )
                
                # Try to parse as JSON
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    # Try to extract JSON
                    json_match = re.search(r'(\{.*\}|\[.*\])', response.text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                    else:
                        return {"analysis": response.text}
                    
            except Exception as e:
                logger.error(f"Error with advanced analysis: {e}")
                
        # Fallback to regular structured response
        return self.generate_structured_response(prompt)
    
    def _generate_cache_key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate a cache key for a prompt."""
        # Hash prompt and parameters
        import hashlib
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        param_str = f"_t{temperature}_m{max_tokens}"
        return f"{prompt_hash}{param_str}"
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if result exists in cache."""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read cache: {str(e)}")
        return None
    
    def _cache_result(self, cache_key: str, result: str) -> None:
        """Cache a result."""
        cache_file = self.cache_dir / f"{cache_key}.txt"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(result)
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}") 