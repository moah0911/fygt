import os
import re
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import json
import langid
from ..utils.logger import log_system_event

class MultilingualFeedbackService:
    """Service for providing multilingual feedback to students"""
    
    def __init__(self, gemini_service=None, language_detection_service=None, data_dir: str = 'data'):
        """
        Initialize the multilingual feedback service
        
        Args:
            gemini_service: The GeminiService for translation and content generation
            language_detection_service: Optional custom language detection service
            data_dir: Directory for storing data
        """
        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, 'cache', 'translations')
        self.gemini_service = gemini_service
        self.language_detection_service = language_detection_service
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Initialize supported languages
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'te': 'Telugu', 
            'ta': 'Tamil',
            'ur': 'Urdu'
        }
        
        # Initialize language detection if needed
        try:
            langid.set_languages(list(self.supported_languages.keys()))
        except Exception as e:
            log_system_event(f"Error initializing language detection: {str(e)}")
    
    def detect_language(self, text: str) -> Dict:
        """
        Detect the language of the provided text
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict with language code and confidence
        """
        try:
            # Use custom service if provided
            if self.language_detection_service:
                return self.language_detection_service.detect_language(text)
            
            # Use langid for language detection
            lang, confidence = langid.classify(text)
            
            # Ensure the detected language is in our supported list
            if lang not in self.supported_languages:
                lang = 'en'  # Default to English if not supported
            
            return {
                'language_code': lang,
                'language_name': self.supported_languages.get(lang, 'Unknown'),
                'confidence': confidence
            }
        except Exception as e:
            log_system_event(f"Error detecting language: {str(e)}")
            # Default to English on error
            return {
                'language_code': 'en',
                'language_name': 'English (default)',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def translate_feedback(self, feedback: str, target_language: str, source_language: str = None) -> Dict:
        """
        Translate feedback to the target language
        
        Args:
            feedback: The feedback text to translate
            target_language: The target language code
            source_language: Optional source language code (auto-detected if not provided)
            
        Returns:
            Dict with original and translated feedback
        """
        try:
            # Check if target language is supported
            if target_language not in self.supported_languages:
                return {
                    'status': 'error',
                    'message': f"Language {target_language} is not supported",
                    'original_feedback': feedback,
                    'translated_feedback': feedback
                }
            
            # Check cache for this translation
            cache_key = f"{hash(feedback)}_{target_language}"
            cached_result = self._check_translation_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Detect source language if not provided
            if not source_language:
                detected = self.detect_language(feedback)
                source_language = detected['language_code']
            
            # Skip translation if target matches source
            if source_language == target_language:
                return {
                    'status': 'success',
                    'message': 'No translation needed - source and target languages match',
                    'original_feedback': feedback,
                    'translated_feedback': feedback,
                    'source_language': source_language,
                    'target_language': target_language
                }
            
            # Use Gemini for translation
            if self.gemini_service:
                translated_feedback = self._translate_with_gemini(
                    feedback, 
                    source_language, 
                    target_language
                )
            else:
                # Fallback message if no translation service is available
                log_system_event("No translation service available")
                return {
                    'status': 'error',
                    'message': 'Translation service not available',
                    'original_feedback': feedback,
                    'translated_feedback': feedback
                }
            
            # Create result
            result = {
                'status': 'success',
                'original_feedback': feedback,
                'translated_feedback': translated_feedback,
                'source_language': source_language,
                'source_language_name': self.supported_languages.get(source_language, 'Unknown'),
                'target_language': target_language,
                'target_language_name': self.supported_languages.get(target_language, 'Unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self._cache_translation(cache_key, result)
            
            return result
            
        except Exception as e:
            log_system_event(f"Error translating feedback: {str(e)}")
            return {
                'status': 'error',
                'message': f"Translation error: {str(e)}",
                'original_feedback': feedback,
                'translated_feedback': feedback
            }
    
    def localize_feedback(self, feedback: Dict, target_language: str) -> Dict:
        """
        Translate and localize structured feedback
        
        Args:
            feedback: Dictionary containing feedback sections
            target_language: Target language code
            
        Returns:
            Dict with translated feedback sections
        """
        try:
            # Process different types of feedback structures
            if isinstance(feedback, str):
                # Simple string feedback
                translation = self.translate_feedback(feedback, target_language)
                return translation['translated_feedback']
            
            elif isinstance(feedback, dict):
                # Structure with sections
                localized_feedback = {}
                
                # Handle common feedback structures
                if 'strengths' in feedback and isinstance(feedback['strengths'], str):
                    translation = self.translate_feedback(feedback['strengths'], target_language)
                    localized_feedback['strengths'] = translation['translated_feedback']
                
                if 'weaknesses' in feedback and isinstance(feedback['weaknesses'], str):
                    translation = self.translate_feedback(feedback['weaknesses'], target_language)
                    localized_feedback['weaknesses'] = translation['translated_feedback']
                
                if 'suggestions' in feedback and isinstance(feedback['suggestions'], str):
                    translation = self.translate_feedback(feedback['suggestions'], target_language)
                    localized_feedback['suggestions'] = translation['translated_feedback']
                
                if 'summary' in feedback and isinstance(feedback['summary'], str):
                    translation = self.translate_feedback(feedback['summary'], target_language)
                    localized_feedback['summary'] = translation['translated_feedback']
                
                # Handle general text content
                if 'content' in feedback and isinstance(feedback['content'], str):
                    translation = self.translate_feedback(feedback['content'], target_language)
                    localized_feedback['content'] = translation['translated_feedback']
                
                # Handle detailed feedback items
                if 'feedback_items' in feedback and isinstance(feedback['feedback_items'], list):
                    localized_feedback['feedback_items'] = []
                    for item in feedback['feedback_items']:
                        if isinstance(item, dict):
                            localized_item = {}
                            for key, value in item.items():
                                if isinstance(value, str) and len(value) > 10:  # Only translate longer text
                                    translation = self.translate_feedback(value, target_language)
                                    localized_item[key] = translation['translated_feedback']
                                else:
                                    localized_item[key] = value
                            localized_feedback['feedback_items'].append(localized_item)
                        else:
                            localized_feedback['feedback_items'].append(item)
                
                # Include any untranslated fields from the original
                for key, value in feedback.items():
                    if key not in localized_feedback:
                        localized_feedback[key] = value
                
                return localized_feedback
            
            elif isinstance(feedback, list):
                # List of feedback items
                localized_items = []
                for item in feedback:
                    if isinstance(item, str) and len(item) > 10:
                        translation = self.translate_feedback(item, target_language)
                        localized_items.append(translation['translated_feedback'])
                    elif isinstance(item, dict):
                        localized_item = self.localize_feedback(item, target_language)
                        localized_items.append(localized_item)
                    else:
                        localized_items.append(item)
                return localized_items
            
            else:
                # Unsupported type
                log_system_event(f"Unsupported feedback type for localization: {type(feedback)}")
                return feedback
                
        except Exception as e:
            log_system_event(f"Error localizing feedback: {str(e)}")
            return feedback
    
    def get_language_preferences(self, student_id: str) -> Dict:
        """
        Get language preferences for a student
        
        Args:
            student_id: The student's ID
            
        Returns:
            Dict with language preferences
        """
        try:
            # Path to student preferences file
            prefs_file = os.path.join(self.data_dir, 'student_preferences', f'{student_id}.json')
            
            # Check if file exists
            if os.path.exists(prefs_file):
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
                    
                if 'language' in prefs:
                    return {
                        'preferred_language': prefs['language'],
                        'language_name': self.supported_languages.get(prefs['language'], 'Unknown')
                    }
            
            # Default to English if no preferences found
            return {
                'preferred_language': 'en',
                'language_name': 'English (default)'
            }
            
        except Exception as e:
            log_system_event(f"Error getting language preferences: {str(e)}")
            return {
                'preferred_language': 'en',
                'language_name': 'English (default)',
                'error': str(e)
            }
    
    def save_language_preference(self, student_id: str, language_code: str) -> Dict:
        """
        Save language preference for a student
        
        Args:
            student_id: The student's ID
            language_code: The preferred language code
            
        Returns:
            Dict with status and message
        """
        try:
            # Check if language is supported
            if language_code not in self.supported_languages:
                return {
                    'status': 'error',
                    'message': f"Language {language_code} is not supported"
                }
            
            # Directory for student preferences
            prefs_dir = os.path.join(self.data_dir, 'student_preferences')
            if not os.path.exists(prefs_dir):
                os.makedirs(prefs_dir)
            
            # Path to student preferences file
            prefs_file = os.path.join(prefs_dir, f'{student_id}.json')
            
            # Load existing preferences if any
            prefs = {}
            if os.path.exists(prefs_file):
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    prefs = json.load(f)
            
            # Update language preference
            prefs['language'] = language_code
            prefs['language_name'] = self.supported_languages[language_code]
            prefs['updated_at'] = datetime.now().isoformat()
            
            # Save updated preferences
            with open(prefs_file, 'w', encoding='utf-8') as f:
                json.dump(prefs, f, indent=2)
            
            return {
                'status': 'success',
                'message': f"Language preference updated to {self.supported_languages[language_code]}",
                'language_code': language_code,
                'language_name': self.supported_languages[language_code]
            }
            
        except Exception as e:
            log_system_event(f"Error saving language preference: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error saving language preference: {str(e)}"
            }
    
    def get_supported_languages(self) -> Dict:
        """
        Get list of supported languages
        
        Returns:
            Dict with language codes and names
        """
        return {
            'count': len(self.supported_languages),
            'languages': [
                {'code': code, 'name': name} 
                for code, name in self.supported_languages.items()
            ]
        }
    
    def _translate_with_gemini(self, text: str, source_language: str, target_language: str) -> str:
        """Use Gemini service for translation"""
        try:
            # Skip if text is too short
            if len(text) < 5:
                return text
                
            # Format prompt for translation
            source_lang_name = self.supported_languages.get(source_language, "unknown language")
            target_lang_name = self.supported_languages.get(target_language, "unknown language")
            
            prompt = f"""Translate the following text from {source_lang_name} to {target_lang_name}. 
            Maintain the original formatting, including paragraphs, bullet points, and line breaks.
            For any technical terms that don't have a direct translation, keep the original term in parentheses.
            
            Text to translate:
            
            {text}
            
            Translation ({target_lang_name}):
            """
            
            # Call Gemini service for translation
            response = self.gemini_service.generate_content(prompt)
            
            # Extract the translation from the response
            if hasattr(response, 'text'):
                translated_text = response.text.strip()
            elif isinstance(response, dict) and 'text' in response:
                translated_text = response['text'].strip()
            elif isinstance(response, str):
                translated_text = response.strip()
            else:
                log_system_event(f"Unexpected response from translation service: {type(response)}")
                return text
            
            # Clean up the translation if needed
            if translated_text.startswith('Translation'):
                # Remove any "Translation:" prefix if present
                match = re.search(r'Translation.*?:(.*)', translated_text, re.DOTALL)
                if match:
                    translated_text = match.group(1).strip()
            
            return translated_text
            
        except Exception as e:
            log_system_event(f"Error translating with Gemini: {str(e)}")
            return text  # Return original text on error
    
    def _check_translation_cache(self, cache_key: str) -> Optional[Dict]:
        """Check cache for existing translation"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                return cached
            return None
        except Exception as e:
            log_system_event(f"Error checking translation cache: {str(e)}")
            return None
    
    def _cache_translation(self, cache_key: str, result: Dict) -> None:
        """Cache translation result"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            log_system_event(f"Error caching translation: {str(e)}") 