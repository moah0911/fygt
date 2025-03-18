"""EduMate Services Module

This module contains various services for the EduMate platform.
"""

# Import all service classes for easier access
from .ai_service import AIService
from .ai_grading import AIGradingService
from .feedback_service import FeedbackService
from .grading_service import GradingService
from .plagiarism_service import PlagiarismService
from .gemini_service import GeminiService
from .learning_path_service import LearningPathService
from .multilingual_feedback_service import MultilingualFeedbackService
from .study_recommendations_service import StudyRecommendationsService
from .group_formation_service import GroupFormationService
from .teacher_analytics_service import TeacherAnalyticsService

# Export all service classes
__all__ = [
    'AIService',
    'AIGradingService',
    'FeedbackService',
    'GradingService', 
    'PlagiarismService',
    'GeminiService',
    'LearningPathService',
    'MultilingualFeedbackService',
    'StudyRecommendationsService',
    'GroupFormationService',
    'TeacherAnalyticsService'
] 