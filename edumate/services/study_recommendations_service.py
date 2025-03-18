import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
import difflib
from ..utils.logger import log_system_event

class StudyRecommendationsService:
    """Service for generating personalized study recommendations based on student performance"""
    
    def __init__(self, gemini_service=None, data_dir: str = 'data'):
        """
        Initialize the study recommendations service
        
        Args:
            gemini_service: The GeminiService for generating personalized recommendations
            data_dir: Directory for storing data
        """
        self.data_dir = data_dir
        self.gemini_service = gemini_service
        self.resources_dir = os.path.join(data_dir, 'resources')
        self.recommendations_dir = os.path.join(data_dir, 'recommendations')
        
        # Create directories if they don't exist
        for directory in [self.resources_dir, self.recommendations_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # Load resource database if exists
        self.resources = self._load_resources()
    
    def generate_recommendations(self, student_id: str, performance_data: Dict,
                               submission_history: List[Dict] = None,
                               course_data: List[Dict] = None) -> Dict:
        """
        Generate personalized study recommendations based on student performance
        
        Args:
            student_id: ID of the student
            performance_data: Dictionary with performance metrics by skill/topic
            submission_history: Optional list of previous submissions
            course_data: Optional list of course information
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            # Identify areas for improvement
            improvement_areas = self._identify_improvement_areas(performance_data)
            
            # Find relevant learning resources
            resource_recommendations = self._find_learning_resources(improvement_areas)
            
            # Generate practice recommendations
            practice_recommendations = self._generate_practice_activities(
                improvement_areas, submission_history)
            
            # Generate time management recommendations
            time_recommendations = self._generate_time_management_plan(
                improvement_areas, submission_history)
            
            # Generate personalized AI suggestions if Gemini service available
            ai_suggestions = self._generate_ai_suggestions(
                student_id, performance_data, improvement_areas)
            
            # Create study schedule
            study_schedule = self._create_study_schedule(
                improvement_areas, course_data)
            
            # Build recommendations object
            recommendations = {
                "student_id": student_id,
                "generated_at": datetime.now().isoformat(),
                "improvement_areas": improvement_areas,
                "resources": resource_recommendations,
                "practice": practice_recommendations,
                "time_management": time_recommendations,
                "ai_suggestions": ai_suggestions,
                "study_schedule": study_schedule
            }
            
            # Save recommendations for future reference
            self._save_recommendations(student_id, recommendations)
            
            return recommendations
            
        except Exception as e:
            log_system_event(f"Error generating study recommendations: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating recommendations: {str(e)}"
            }
    
    def get_latest_recommendations(self, student_id: str) -> Dict:
        """
        Get the most recent study recommendations for a student
        
        Args:
            student_id: ID of the student
            
        Returns:
            Dictionary containing recommendations
        """
        try:
            # Path to student recommendations
            recommendations_path = os.path.join(self.recommendations_dir, f"{student_id}.json")
            
            if os.path.exists(recommendations_path):
                with open(recommendations_path, 'r', encoding='utf-8') as f:
                    recommendations = json.load(f)
                
                # Check if recommendations are recent (within a week)
                generated_at = datetime.fromisoformat(recommendations.get('generated_at', ''))
                if datetime.now() - generated_at <= timedelta(days=7):
                    recommendations['is_recent'] = True
                else:
                    recommendations['is_recent'] = False
                    recommendations['freshness_warning'] = "These recommendations are more than a week old. Consider generating new ones for the most relevant suggestions."
                
                return recommendations
            else:
                return {
                    "status": "not_found",
                    "message": "No recommendations found for this student"
                }
                
        except Exception as e:
            log_system_event(f"Error retrieving recommendations: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving recommendations: {str(e)}"
            }
    
    def add_learning_resource(self, resource: Dict) -> Dict:
        """
        Add a new learning resource to the database
        
        Args:
            resource: Dictionary containing resource information
                (title, url, type, topics, description, etc.)
                
        Returns:
            Dictionary with status information
        """
        try:
            # Validate required fields
            required_fields = ['title', 'type', 'topics']
            for field in required_fields:
                if field not in resource:
                    return {
                        "status": "error",
                        "message": f"Missing required field: {field}"
                    }
            
            # Add timestamp if not present
            if 'added_at' not in resource:
                resource['added_at'] = datetime.now().isoformat()
            
            # Generate ID if not provided
            if 'id' not in resource:
                resource['id'] = f"res_{len(self.resources) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Add resource to database
            self.resources.append(resource)
            
            # Save updated resources
            self._save_resources()
            
            return {
                "status": "success",
                "message": "Resource added successfully",
                "resource_id": resource['id']
            }
            
        except Exception as e:
            log_system_event(f"Error adding learning resource: {str(e)}")
            return {
                "status": "error",
                "message": f"Error adding resource: {str(e)}"
            }
    
    def search_resources(self, query: str, topic: str = None, 
                       resource_type: str = None, limit: int = 10) -> Dict:
        """
        Search for learning resources
        
        Args:
            query: Search query
            topic: Optional topic filter
            resource_type: Optional resource type filter
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            # Filter resources based on criteria
            filtered_resources = self.resources
            
            # Apply topic filter if provided
            if topic:
                filtered_resources = [r for r in filtered_resources if topic.lower() in [t.lower() for t in r.get('topics', [])]]
            
            # Apply type filter if provided
            if resource_type:
                filtered_resources = [r for r in filtered_resources if r.get('type', '').lower() == resource_type.lower()]
            
            # Find matching resources using search query
            matches = []
            for resource in filtered_resources:
                score = 0
                
                # Check title
                if 'title' in resource and query.lower() in resource['title'].lower():
                    score += 3
                
                # Check description
                if 'description' in resource and query.lower() in resource['description'].lower():
                    score += 2
                
                # Check topics
                if 'topics' in resource:
                    for topic in resource['topics']:
                        if query.lower() in topic.lower():
                            score += 1
                
                # Include resource if it matches
                if score > 0:
                    matches.append({
                        "resource": resource,
                        "relevance_score": score
                    })
            
            # Sort by relevance (highest first)
            matches.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Limit results
            matches = matches[:limit]
            
            return {
                "status": "success",
                "query": query,
                "filters": {
                    "topic": topic,
                    "type": resource_type
                },
                "total_results": len(matches),
                "results": [m['resource'] for m in matches]
            }
            
        except Exception as e:
            log_system_event(f"Error searching resources: {str(e)}")
            return {
                "status": "error",
                "message": f"Error searching resources: {str(e)}"
            }
    
    def _identify_improvement_areas(self, performance_data: Dict) -> List[Dict]:
        """Identify areas that need improvement based on performance data"""
        improvement_areas = []
        
        # Process different performance data formats
        if 'skills' in performance_data:
            # Skills-based performance data
            for skill in performance_data['skills']:
                if skill.get('needs_improvement', False) or skill.get('proficiency_score', 1.0) < 0.7:
                    improvement_areas.append({
                        "name": skill['name'],
                        "type": "skill",
                        "current_score": skill.get('proficiency_score', 0) * 100,
                        "priority": self._calculate_priority(skill)
                    })
        
        elif 'assignments' in performance_data:
            # Assignment-based performance data
            low_scoring = []
            for assignment in performance_data['assignments']:
                score = assignment.get('score', 0)
                max_score = assignment.get('max_score', 100)
                percentage = (score / max_score * 100) if max_score > 0 else 0
                
                if percentage < 70:
                    low_scoring.append({
                        "name": assignment.get('title', 'Assignment'),
                        "type": "assignment",
                        "score": score,
                        "max_score": max_score,
                        "percentage": percentage,
                        "topics": assignment.get('topics', [])
                    })
            
            # Group by topics from low-scoring assignments
            topic_scores = {}
            for assignment in low_scoring:
                for topic in assignment.get('topics', []):
                    if topic not in topic_scores:
                        topic_scores[topic] = []
                    topic_scores[topic].append(assignment['percentage'])
            
            # Create improvement areas for topics
            for topic, scores in topic_scores.items():
                avg_score = sum(scores) / len(scores)
                improvement_areas.append({
                    "name": topic,
                    "type": "topic",
                    "current_score": avg_score,
                    "priority": "high" if avg_score < 60 else "medium"
                })
        
        # Sort improvement areas by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        improvement_areas.sort(key=lambda x: priority_order.get(x['priority'], 999))
        
        return improvement_areas
    
    def _calculate_priority(self, skill: Dict) -> str:
        """Calculate priority level for a skill"""
        score = skill.get('proficiency_score', 0)
        
        if score < 0.5:
            return "high"
        elif score < 0.7:
            return "medium"
        else:
            return "low"
    
    def _find_learning_resources(self, improvement_areas: List[Dict]) -> List[Dict]:
        """Find relevant learning resources for improvement areas"""
        recommendations = []
        
        for area in improvement_areas:
            area_name = area['name']
            
            # Find resources for this area
            matching_resources = []
            
            for resource in self.resources:
                # Check if resource topics match the improvement area
                if any(difflib.SequenceMatcher(None, area_name.lower(), topic.lower()).ratio() > 0.8 
                      for topic in resource.get('topics', [])):
                    matching_resources.append(resource)
            
            # Sort by relevance (based on how well topics match)
            matching_resources.sort(
                key=lambda r: max([difflib.SequenceMatcher(None, area_name.lower(), topic.lower()).ratio() 
                                 for topic in r.get('topics', [])]), 
                reverse=True
            )
            
            # Take top 3 resources for this area
            resources_for_area = matching_resources[:3]
            
            # If no resources found, add generic placeholders
            if not resources_for_area:
                if area['type'] == 'skill' or area['type'] == 'topic':
                    resources_for_area = [
                        {
                            "title": f"Khan Academy: {area_name}",
                            "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={area_name.replace(' ', '+')}",
                            "type": "video",
                            "topics": [area_name],
                            "is_placeholder": True
                        },
                        {
                            "title": f"YouTube Tutorials: {area_name}",
                            "url": f"https://www.youtube.com/results?search_query={area_name.replace(' ', '+')}+tutorial",
                            "type": "video",
                            "topics": [area_name],
                            "is_placeholder": True
                        }
                    ]
            
            # Add resources to recommendations
            for resource in resources_for_area:
                recommendation = {
                    "area": area_name,
                    "area_type": area['type'],
                    "priority": area['priority'],
                    "resource": resource
                }
                recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_practice_activities(self, improvement_areas: List[Dict], 
                                    submission_history: List[Dict] = None) -> List[Dict]:
        """Generate practice activity recommendations"""
        practice_recommendations = []
        
        for area in improvement_areas:
            area_name = area['name']
            
            # Create different practice recommendations based on area type
            if area['type'] == 'skill':
                # Skill-based recommendations
                practice_recommendations.append({
                    "area": area_name,
                    "type": "exercise",
                    "title": f"Practice exercises for {area_name}",
                    "description": f"Complete targeted practice problems focusing on {area_name}",
                    "difficulty": "adaptive",
                    "priority": area['priority']
                })
                
                # Add flashcard recommendation if appropriate
                if area['current_score'] < 60:
                    practice_recommendations.append({
                        "area": area_name,
                        "type": "flashcards",
                        "title": f"Create flashcards for {area_name} concepts",
                        "description": "Create and review flashcards with key concepts and definitions",
                        "difficulty": "basic",
                        "priority": area['priority']
                    })
            
            elif area['type'] == 'topic':
                # Topic-based recommendations
                practice_recommendations.append({
                    "area": area_name,
                    "type": "summary",
                    "title": f"Create a summary sheet for {area_name}",
                    "description": "Create a one-page summary of key concepts and principles",
                    "difficulty": "medium",
                    "priority": area['priority']
                })
                
                practice_recommendations.append({
                    "area": area_name,
                    "type": "exercise",
                    "title": f"Practice problems for {area_name}",
                    "description": "Complete a variety of problems to reinforce understanding",
                    "difficulty": "medium",
                    "priority": area['priority']
                })
            
            # Add specific recommendations based on submission history if available
            if submission_history:
                topic_submissions = [s for s in submission_history 
                                    if area_name.lower() in [t.lower() for t in s.get('topics', [])]]
                
                if topic_submissions:
                    # Find most challenging aspects based on submission history
                    low_scores = [s for s in topic_submissions if s.get('score', 100) / s.get('max_score', 100) < 0.7]
                    
                    if low_scores and len(low_scores) > 0:
                        # Identify common feedback themes
                        feedback_points = []
                        for submission in low_scores:
                            if 'feedback' in submission:
                                feedback_points.append(submission['feedback'])
                        
                        if feedback_points:
                            # Create practice recommendation based on feedback
                            practice_recommendations.append({
                                "area": area_name,
                                "type": "targeted",
                                "title": f"Targeted practice for {area_name}",
                                "description": "Focus on previously challenging problems based on your submission history",
                                "difficulty": "challenging",
                                "priority": "high",
                                "based_on_history": True
                            })
        
        return practice_recommendations
    
    def _generate_time_management_plan(self, improvement_areas: List[Dict],
                                     submission_history: List[Dict] = None) -> Dict:
        """Generate time management plan for study sessions"""
        # Calculate total study time needed
        high_priority = len([a for a in improvement_areas if a['priority'] == 'high'])
        medium_priority = len([a for a in improvement_areas if a['priority'] == 'medium'])
        low_priority = len([a for a in improvement_areas if a['priority'] == 'low'])
        
        # Base times per week (in hours)
        high_time = 3.0  # 3 hours per high priority area per week
        medium_time = 2.0  # 2 hours per medium priority area per week
        low_time = 1.0  # 1 hour per low priority area per week
        
        total_weekly_hours = (high_priority * high_time) + (medium_priority * medium_time) + (low_priority * low_time)
        
        # If too high, cap and adjust proportionally
        if total_weekly_hours > 15:
            scale_factor = 15 / total_weekly_hours
            high_time *= scale_factor
            medium_time *= scale_factor
            low_time *= scale_factor
            total_weekly_hours = 15
        
        # Create time allocation by priority
        time_allocation = {
            "high_priority": round(high_priority * high_time, 1),
            "medium_priority": round(medium_priority * medium_time, 1),
            "low_priority": round(low_priority * low_time, 1)
        }
        
        # Generate recommended sessions
        sessions = []
        remaining_areas = improvement_areas.copy()
        
        # Create study blocks for each priority level
        for priority in ["high", "medium", "low"]:
            priority_areas = [a for a in remaining_areas if a['priority'] == priority]
            
            for area in priority_areas:
                # Calculate time per area based on priority
                if priority == "high":
                    time_per_area = high_time
                elif priority == "medium":
                    time_per_area = medium_time
                else:
                    time_per_area = low_time
                
                # Create sessions for this area
                session_length = 1.0  # 1 hour per session
                num_sessions = round(time_per_area / session_length)
                
                for i in range(num_sessions):
                    sessions.append({
                        "area": area['name'],
                        "duration_hours": session_length,
                        "priority": priority,
                        "focus": "Concept understanding" if i == 0 else "Practice problems" if i == 1 else "Review and consolidation"
                    })
        
        # Distribute sessions throughout the week
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        max_daily_sessions = 2  # Maximum 2 sessions per day
        
        # Shuffle sessions to distribute priorities
        random.shuffle(sessions)
        
        # Assign days
        for i, session in enumerate(sessions):
            day_index = min(i // max_daily_sessions, len(days_of_week) - 1)
            session["day"] = days_of_week[day_index]
        
        # Sort by day and then by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sessions.sort(key=lambda x: (days_of_week.index(x["day"]), priority_order.get(x["priority"], 999)))
        
        # Create the time management plan
        time_management_plan = {
            "total_weekly_hours": round(total_weekly_hours, 1),
            "time_allocation": time_allocation,
            "sessions": sessions,
            "recommendations": [
                "Try to study at the same time each day to build a routine",
                "Take short breaks of 5-10 minutes for each hour of studying",
                "Start with high-priority areas when you have the most energy",
                "Review your notes or flashcards for 10 minutes before bed to improve retention"
            ]
        }
        
        return time_management_plan
    
    def _generate_ai_suggestions(self, student_id: str, performance_data: Dict,
                              improvement_areas: List[Dict]) -> List[Dict]:
        """Generate personalized AI suggestions using Gemini"""
        if not self.gemini_service:
            return [{
                "type": "general",
                "title": "Study Tip",
                "content": "Consider creating a dedicated study space free from distractions to improve focus."
            }]
        
        try:
            # Extract key information for AI prompt
            skill_summary = []
            for area in improvement_areas[:5]:  # Limit to top 5 areas
                skill_summary.append({
                    "name": area['name'],
                    "type": area['type'],
                    "current_score": area['current_score'],
                    "priority": area['priority']
                })
            
            # Format the prompt
            prompt = f"""Generate personalized study suggestions for a student who needs to improve in the following areas:

{json.dumps(skill_summary, indent=2)}

Provide 3-5 specific, actionable suggestions that would help this student improve their understanding and performance in these areas. 
Each suggestion should include:
1. A specific learning strategy or technique
2. An explanation of why this technique would be effective for these particular areas
3. A concrete example of how to apply it

Format each suggestion as a JSON object with "title", "content", and "technique" fields.
Provide your entire response as a valid JSON array of suggestion objects.
"""
            
            # Call Gemini service
            response = self.gemini_service.generate_content(prompt)
            
            # Parse the response
            if hasattr(response, 'text'):
                text = response.text.strip()
            elif isinstance(response, dict) and 'text' in response:
                text = response['text'].strip()
            elif isinstance(response, str):
                text = response.strip()
            else:
                text = ""
            
            # Try to extract JSON from the response
            try:
                # Find JSON array in response
                json_text = text
                if not json_text.startswith('['):
                    # Try to extract JSON from markdown code block
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(1)
                    else:
                        # Try to find array within text
                        json_match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
                        if json_match:
                            json_text = json_match.group(0)
                
                # Parse JSON
                suggestions = json.loads(json_text)
                
                # Ensure proper format
                formatted_suggestions = []
                for suggestion in suggestions:
                    formatted_suggestions.append({
                        "type": suggestion.get('type', 'strategy'),
                        "title": suggestion.get('title', 'Study Suggestion'),
                        "content": suggestion.get('content', ''),
                        "technique": suggestion.get('technique', '')
                    })
                
                return formatted_suggestions
                
            except Exception as e:
                log_system_event(f"Error parsing AI suggestions: {str(e)}")
                
                # Fall back to generic suggestions
                return [
                    {
                        "type": "strategy",
                        "title": "Active Recall",
                        "content": "Instead of passively re-reading notes, practice actively recalling information. This strengthens memory and identifies gaps in understanding.",
                        "technique": "Test yourself after studying each topic, writing down everything you remember before checking your notes."
                    },
                    {
                        "type": "strategy",
                        "title": "Spaced Repetition",
                        "content": "Review material at increasing intervals to enhance long-term retention and combat forgetting.",
                        "technique": "Use flashcards and review them after 1 day, then 3 days, then 7 days, and so on."
                    },
                    {
                        "type": "focus",
                        "title": "Pomodoro Technique",
                        "content": "Work in focused intervals (typically 25 minutes) followed by short breaks to maintain concentration and productivity.",
                        "technique": "Set a timer for 25 minutes of distraction-free study, then take a 5-minute break."
                    }
                ]
        except Exception as e:
            log_system_event(f"Error generating AI suggestions: {str(e)}")
            return []
    
    def _create_study_schedule(self, improvement_areas: List[Dict],
                             course_data: List[Dict] = None) -> Dict:
        """Create a recommended weekly study schedule"""
        # Get top improvement areas
        top_areas = improvement_areas[:min(len(improvement_areas), 5)]
        
        # Calculate time needed for each area based on priority
        time_per_area = {}
        for area in top_areas:
            if area['priority'] == 'high':
                time_per_area[area['name']] = 3.0  # 3 hours per week
            elif area['priority'] == 'medium':
                time_per_area[area['name']] = 2.0  # 2 hours per week
            else:
                time_per_area[area['name']] = 1.0  # 1 hour per week
        
        # Create daily schedule
        schedule = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Get course schedule if available
        course_schedule = {}
        if course_data:
            for course in course_data:
                if 'schedule' in course:
                    for session in course['schedule']:
                        day = session.get('day')
                        if day:
                            if day not in course_schedule:
                                course_schedule[day] = []
                            course_schedule[day].append({
                                'course': course.get('name', 'Course'),
                                'start_time': session.get('start_time'),
                                'end_time': session.get('end_time')
                            })
        
        # Distribute study time throughout the week
        remaining_time = dict(time_per_area)
        area_names = list(remaining_time.keys())
        
        for day in days:
            schedule[day] = []
            
            # Add course sessions first
            if day in course_schedule:
                for session in course_schedule[day]:
                    schedule[day].append({
                        'type': 'class',
                        'title': session['course'],
                        'start_time': session['start_time'],
                        'end_time': session['end_time']
                    })
            
            # Add study sessions
            if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                # Weekday - add 1-2 study sessions
                max_sessions = 2
            else:
                # Weekend - add more study sessions
                max_sessions = 3
            
            added_sessions = 0
            for area in area_names:
                if remaining_time[area] > 0 and added_sessions < max_sessions:
                    # Add a study session for this area
                    session_length = min(remaining_time[area], 1.0)  # Maximum 1 hour per session
                    
                    # Find area details
                    area_details = next((a for a in improvement_areas if a['name'] == area), None)
                    area_type = area_details['type'] if area_details else 'topic'
                    
                    schedule[day].append({
                        'type': 'study',
                        'title': f"Study {area}",
                        'area': area,
                        'area_type': area_type,
                        'duration': session_length,
                        'suggested_time': '17:00' if added_sessions == 0 else '19:00'  # After classes
                    })
                    
                    remaining_time[area] -= session_length
                    added_sessions += 1
        
        return {
            "weekly_schedule": schedule,
            "total_study_hours": sum(time_per_area.values()),
            "focus_areas": [{"name": area, "hours": hours} for area, hours in time_per_area.items()]
        }
    
    def _save_recommendations(self, student_id: str, recommendations: Dict) -> None:
        """Save recommendations for future reference"""
        try:
            # Create recommendations directory if it doesn't exist
            if not os.path.exists(self.recommendations_dir):
                os.makedirs(self.recommendations_dir)
            
            # Path to save recommendations
            file_path = os.path.join(self.recommendations_dir, f"{student_id}.json")
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, indent=2)
                
        except Exception as e:
            log_system_event(f"Error saving recommendations: {str(e)}")
    
    def _load_resources(self) -> List[Dict]:
        """Load learning resources from database file"""
        try:
            # Path to resources database
            resources_file = os.path.join(self.resources_dir, "resources.json")
            
            if os.path.exists(resources_file):
                with open(resources_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create empty database if it doesn't exist
                return []
                
        except Exception as e:
            log_system_event(f"Error loading resources: {str(e)}")
            return []
    
    def _save_resources(self) -> None:
        """Save resources to database file"""
        try:
            # Create resources directory if it doesn't exist
            if not os.path.exists(self.resources_dir):
                os.makedirs(self.resources_dir)
            
            # Path to resources database
            resources_file = os.path.join(self.resources_dir, "resources.json")
            
            # Save to file
            with open(resources_file, 'w', encoding='utf-8') as f:
                json.dump(self.resources, f, indent=2)
                
        except Exception as e:
            log_system_event(f"Error saving resources: {str(e)}") 