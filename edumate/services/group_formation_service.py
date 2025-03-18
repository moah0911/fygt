import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
from ..utils.logger import log_system_event

class GroupFormationService:
    """Service for creating optimal student groups based on various criteria and algorithms"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the group formation service
        
        Args:
            data_dir: Directory for storing data
        """
        self.data_dir = data_dir
        self.groups_dir = os.path.join(data_dir, 'groups')
        self.history_dir = os.path.join(data_dir, 'groups', 'history')
        
        # Create directories if they don't exist
        for directory in [self.groups_dir, self.history_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def create_groups(self, course_id: str, students: List[Dict], 
                    group_size: int = 4, 
                    algorithm: str = 'balanced', 
                    criteria: List[str] = None, 
                    constraints: Dict = None) -> Dict:
        """
        Create student groups based on specified algorithm and criteria
        
        Args:
            course_id: ID of the course
            students: List of student data dictionaries
            group_size: Target size for each group
            algorithm: Algorithm to use for group formation
                - 'random': Random assignment
                - 'balanced': Balance groups based on criteria
                - 'similar': Group similar students
                - 'diverse': Maximize diversity within groups
                - 'performance': Optimize for performance
            criteria: List of student attributes to consider
            constraints: Additional constraints for group formation
            
        Returns:
            Dictionary containing group assignments and metadata
        """
        try:
            if not students:
                return {
                    "status": "error",
                    "message": "No students provided"
                }
                
            if group_size < 2:
                return {
                    "status": "error", 
                    "message": "Group size must be at least 2"
                }
                
            # Default criteria if none provided
            if not criteria:
                criteria = ['skill_level']
                
            # Use selected algorithm for group formation
            if algorithm == 'random':
                groups = self._create_random_groups(students, group_size)
            elif algorithm == 'balanced':
                groups = self._create_balanced_groups(students, group_size, criteria)
            elif algorithm == 'similar':
                groups = self._create_similar_groups(students, group_size, criteria)
            elif algorithm == 'diverse':
                groups = self._create_diverse_groups(students, group_size, criteria)
            elif algorithm == 'performance':
                groups = self._create_performance_groups(students, group_size, criteria)
            else:
                # Default to balanced if algorithm not recognized
                groups = self._create_balanced_groups(students, group_size, criteria)
                
            # Apply any additional constraints
            if constraints:
                groups = self._apply_constraints(groups, students, constraints)
                
            # Create assignment metadata
            assignment = {
                "course_id": course_id,
                "created_at": datetime.now().isoformat(),
                "algorithm": algorithm,
                "criteria": criteria,
                "target_group_size": group_size,
                "constraints": constraints,
                "groups": groups,
                "stats": self._calculate_group_stats(groups, students, criteria)
            }
            
            # Save assignment
            assignment_id = f"{course_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self._save_group_assignment(assignment_id, assignment)
            
            # Add assignment ID to result
            assignment["assignment_id"] = assignment_id
            assignment["status"] = "success"
            
            return assignment
            
        except Exception as e:
            log_system_event(f"Error creating groups: {str(e)}")
            return {
                "status": "error",
                "message": f"Error creating groups: {str(e)}"
            }
    
    def get_group_assignment(self, assignment_id: str) -> Dict:
        """
        Get a specific group assignment
        
        Args:
            assignment_id: ID of the group assignment
            
        Returns:
            Dictionary containing group assignment data
        """
        try:
            # Path to assignment data
            assignment_path = os.path.join(self.groups_dir, f"{assignment_id}.json")
            
            if os.path.exists(assignment_path):
                with open(assignment_path, 'r', encoding='utf-8') as f:
                    assignment = json.load(f)
                return assignment
            else:
                return {
                    "status": "error",
                    "message": f"Group assignment {assignment_id} not found"
                }
                
        except Exception as e:
            log_system_event(f"Error retrieving group assignment: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving group assignment: {str(e)}"
            }
    
    def get_course_group_history(self, course_id: str) -> Dict:
        """
        Get history of group assignments for a course
        
        Args:
            course_id: ID of the course
            
        Returns:
            Dictionary containing group assignment history
        """
        try:
            # List all assignments
            assignments = []
            
            for filename in os.listdir(self.groups_dir):
                if filename.endswith('.json') and filename.startswith(f"{course_id}_"):
                    assignment_id = filename[:-5]  # Remove .json extension
                    
                    with open(os.path.join(self.groups_dir, filename), 'r', encoding='utf-8') as f:
                        assignment = json.load(f)
                    
                    # Add summary to list
                    assignments.append({
                        "assignment_id": assignment_id,
                        "created_at": assignment.get("created_at"),
                        "algorithm": assignment.get("algorithm"),
                        "group_count": len(assignment.get("groups", [])),
                        "target_group_size": assignment.get("target_group_size")
                    })
            
            # Sort by creation date (newest first)
            assignments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return {
                "status": "success",
                "course_id": course_id,
                "assignment_count": len(assignments),
                "assignments": assignments
            }
                
        except Exception as e:
            log_system_event(f"Error retrieving group history: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving group history: {str(e)}"
            }
    
    def get_student_groups(self, student_id: str) -> Dict:
        """
        Get all groups a student has been assigned to
        
        Args:
            student_id: ID of the student
            
        Returns:
            Dictionary containing student's group assignments
        """
        try:
            # Find all group assignments containing this student
            student_assignments = []
            
            for filename in os.listdir(self.groups_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(self.groups_dir, filename), 'r', encoding='utf-8') as f:
                        assignment = json.load(f)
                    
                    # Check if student is in any group
                    student_found = False
                    group_index = -1
                    
                    for i, group in enumerate(assignment.get("groups", [])):
                        for member in group.get("members", []):
                            if member.get("id") == student_id:
                                student_found = True
                                group_index = i
                                break
                        if student_found:
                            break
                    
                    if student_found:
                        assignment_id = filename[:-5]  # Remove .json extension
                        student_assignments.append({
                            "assignment_id": assignment_id,
                            "course_id": assignment.get("course_id"),
                            "created_at": assignment.get("created_at"),
                            "group_number": group_index + 1,
                            "group_members": assignment.get("groups", [])[group_index].get("members", [])
                        })
            
            # Sort by creation date (newest first)
            student_assignments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            
            return {
                "status": "success",
                "student_id": student_id,
                "assignment_count": len(student_assignments),
                "assignments": student_assignments
            }
                
        except Exception as e:
            log_system_event(f"Error retrieving student groups: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving student groups: {str(e)}"
            }
    
    def update_group_feedback(self, assignment_id: str, group_feedback: List[Dict]) -> Dict:
        """
        Update group assignment with performance feedback
        
        Args:
            assignment_id: ID of the group assignment
            group_feedback: List of feedback data for each group
            
        Returns:
            Dictionary containing updated assignment data
        """
        try:
            # Get current assignment data
            assignment = self.get_group_assignment(assignment_id)
            
            if assignment.get("status") == "error":
                return assignment
            
            # Update groups with feedback
            groups = assignment.get("groups", [])
            
            for feedback in group_feedback:
                group_index = feedback.get("group_index")
                if 0 <= group_index < len(groups):
                    groups[group_index]["feedback"] = feedback.get("feedback")
                    groups[group_index]["performance_score"] = feedback.get("performance_score")
                    groups[group_index]["collaboration_score"] = feedback.get("collaboration_score")
            
            # Update assignment
            assignment["groups"] = groups
            assignment["last_updated"] = datetime.now().isoformat()
            
            # Save updated assignment
            self._save_group_assignment(assignment_id, assignment)
            
            return {
                "status": "success",
                "message": "Group feedback updated successfully",
                "assignment_id": assignment_id
            }
                
        except Exception as e:
            log_system_event(f"Error updating group feedback: {str(e)}")
            return {
                "status": "error",
                "message": f"Error updating group feedback: {str(e)}"
            }
    
    def analyze_group_effectiveness(self, course_id: str) -> Dict:
        """
        Analyze the effectiveness of different grouping strategies
        
        Args:
            course_id: ID of the course
            
        Returns:
            Dictionary containing analysis of group effectiveness
        """
        try:
            # Get all assignments for this course
            history = self.get_course_group_history(course_id)
            
            if history.get("status") == "error":
                return history
            
            # Get assignments with feedback
            assignments_with_feedback = []
            
            for assignment_summary in history.get("assignments", []):
                assignment_id = assignment_summary.get("assignment_id")
                assignment = self.get_group_assignment(assignment_id)
                
                # Check if groups have feedback
                has_feedback = False
                for group in assignment.get("groups", []):
                    if "performance_score" in group:
                        has_feedback = True
                        break
                
                if has_feedback:
                    assignments_with_feedback.append(assignment)
            
            # Analyze by algorithm
            algorithm_performance = defaultdict(list)
            
            for assignment in assignments_with_feedback:
                algorithm = assignment.get("algorithm")
                
                # Calculate average performance score
                total_score = 0
                group_count = 0
                
                for group in assignment.get("groups", []):
                    if "performance_score" in group:
                        total_score += group.get("performance_score", 0)
                        group_count += 1
                
                if group_count > 0:
                    avg_score = total_score / group_count
                    algorithm_performance[algorithm].append(avg_score)
            
            # Calculate average performance by algorithm
            algorithm_analysis = []
            
            for algorithm, scores in algorithm_performance.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    algorithm_analysis.append({
                        "algorithm": algorithm,
                        "average_score": avg_score,
                        "group_count": len(scores)
                    })
            
            # Sort by average score (highest first)
            algorithm_analysis.sort(key=lambda x: x.get("average_score", 0), reverse=True)
            
            return {
                "status": "success",
                "course_id": course_id,
                "analysis_date": datetime.now().isoformat(),
                "assignments_analyzed": len(assignments_with_feedback),
                "algorithm_analysis": algorithm_analysis
            }
                
        except Exception as e:
            log_system_event(f"Error analyzing group effectiveness: {str(e)}")
            return {
                "status": "error",
                "message": f"Error analyzing group effectiveness: {str(e)}"
            }
    
    def recommend_grouping_strategy(self, course_id: str, students: List[Dict]) -> Dict:
        """
        Recommend the best grouping strategy based on past performance
        
        Args:
            course_id: ID of the course
            students: List of student data dictionaries
            
        Returns:
            Dictionary containing recommended grouping strategy
        """
        try:
            # Analyze past group effectiveness
            effectiveness = self.analyze_group_effectiveness(course_id)
            
            if effectiveness.get("status") == "error" or len(effectiveness.get("algorithm_analysis", [])) == 0:
                # No past data, use balanced algorithm as default
                recommendation = {
                    "algorithm": "balanced",
                    "criteria": ["skill_level", "learning_style"],
                    "group_size": 4,
                    "confidence": "low",
                    "reasoning": "No past group data available. Using balanced algorithm as default."
                }
            else:
                # Get best performing algorithm
                best_algorithm = effectiveness.get("algorithm_analysis", [])[0]
                
                # Determine ideal group size based on number of students
                student_count = len(students)
                
                if student_count <= 9:
                    ideal_size = 3
                elif student_count <= 16:
                    ideal_size = 4
                else:
                    ideal_size = 5
                
                # Determine criteria based on available student data
                available_criteria = []
                
                # Check first student to see what fields are available
                if students and len(students) > 0:
                    student = students[0]
                    
                    if "skill_level" in student or "skills" in student:
                        available_criteria.append("skill_level")
                    
                    if "learning_style" in student:
                        available_criteria.append("learning_style")
                    
                    if "interests" in student:
                        available_criteria.append("interests")
                    
                    if "performance" in student or "grades" in student:
                        available_criteria.append("performance")
                
                if not available_criteria:
                    available_criteria = ["skill_level"]  # Default
                
                recommendation = {
                    "algorithm": best_algorithm.get("algorithm"),
                    "criteria": available_criteria,
                    "group_size": ideal_size,
                    "confidence": "high" if best_algorithm.get("group_count", 0) > 3 else "medium",
                    "reasoning": f"Based on analysis of {best_algorithm.get('group_count', 0)} previous group assignments."
                }
            
            return {
                "status": "success",
                "course_id": course_id,
                "recommendation": recommendation,
                "student_count": len(students)
            }
                
        except Exception as e:
            log_system_event(f"Error recommending grouping strategy: {str(e)}")
            return {
                "status": "error",
                "message": f"Error recommending grouping strategy: {str(e)}"
            }
    
    def _create_random_groups(self, students: List[Dict], group_size: int) -> List[Dict]:
        """Create random groups of specified size"""
        # Shuffle students randomly
        shuffled_students = random.sample(students, len(students))
        
        # Create groups
        groups = []
        num_groups = (len(shuffled_students) + group_size - 1) // group_size  # Ceiling division
        
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = min(start_idx + group_size, len(shuffled_students))
            
            group_members = shuffled_students[start_idx:end_idx]
            
            groups.append({
                "name": f"Group {i + 1}",
                "members": group_members
            })
        
        return groups
    
    def _create_balanced_groups(self, students: List[Dict], group_size: int, criteria: List[str]) -> List[Dict]:
        """Create groups with balanced distribution of specified criteria"""
        # Calculate number of groups
        num_groups = (len(students) + group_size - 1) // group_size  # Ceiling division
        
        # Initialize empty groups
        groups = [{"name": f"Group {i + 1}", "members": []} for i in range(num_groups)]
        
        # Score students on criteria
        scored_students = self._score_students_on_criteria(students, criteria)
        
        # Sort students by score
        scored_students.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute students using serpentine pattern
        for i, (student, _) in enumerate(scored_students):
            group_idx = i % num_groups if (i // num_groups) % 2 == 0 else num_groups - 1 - (i % num_groups)
            groups[group_idx]["members"].append(student)
        
        return groups
    
    def _create_similar_groups(self, students: List[Dict], group_size: int, criteria: List[str]) -> List[Dict]:
        """Create groups with similar students based on criteria"""
        # Calculate number of groups
        num_groups = (len(students) + group_size - 1) // group_size  # Ceiling division
        
        # Score students on criteria
        scored_students = self._score_students_on_criteria(students, criteria)
        
        # Sort students by score
        scored_students.sort(key=lambda x: x[1])
        
        # Initialize groups
        groups = [{"name": f"Group {i + 1}", "members": []} for i in range(num_groups)]
        
        # Distribute students sequentially
        for i, (student, _) in enumerate(scored_students):
            group_idx = i // group_size
            if group_idx < num_groups:
                groups[group_idx]["members"].append(student)
            else:
                # Distribute remaining students
                groups[i % num_groups]["members"].append(student)
        
        return groups
    
    def _create_diverse_groups(self, students: List[Dict], group_size: int, criteria: List[str]) -> List[Dict]:
        """Create groups with maximum diversity based on criteria"""
        # Calculate number of groups
        num_groups = (len(students) + group_size - 1) // group_size  # Ceiling division
        
        # Score students on multiple criteria
        multi_scored_students = []
        
        for student in students:
            criterion_scores = []
            
            for criterion in criteria:
                score = self._get_criterion_score(student, criterion)
                criterion_scores.append(score)
            
            multi_scored_students.append((student, criterion_scores))
        
        # Cluster students to maximize diversity within groups
        return self._cluster_for_diversity(multi_scored_students, num_groups, group_size)
    
    def _create_performance_groups(self, students: List[Dict], group_size: int, criteria: List[str]) -> List[Dict]:
        """Create groups optimized for performance"""
        # Calculate number of groups
        num_groups = (len(students) + group_size - 1) // group_size  # Ceiling division
        
        # Prioritize performance and skill level
        performance_criteria = [c for c in criteria if c in ["performance", "skill_level"]]
        if not performance_criteria:
            performance_criteria = ["skill_level"]
        
        # Score students based on performance
        performance_scores = self._score_students_on_criteria(students, performance_criteria)
        
        # Sort students by performance score
        performance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize groups
        groups = [{"name": f"Group {i + 1}", "members": []} for i in range(num_groups)]
        
        # Distribute high performers among groups
        top_performers_count = min(num_groups, len(performance_scores))
        for i in range(top_performers_count):
            groups[i]["members"].append(performance_scores[i][0])
        
        # Distribute remaining students to balance groups
        remaining_students = [s[0] for s in performance_scores[top_performers_count:]]
        
        # Create balanced groups from remaining students
        if remaining_students:
            remaining_balanced = self._create_balanced_groups(
                remaining_students, 
                (len(remaining_students) + num_groups - 1) // num_groups,  # Ceiling division
                criteria
            )
            
            # Merge with existing groups
            for i, balanced_group in enumerate(remaining_balanced):
                if i < len(groups):
                    groups[i]["members"].extend(balanced_group["members"])
        
        return groups
    
    def _score_students_on_criteria(self, students: List[Dict], criteria: List[str]) -> List[Tuple[Dict, float]]:
        """Score students based on specified criteria"""
        scored_students = []
        
        for student in students:
            total_score = 0.0
            
            for criterion in criteria:
                score = self._get_criterion_score(student, criterion)
                total_score += score
            
            # Normalize by number of criteria
            if criteria:
                total_score /= len(criteria)
            
            scored_students.append((student, total_score))
        
        return scored_students
    
    def _get_criterion_score(self, student: Dict, criterion: str) -> float:
        """Get normalized score (0-1) for a specific criterion"""
        if criterion == "skill_level":
            # Check different possible fields for skill level
            if "skill_level" in student:
                return float(student["skill_level"]) / 10.0 if isinstance(student["skill_level"], (int, float)) else 0.5
            elif "skills" in student and isinstance(student["skills"], list):
                # Average skill scores if available
                skills = student["skills"]
                if skills and all(isinstance(s.get("proficiency_score"), (int, float)) for s in skills):
                    return sum(s.get("proficiency_score", 0) for s in skills) / len(skills)
                return 0.5
            else:
                return 0.5
        
        elif criterion == "performance":
            # Check different possible fields for performance
            if "performance" in student:
                return float(student["performance"]) / 100.0
            elif "grades" in student and isinstance(student["grades"], list):
                # Average grades if available
                grades = student["grades"]
                if grades and all(isinstance(g.get("score"), (int, float)) for g in grades):
                    return sum(g.get("score", 0) / 100.0 for g in grades) / len(grades)
                return 0.5
            else:
                return 0.5
        
        elif criterion == "learning_style":
            # Map learning style to numeric value (for ordering purposes)
            if "learning_style" in student:
                style = student["learning_style"].lower() if isinstance(student["learning_style"], str) else ""
                style_map = {
                    "visual": 0.2,
                    "auditory": 0.4,
                    "reading": 0.6,
                    "kinesthetic": 0.8
                }
                return style_map.get(style, 0.5)
            return 0.5
        
        elif criterion == "interests":
            # Use hash of interests for consistent random-like distribution
            if "interests" in student and isinstance(student["interests"], list):
                interests = sorted(student["interests"])
                return (hash(tuple(interests)) % 1000) / 1000.0
            return 0.5
        
        # Default score
        return 0.5
    
    def _cluster_for_diversity(self, multi_scored_students: List[Tuple[Dict, List[float]]], 
                               num_groups: int, group_size: int) -> List[Dict]:
        """Use clustering to create diverse groups"""
        # Initialize empty groups
        groups = [{"name": f"Group {i + 1}", "members": []} for i in range(num_groups)]
        
        # Convert multi-criteria scores to numpy array
        if not multi_scored_students:
            return groups
            
        # Extract students and their feature vectors
        students = [s[0] for s in multi_scored_students]
        features = np.array([s[1] for s in multi_scored_students])
        
        # Simple algorithm: alternate adding highest and lowest scoring students
        while students:
            for group in groups:
                if not students:
                    break
                
                # Find student furthest from current group members
                if not group["members"]:
                    # For empty group, add random student
                    idx = random.randint(0, len(students) - 1)
                else:
                    # Calculate average feature vector of current group members
                    group_indices = [students.index(m) for m in group["members"] if m in students]
                    if not group_indices:
                        idx = random.randint(0, len(students) - 1)
                    else:
                        group_features = np.mean([features[i] for i in group_indices], axis=0)
                        
                        # Find student with most different features
                        distances = [np.linalg.norm(features[i] - group_features) for i in range(len(students))]
                        idx = distances.index(max(distances))
                
                # Add student to group
                group["members"].append(students[idx])
                
                # Remove student from candidates
                students.pop(idx)
                features = np.delete(features, idx, axis=0)
        
        return groups
    
    def _apply_constraints(self, groups: List[Dict], students: List[Dict], constraints: Dict) -> List[Dict]:
        """Apply additional constraints to group formation"""
        if not constraints:
            return groups
        
        # Handle "must be together" constraint
        if "together" in constraints and isinstance(constraints["together"], list):
            for pair in constraints["together"]:
                if len(pair) == 2:
                    student1_id, student2_id = pair
                    # Find groups containing these students
                    group1_idx = -1
                    group2_idx = -1
                    
                    for i, group in enumerate(groups):
                        for member in group["members"]:
                            if member.get("id") == student1_id:
                                group1_idx = i
                            elif member.get("id") == student2_id:
                                group2_idx = i
                    
                    # Move students if they're in different groups
                    if group1_idx != -1 and group2_idx != -1 and group1_idx != group2_idx:
                        # Find student2 in group2
                        for j, member in enumerate(groups[group2_idx]["members"]):
                            if member.get("id") == student2_id:
                                # Move student2 to group1
                                student2 = groups[group2_idx]["members"].pop(j)
                                groups[group1_idx]["members"].append(student2)
                                break
        
        # Handle "must be separate" constraint
        if "separate" in constraints and isinstance(constraints["separate"], list):
            for pair in constraints["separate"]:
                if len(pair) == 2:
                    student1_id, student2_id = pair
                    # Find if they're in the same group
                    for i, group in enumerate(groups):
                        student1_idx = -1
                        student2_idx = -1
                        
                        for j, member in enumerate(group["members"]):
                            if member.get("id") == student1_id:
                                student1_idx = j
                            elif member.get("id") == student2_id:
                                student2_idx = j
                        
                        # If both in same group, move one to another group
                        if student1_idx != -1 and student2_idx != -1:
                            # Find group with fewest members
                            target_group_idx = min(range(len(groups)), key=lambda x: len(groups[x]["members"]))
                            
                            # Don't move if target is the same group
                            if target_group_idx != i:
                                # Move student2 to target group
                                student2 = group["members"].pop(student2_idx)
                                groups[target_group_idx]["members"].append(student2)
                            break
        
        return groups
    
    def _calculate_group_stats(self, groups: List[Dict], students: List[Dict], criteria: List[str]) -> Dict:
        """Calculate statistics for the group assignment"""
        stats = {
            "total_students": len(students),
            "num_groups": len(groups),
            "average_group_size": len(students) / len(groups) if groups else 0,
            "criteria_balance": {}
        }
        
        # Calculate statistics for each criterion
        for criterion in criteria:
            group_averages = []
            
            for group in groups:
                criterion_scores = []
                for member in group["members"]:
                    score = self._get_criterion_score(member, criterion)
                    criterion_scores.append(score)
                
                if criterion_scores:
                    group_avg = sum(criterion_scores) / len(criterion_scores)
                    group_averages.append(group_avg)
            
            if group_averages:
                # Calculate standard deviation to measure balance
                avg = sum(group_averages) / len(group_averages)
                variance = sum((x - avg) ** 2 for x in group_averages) / len(group_averages)
                std_dev = variance ** 0.5
                
                stats["criteria_balance"][criterion] = {
                    "average": avg,
                    "std_dev": std_dev,
                    "min": min(group_averages),
                    "max": max(group_averages)
                }
        
        return stats
    
    def _save_group_assignment(self, assignment_id: str, assignment: Dict) -> None:
        """Save group assignment to file"""
        try:
            # Path to save assignment
            file_path = os.path.join(self.groups_dir, f"{assignment_id}.json")
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(assignment, f, indent=2)
                
        except Exception as e:
            log_system_event(f"Error saving group assignment: {str(e)}") 