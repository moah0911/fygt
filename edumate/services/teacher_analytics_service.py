import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from ..utils.logger import log_system_event
import json

class TeacherAnalyticsService:
    """Service for providing enhanced analytics for teachers"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.reports_dir = os.path.join(data_dir, 'reports', 'teacher_analytics')
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    def generate_class_dashboard(self, course_id: str, submissions: List[Dict], 
                                students: List[Dict]) -> Dict:
        """
        Generate comprehensive analytics dashboard for a class
        
        Args:
            course_id: The ID of the course
            submissions: List of submissions for the course
            students: List of students in the course
            
        Returns:
            Dict containing analytics dashboard data
        """
        try:
            if not submissions or not students:
                return {
                    "status": "error",
                    "message": "Not enough data to generate analytics"
                }
            
            # Convert submissions to DataFrame for analysis
            submissions_df = pd.DataFrame(submissions)
            
            # Basic performance metrics
            performance_metrics = self._calculate_performance_metrics(submissions_df)
            
            # Identify struggling students
            struggling_students = self._identify_struggling_students(submissions_df, students)
            
            # Skill gap analysis
            skill_gaps = self._analyze_skill_gaps(submissions_df)
            
            # Assignment difficulty analysis
            assignment_analysis = self._analyze_assignment_difficulty(submissions_df)
            
            # Time management analysis
            time_analysis = self._analyze_submission_timing(submissions_df)
            
            # Generate performance distribution chart
            chart_path = self._generate_performance_distribution_chart(
                submissions_df, course_id)
            
            # Generate skill heatmap
            skill_heatmap_path = self._generate_skill_heatmap(submissions_df, course_id)
            
            # Generate recommended actions
            actions = self._generate_recommended_actions(
                struggling_students, 
                skill_gaps,
                assignment_analysis
            )
            
            return {
                "status": "success",
                "performance_metrics": performance_metrics,
                "struggling_students": struggling_students,
                "skill_gaps": skill_gaps,
                "assignment_analysis": assignment_analysis,
                "time_analysis": time_analysis,
                "charts": {
                    "performance_distribution": chart_path,
                    "skill_heatmap": skill_heatmap_path
                },
                "recommended_actions": actions,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_system_event(f"Error generating class dashboard: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating analytics: {str(e)}"
            }
    
    def generate_student_comparison(self, student_id: str, course_id: str, 
                                   submissions: List[Dict]) -> Dict:
        """
        Generate comparative analysis of a student against class performance
        
        Args:
            student_id: The ID of the student
            course_id: The ID of the course
            submissions: List of all submissions for the course
            
        Returns:
            Dict containing comparative analytics
        """
        try:
            submissions_df = pd.DataFrame(submissions)
            
            # Filter student's submissions
            student_submissions = submissions_df[submissions_df['student_id'] == student_id]
            other_submissions = submissions_df[submissions_df['student_id'] != student_id]
            
            if student_submissions.empty:
                return {
                    "status": "error",
                    "message": "No submissions found for this student"
                }
            
            # Calculate average scores
            student_avg = student_submissions['score'].mean() if 'score' in student_submissions else 0
            class_avg = other_submissions['score'].mean() if 'score' in other_submissions else 0
            
            # Calculate percentile
            percentile = 0
            if not other_submissions.empty and 'score' in other_submissions:
                percentile = sum(student_avg > other_submissions['score']) / len(other_submissions) * 100
            
            # Analyze strength areas
            strength_areas = self._identify_student_strengths(student_submissions)
            
            # Analyze improvement areas
            improvement_areas = self._identify_student_improvements(student_submissions)
            
            # Generate comparison chart
            chart_path = self._generate_student_comparison_chart(
                student_submissions, other_submissions, student_id, course_id)
            
            return {
                "status": "success",
                "student_id": student_id,
                "student_average": round(student_avg, 2),
                "class_average": round(class_avg, 2),
                "percentile": round(percentile, 2),
                "strength_areas": strength_areas,
                "improvement_areas": improvement_areas,
                "charts": {
                    "comparison": chart_path
                },
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_system_event(f"Error generating student comparison: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating comparison: {str(e)}"
            }
    
    def generate_longitudinal_analysis(self, course_id: str, submissions: List[Dict], 
                                       time_period: int = 30) -> Dict:
        """
        Generate longitudinal analysis of class performance over time
        
        Args:
            course_id: The ID of the course
            submissions: List of submissions for the course
            time_period: Number of days to analyze
            
        Returns:
            Dict containing longitudinal analysis
        """
        try:
            submissions_df = pd.DataFrame(submissions)
            
            if 'submitted_at' not in submissions_df.columns:
                return {
                    "status": "error",
                    "message": "Submission date information not available"
                }
            
            # Convert date strings to datetime
            submissions_df['submitted_at'] = pd.to_datetime(submissions_df['submitted_at'])
            
            # Filter submissions within time period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_period)
            mask = (submissions_df['submitted_at'] >= start_date) & (submissions_df['submitted_at'] <= end_date)
            recent_submissions = submissions_df.loc[mask]
            
            if recent_submissions.empty:
                return {
                    "status": "error",
                    "message": f"No submissions found in the last {time_period} days"
                }
            
            # Group by date and calculate average score
            recent_submissions['date'] = recent_submissions['submitted_at'].dt.date
            daily_avg = recent_submissions.groupby('date')['score'].mean().reset_index()
            
            # Calculate trend
            if len(daily_avg) > 1:
                x = np.arange(len(daily_avg))
                y = daily_avg['score'].values
                z = np.polyfit(x, y, 1)
                trend = z[0]  # Slope of the trend line
            else:
                trend = 0
            
            # Generate trend chart
            chart_path = self._generate_trend_chart(daily_avg, course_id)
            
            return {
                "status": "success",
                "trend": round(trend, 4),
                "trend_direction": "improving" if trend > 0 else "declining" if trend < 0 else "stable",
                "daily_averages": daily_avg.to_dict('records'),
                "charts": {
                    "trend": chart_path
                },
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_system_event(f"Error generating longitudinal analysis: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating analysis: {str(e)}"
            }
    
    def identify_at_risk_students(self, course_id: str, submissions: List[Dict], 
                                 students: List[Dict], threshold: float = 0.6) -> Dict:
        """
        Identify students who are at risk of falling behind
        
        Args:
            course_id: The ID of the course
            submissions: List of submissions for the course
            students: List of students in the course
            threshold: Score threshold for identifying at-risk students
            
        Returns:
            Dict containing list of at-risk students and recommendations
        """
        try:
            submissions_df = pd.DataFrame(submissions)
            
            # Calculate average score by student
            student_avg = submissions_df.groupby('student_id')['score'].mean().reset_index()
            
            # Identify at-risk students
            at_risk_students = student_avg[student_avg['score'] < threshold * 100]
            
            # Get complete student information
            at_risk_info = []
            for _, row in at_risk_students.iterrows():
                student = next((s for s in students if s['id'] == row['student_id']), None)
                if student:
                    at_risk_info.append({
                        "student_id": row['student_id'],
                        "name": student.get('name', 'Unknown'),
                        "email": student.get('email', 'Unknown'),
                        "average_score": round(row['score'], 2),
                        "missing_assignments": len(submissions_df[(submissions_df['student_id'] == row['student_id']) & 
                                                  (submissions_df['status'] == 'missing')])
                    })
            
            # Generate recommendations for each student
            for student in at_risk_info:
                student_submissions = submissions_df[submissions_df['student_id'] == student['student_id']]
                student['recommendations'] = self._generate_student_recommendations(student_submissions)
            
            return {
                "status": "success",
                "at_risk_count": len(at_risk_info),
                "at_risk_students": at_risk_info,
                "threshold_used": threshold,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_system_event(f"Error identifying at-risk students: {str(e)}")
            return {
                "status": "error",
                "message": f"Error identifying at-risk students: {str(e)}"
            }
    
    def _calculate_performance_metrics(self, submissions_df: pd.DataFrame) -> Dict:
        """Calculate key performance metrics for the class"""
        metrics = {}
        
        if 'score' in submissions_df.columns:
            metrics['average_score'] = round(submissions_df['score'].mean(), 2)
            metrics['median_score'] = round(submissions_df['score'].median(), 2)
            metrics['std_deviation'] = round(submissions_df['score'].std(), 2)
            metrics['min_score'] = round(submissions_df['score'].min(), 2)
            metrics['max_score'] = round(submissions_df['score'].max(), 2)
        
        metrics['total_submissions'] = len(submissions_df)
        metrics['unique_students'] = submissions_df['student_id'].nunique()
        
        if 'status' in submissions_df.columns:
            metrics['graded_count'] = len(submissions_df[submissions_df['status'].isin(['graded', 'auto-graded'])])
            metrics['pending_count'] = len(submissions_df[submissions_df['status'] == 'submitted'])
            metrics['late_count'] = len(submissions_df[submissions_df['status'] == 'late'])
        
        return metrics
    
    def _identify_struggling_students(self, submissions_df: pd.DataFrame, 
                                     students: List[Dict]) -> List[Dict]:
        """Identify students who are struggling based on their submissions"""
        struggling = []
        
        # Group submissions by student
        student_groups = submissions_df.groupby('student_id')
        
        for student_id, group in student_groups:
            if 'score' not in group.columns or group.empty:
                continue
                
            avg_score = group['score'].mean()
            
            if avg_score < 70:  # Threshold for "struggling"
                student = next((s for s in students if s['id'] == student_id), None)
                if student:
                    struggling.append({
                        "student_id": student_id,
                        "name": student.get('name', 'Unknown'),
                        "average_score": round(avg_score, 2),
                        "submission_count": len(group),
                        "recent_scores": group.sort_values('submitted_at', ascending=False)['score'].head(3).tolist()
                    })
        
        # Sort by average score, lowest first
        struggling.sort(key=lambda x: x['average_score'])
        
        return struggling
    
    def _analyze_skill_gaps(self, submissions_df: pd.DataFrame) -> Dict:
        """Analyze skill gaps based on submission metadata"""
        skill_scores = {}
        
        # Extract skills data from grading metadata
        for _, row in submissions_df.iterrows():
            if 'grading_metadata' in row and row['grading_metadata']:
                metadata = row['grading_metadata']
                
                # Code analysis skills
                if 'code_analysis' in metadata:
                    code_analysis = metadata['code_analysis']
                    for skill, score in code_analysis.items():
                        if isinstance(score, (int, float)):
                            if skill not in skill_scores:
                                skill_scores[skill] = []
                            skill_scores[skill].append(score)
                
                # Question results
                if 'question_results' in metadata:
                    for q_res in metadata['question_results']:
                        if 'category' in q_res and 'score' in q_res:
                            skill = q_res['category']
                            score = q_res['score']
                            if skill not in skill_scores:
                                skill_scores[skill] = []
                            skill_scores[skill].append(score)
        
        # Calculate average scores and identify gaps
        skill_gaps = {}
        for skill, scores in skill_scores.items():
            avg_score = sum(scores) / len(scores)
            skill_gaps[skill] = {
                "average_score": round(avg_score, 2),
                "is_gap": avg_score < 0.6,  # Threshold for identifying a skill gap
                "sample_size": len(scores)
            }
        
        return skill_gaps
    
    def _analyze_assignment_difficulty(self, submissions_df: pd.DataFrame) -> Dict:
        """Analyze assignment difficulty based on scores"""
        assignment_difficulty = {}
        
        # Group by assignment_id
        assignment_groups = submissions_df.groupby('assignment_id')
        
        for assignment_id, group in assignment_groups:
            if 'score' not in group.columns or group.empty:
                continue
                
            avg_score = group['score'].mean()
            std_dev = group['score'].std()
            
            assignment_difficulty[assignment_id] = {
                "average_score": round(avg_score, 2),
                "std_deviation": round(std_dev, 2) if not np.isnan(std_dev) else 0,
                "submission_count": len(group),
                "difficulty_level": self._calculate_difficulty_level(avg_score)
            }
        
        return assignment_difficulty
    
    def _calculate_difficulty_level(self, avg_score: float) -> str:
        """Calculate difficulty level based on average score"""
        if avg_score < 60:
            return "Very Difficult"
        elif avg_score < 70:
            return "Difficult"
        elif avg_score < 80:
            return "Moderate"
        elif avg_score < 90:
            return "Easy"
        else:
            return "Very Easy"
    
    def _analyze_submission_timing(self, submissions_df: pd.DataFrame) -> Dict:
        """Analyze submission timing patterns"""
        timing_analysis = {}
        
        if 'submitted_at' not in submissions_df.columns or 'due_date' not in submissions_df.columns:
            return timing_analysis
        
        submissions_df['submitted_at'] = pd.to_datetime(submissions_df['submitted_at'])
        submissions_df['due_date'] = pd.to_datetime(submissions_df['due_date'])
        
        # Calculate days before deadline
        submissions_df['days_before_deadline'] = (submissions_df['due_date'] - 
                                                 submissions_df['submitted_at']).dt.total_seconds() / (24 * 3600)
        
        # Calculate statistics
        on_time_count = len(submissions_df[submissions_df['days_before_deadline'] >= 0])
        late_count = len(submissions_df[submissions_df['days_before_deadline'] < 0])
        
        if not submissions_df.empty:
            timing_analysis = {
                "on_time_percentage": round((on_time_count / len(submissions_df)) * 100, 2),
                "late_percentage": round((late_count / len(submissions_df)) * 100, 2),
                "average_days_before_deadline": round(submissions_df[submissions_df['days_before_deadline'] >= 0]['days_before_deadline'].mean(), 2),
                "last_minute_percentage": round((len(submissions_df[submissions_df['days_before_deadline'].between(0, 1)]) / len(submissions_df)) * 100, 2)
            }
        
        return timing_analysis
    
    def _generate_performance_distribution_chart(self, submissions_df: pd.DataFrame, 
                                                course_id: str) -> str:
        """Generate a chart showing the distribution of scores"""
        try:
            if 'score' not in submissions_df.columns or submissions_df.empty:
                return ""
            
            plt.figure(figsize=(10, 6))
            sns.histplot(submissions_df['score'], kde=True, bins=10)
            plt.title('Score Distribution')
            plt.xlabel('Score')
            plt.ylabel('Number of Submissions')
            plt.axvline(submissions_df['score'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {submissions_df["score"].mean():.2f}')
            plt.axvline(submissions_df['score'].median(), color='g', linestyle='--', 
                       label=f'Median: {submissions_df["score"].median():.2f}')
            plt.legend()
            
            # Save chart
            chart_path = os.path.join(self.reports_dir, f'score_dist_{course_id}.png')
            plt.savefig(chart_path)
            plt.close()
            
            return chart_path
        except Exception as e:
            log_system_event(f"Error generating performance chart: {str(e)}")
            return ""
    
    def _generate_skill_heatmap(self, submissions_df: pd.DataFrame, course_id: str) -> str:
        """Generate a heatmap showing skill mastery across students"""
        try:
            # Extract skill data
            skill_data = {}
            student_ids = []
            
            for _, row in submissions_df.iterrows():
                if 'student_id' not in row or 'grading_metadata' not in row:
                    continue
                    
                student_id = row['student_id']
                if student_id not in student_ids:
                    student_ids.append(student_id)
                
                if row['grading_metadata']:
                    metadata = row['grading_metadata']
                    
                    # Code analysis skills
                    if 'code_analysis' in metadata:
                        for skill, score in metadata['code_analysis'].items():
                            if isinstance(score, (int, float)):
                                if skill not in skill_data:
                                    skill_data[skill] = {}
                                if student_id not in skill_data[skill]:
                                    skill_data[skill][student_id] = []
                                skill_data[skill][student_id].append(score)
                    
                    # Question results
                    if 'question_results' in metadata:
                        for q_res in metadata['question_results']:
                            if 'category' in q_res and 'score' in q_res:
                                skill = q_res['category']
                                score = q_res['score']
                                if skill not in skill_data:
                                    skill_data[skill] = {}
                                if student_id not in skill_data[skill]:
                                    skill_data[skill][student_id] = []
                                skill_data[skill][student_id].append(score)
            
            if not skill_data or not student_ids:
                return ""
            
            # Create dataframe for heatmap
            heatmap_data = []
            for skill, students in skill_data.items():
                for student_id in student_ids:
                    if student_id in students:
                        avg_score = sum(students[student_id]) / len(students[student_id])
                        heatmap_data.append({
                            'Skill': skill,
                            'Student': student_id,
                            'Score': avg_score
                        })
                    else:
                        heatmap_data.append({
                            'Skill': skill,
                            'Student': student_id,
                            'Score': np.nan
                        })
            
            if not heatmap_data:
                return ""
                
            heatmap_df = pd.DataFrame(heatmap_data)
            pivot_table = heatmap_df.pivot_table(index='Skill', columns='Student', values='Score')
            
            # Generate heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', vmin=0, vmax=1, 
                       linewidths=.5, cbar_kws={'label': 'Score'})
            plt.title('Skill Mastery Heatmap')
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.reports_dir, f'skill_heatmap_{course_id}.png')
            plt.savefig(chart_path)
            plt.close()
            
            return chart_path
        except Exception as e:
            log_system_event(f"Error generating skill heatmap: {str(e)}")
            return ""
    
    def _generate_student_comparison_chart(self, student_submissions: pd.DataFrame, 
                                          other_submissions: pd.DataFrame, 
                                          student_id: str, course_id: str) -> str:
        """Generate a chart comparing student performance to class average"""
        try:
            if 'score' not in student_submissions.columns or student_submissions.empty:
                return ""
            
            # Group by assignment
            student_by_assignment = student_submissions.groupby('assignment_id')['score'].mean()
            class_by_assignment = other_submissions.groupby('assignment_id')['score'].mean()
            
            # Find common assignments
            common_assignments = set(student_by_assignment.index).intersection(set(class_by_assignment.index))
            
            if not common_assignments:
                return ""
            
            # Prepare data for common assignments
            compare_data = pd.DataFrame({
                'Assignment': list(common_assignments),
                'Student': [student_by_assignment[a] for a in common_assignments],
                'Class Average': [class_by_assignment[a] for a in common_assignments]
            })
            
            # Generate chart
            plt.figure(figsize=(12, 6))
            bar_width = 0.35
            opacity = 0.8
            
            index = np.arange(len(compare_data))
            plt.bar(index, compare_data['Student'], bar_width,
                   alpha=opacity, color='b', label='Student')
            plt.bar(index + bar_width, compare_data['Class Average'], bar_width,
                   alpha=opacity, color='g', label='Class Average')
            
            plt.xlabel('Assignment')
            plt.ylabel('Score')
            plt.title('Student vs. Class Average')
            plt.xticks(index + bar_width/2, [f"Asgmt {i+1}" for i in range(len(compare_data))])
            plt.legend()
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.reports_dir, f'student_compare_{student_id}_{course_id}.png')
            plt.savefig(chart_path)
            plt.close()
            
            return chart_path
        except Exception as e:
            log_system_event(f"Error generating comparison chart: {str(e)}")
            return ""
    
    def _generate_trend_chart(self, daily_avg: pd.DataFrame, course_id: str) -> str:
        """Generate a chart showing performance trends over time"""
        try:
            if 'score' not in daily_avg.columns or daily_avg.empty:
                return ""
            
            plt.figure(figsize=(12, 6))
            plt.plot(daily_avg['date'], daily_avg['score'], 'o-', linewidth=2)
            
            # Add trend line
            x = np.arange(len(daily_avg))
            y = daily_avg['score'].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(daily_avg['date'], p(x), "r--", linewidth=1, 
                    label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
            
            plt.xlabel('Date')
            plt.ylabel('Average Score')
            plt.title('Performance Trend Over Time')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.reports_dir, f'trend_{course_id}.png')
            plt.savefig(chart_path)
            plt.close()
            
            return chart_path
        except Exception as e:
            log_system_event(f"Error generating trend chart: {str(e)}")
            return ""
    
    def _identify_student_strengths(self, student_submissions: pd.DataFrame) -> List[Dict]:
        """Identify areas where the student is performing well"""
        strengths = []
        
        try:
            # Check if grading metadata exists
            metadata_exists = False
            for _, row in student_submissions.iterrows():
                if 'grading_metadata' in row and row['grading_metadata']:
                    metadata_exists = True
                    break
            
            if metadata_exists:
                # Extract skill data
                skill_scores = {}
                
                for _, row in student_submissions.iterrows():
                    if 'grading_metadata' not in row or not row['grading_metadata']:
                        continue
                        
                    metadata = row['grading_metadata']
                    
                    # Code analysis skills
                    if 'code_analysis' in metadata:
                        for skill, score in metadata['code_analysis'].items():
                            if isinstance(score, (int, float)):
                                if skill not in skill_scores:
                                    skill_scores[skill] = []
                                skill_scores[skill].append(score)
                    
                    # Question results
                    if 'question_results' in metadata:
                        for q_res in metadata['question_results']:
                            if 'category' in q_res and 'score' in q_res:
                                skill = q_res['category']
                                score = q_res['score']
                                if skill not in skill_scores:
                                    skill_scores[skill] = []
                                skill_scores[skill].append(score)
                
                # Identify strengths (skills with high scores)
                for skill, scores in skill_scores.items():
                    avg_score = sum(scores) / len(scores)
                    if avg_score > 0.8:  # Threshold for strength
                        strengths.append({
                            "skill": skill,
                            "average_score": round(avg_score, 2),
                            "sample_size": len(scores)
                        })
            
            # Check assignment type performance
            if 'assignment_type' in student_submissions.columns and 'score' in student_submissions.columns:
                type_avg = student_submissions.groupby('assignment_type')['score'].mean()
                
                for assignment_type, avg_score in type_avg.items():
                    if avg_score > 85:  # Threshold for strength
                        strengths.append({
                            "assignment_type": assignment_type,
                            "average_score": round(avg_score, 2)
                        })
            
            # Sort by score (highest first)
            strengths.sort(key=lambda x: x.get('average_score', 0), reverse=True)
            
        except Exception as e:
            log_system_event(f"Error identifying student strengths: {str(e)}")
        
        return strengths
    
    def _identify_student_improvements(self, student_submissions: pd.DataFrame) -> List[Dict]:
        """Identify areas where the student needs improvement"""
        improvements = []
        
        try:
            # Check if grading metadata exists
            metadata_exists = False
            for _, row in student_submissions.iterrows():
                if 'grading_metadata' in row and row['grading_metadata']:
                    metadata_exists = True
                    break
            
            if metadata_exists:
                # Extract skill data
                skill_scores = {}
                
                for _, row in student_submissions.iterrows():
                    if 'grading_metadata' not in row or not row['grading_metadata']:
                        continue
                        
                    metadata = row['grading_metadata']
                    
                    # Code analysis skills
                    if 'code_analysis' in metadata:
                        for skill, score in metadata['code_analysis'].items():
                            if isinstance(score, (int, float)):
                                if skill not in skill_scores:
                                    skill_scores[skill] = []
                                skill_scores[skill].append(score)
                    
                    # Question results
                    if 'question_results' in metadata:
                        for q_res in metadata['question_results']:
                            if 'category' in q_res and 'score' in q_res:
                                skill = q_res['category']
                                score = q_res['score']
                                if skill not in skill_scores:
                                    skill_scores[skill] = []
                                skill_scores[skill].append(score)
                
                # Identify areas for improvement (skills with low scores)
                for skill, scores in skill_scores.items():
                    avg_score = sum(scores) / len(scores)
                    if avg_score < 0.6:  # Threshold for improvement needed
                        improvements.append({
                            "skill": skill,
                            "average_score": round(avg_score, 2),
                            "sample_size": len(scores)
                        })
            
            # Check assignment type performance
            if 'assignment_type' in student_submissions.columns and 'score' in student_submissions.columns:
                type_avg = student_submissions.groupby('assignment_type')['score'].mean()
                
                for assignment_type, avg_score in type_avg.items():
                    if avg_score < 70:  # Threshold for improvement needed
                        improvements.append({
                            "assignment_type": assignment_type,
                            "average_score": round(avg_score, 2)
                        })
            
            # Sort by score (lowest first)
            improvements.sort(key=lambda x: x.get('average_score', 100))
            
        except Exception as e:
            log_system_event(f"Error identifying student improvements: {str(e)}")
        
        return improvements
    
    def _generate_recommended_actions(self, struggling_students: List[Dict], 
                                     skill_gaps: Dict, 
                                     assignment_analysis: Dict) -> List[Dict]:
        """Generate recommended actions based on analysis"""
        actions = []
        
        # Actions for struggling students
        if struggling_students:
            actions.append({
                "type": "student_support",
                "priority": "high",
                "description": f"Schedule support sessions for {len(struggling_students)} struggling students",
                "details": f"Students with scores below 70%: " + 
                          ", ".join([s.get('name', f"ID: {s['student_id']}") for s in struggling_students[:3]]) +
                          (f" and {len(struggling_students) - 3} more" if len(struggling_students) > 3 else "")
            })
        
        # Actions for skill gaps
        gap_skills = [skill for skill, data in skill_gaps.items() if data.get('is_gap', False)]
        if gap_skills:
            actions.append({
                "type": "skill_reinforcement",
                "priority": "medium",
                "description": f"Address skill gaps in: {', '.join(gap_skills[:3])}" +
                              (f" and {len(gap_skills) - 3} more" if len(gap_skills) > 3 else ""),
                "details": "Consider providing additional resources or targeted exercises to strengthen these areas"
            })
        
        # Actions for difficult assignments
        difficult_assignments = [a_id for a_id, data in assignment_analysis.items() 
                                if data.get('difficulty_level') in ['Very Difficult', 'Difficult']]
        if difficult_assignments:
            actions.append({
                "type": "assignment_review",
                "priority": "medium",
                "description": f"Review {len(difficult_assignments)} assignments that students found challenging",
                "details": "Consider providing additional examples or clarifying instructions for these assignments"
            })
        
        return actions
    
    def _generate_student_recommendations(self, student_submissions: pd.DataFrame) -> List[str]:
        """Generate personalized recommendations for a student"""
        recommendations = []
        
        try:
            # Check for missing assignments
            if 'status' in student_submissions.columns:
                missing_count = len(student_submissions[student_submissions['status'] == 'missing'])
                if missing_count > 0:
                    recommendations.append(f"Complete {missing_count} missing assignments")
            
            # Check for low scores
            if 'score' in student_submissions.columns and not student_submissions.empty:
                low_scores = student_submissions[student_submissions['score'] < 60]
                if len(low_scores) > 0:
                    recommendations.append(f"Review feedback for {len(low_scores)} low-scoring assignments")
            
            # Check for late submissions
            if 'days_before_deadline' in student_submissions.columns:
                late_count = len(student_submissions[student_submissions['days_before_deadline'] < 0])
                if late_count > 0:
                    recommendations.append(f"Work on time management ({late_count} late submissions)")
            
            # Add generic recommendations if specific ones couldn't be generated
            if not recommendations:
                recommendations = [
                    "Schedule a meeting with the teacher for personalized guidance",
                    "Review course materials regularly",
                    "Participate actively in class discussions"
                ]
                
        except Exception as e:
            log_system_event(f"Error generating student recommendations: {str(e)}")
            recommendations = ["Schedule a meeting with the teacher to discuss your progress"]
        
        return recommendations 