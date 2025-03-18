import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import networkx as nx
import matplotlib.pyplot as plt
from ..utils.logger import log_system_event

class LearningPathService:
    """Service for generating interactive learning paths for students"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.learning_paths_dir = os.path.join(data_dir, 'learning_paths')
        if not os.path.exists(self.learning_paths_dir):
            os.makedirs(self.learning_paths_dir)
    
    def generate_learning_path(self, student_id: str, submissions: List[Dict], 
                             courses: List[Dict], resources: List[Dict] = None) -> Dict:
        """
        Generate a personalized learning path for a student
        
        Args:
            student_id: The ID of the student
            submissions: List of student's submissions
            courses: List of courses student is enrolled in
            resources: Optional list of learning resources
            
        Returns:
            Dict containing learning path data
        """
        try:
            if not submissions:
                return {
                    "status": "error",
                    "message": "Insufficient data to generate learning path"
                }
            
            # Identify skills from submissions
            skills_data = self._extract_skill_data(submissions)
            
            # Create skill nodes
            skill_nodes = self._create_skill_nodes(skills_data)
            
            # Create learning path graph
            path_graph = self._create_skill_graph(skill_nodes)
            
            # Generate skill progression map visualization
            graph_path = self._generate_skill_graph_visualization(path_graph, student_id)
            
            # Identify recommended resources
            recommendations = self._generate_resource_recommendations(
                skill_nodes, resources, courses)
            
            # Create milestone checklist
            milestones = self._generate_milestones(skill_nodes, submissions)
            
            return {
                "status": "success",
                "student_id": student_id,
                "skills": skill_nodes,
                "recommendations": recommendations,
                "milestones": milestones,
                "visualizations": {
                    "skill_graph": graph_path
                },
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_system_event(f"Error generating learning path: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating learning path: {str(e)}"
            }
    
    def _extract_skill_data(self, submissions: List[Dict]) -> Dict:
        """Extract skill data from submissions"""
        skill_data = {}
        
        for submission in submissions:
            # Extract skills from grading metadata
            if 'grading_metadata' in submission and submission['grading_metadata']:
                metadata = submission['grading_metadata']
                
                # Code analysis skills
                if 'code_analysis' in metadata:
                    for skill, score in metadata['code_analysis'].items():
                        if isinstance(score, (int, float)):
                            if skill not in skill_data:
                                skill_data[skill] = {
                                    'scores': [],
                                    'submissions': [],
                                    'category': 'programming'
                                }
                            skill_data[skill]['scores'].append(score)
                            skill_data[skill]['submissions'].append(submission['id'])
                
                # Question results from quizzes/tests
                if 'question_results' in metadata:
                    for q_res in metadata['question_results']:
                        if 'category' in q_res and 'score' in q_res:
                            skill = q_res['category']
                            score = q_res['score']
                            if skill not in skill_data:
                                skill_data[skill] = {
                                    'scores': [],
                                    'submissions': [],
                                    'category': 'academic'
                                }
                            skill_data[skill]['scores'].append(score)
                            skill_data[skill]['submissions'].append(submission['id'])
            
            # Extract keywords from assignments
            if 'assignment_id' in submission and 'tags' in submission:
                tags = submission.get('tags', [])
                for tag in tags:
                    if tag not in skill_data:
                        skill_data[tag] = {
                            'scores': [],
                            'submissions': [],
                            'category': 'topic'
                        }
                    
                    # If the submission has a score, add it
                    if 'score' in submission and submission['score'] is not None:
                        normalized_score = submission['score'] / 100  # Normalize to 0-1
                        skill_data[tag]['scores'].append(normalized_score)
                    
                    skill_data[tag]['submissions'].append(submission['id'])
        
        return skill_data
    
    def _create_skill_nodes(self, skill_data: Dict) -> List[Dict]:
        """Create structured nodes for skills"""
        skill_nodes = []
        
        for skill_name, data in skill_data.items():
            # Calculate average score
            if data['scores']:
                avg_score = sum(data['scores']) / len(data['scores'])
            else:
                avg_score = 0
            
            # Determine proficiency level
            if avg_score >= 0.85:
                proficiency = "mastered"
            elif avg_score >= 0.7:
                proficiency = "proficient"
            elif avg_score >= 0.5:
                proficiency = "developing"
            else:
                proficiency = "beginner"
            
            # Create node
            node = {
                "name": skill_name,
                "category": data['category'],
                "proficiency": proficiency,
                "proficiency_score": round(avg_score, 2),
                "submission_count": len(set(data['submissions'])),
                "needs_improvement": avg_score < 0.7,
                "is_strength": avg_score >= 0.85
            }
            
            skill_nodes.append(node)
        
        # Sort nodes by proficiency score (highest first)
        skill_nodes.sort(key=lambda x: x['proficiency_score'], reverse=True)
        
        return skill_nodes
    
    def _create_skill_graph(self, skill_nodes: List[Dict]) -> nx.Graph:
        """Create a graph of related skills"""
        G = nx.Graph()
        
        # Add nodes
        for i, node in enumerate(skill_nodes):
            G.add_node(node['name'], 
                      proficiency=node['proficiency'], 
                      category=node['category'],
                      score=node['proficiency_score'])
        
        # Add edges between related skills (based on category)
        categories = {}
        for node in skill_nodes:
            category = node['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(node['name'])
        
        # Connect skills within the same category
        for category, skills in categories.items():
            for i in range(len(skills)):
                for j in range(i + 1, len(skills)):
                    # Add edge with weight based on skill similarity
                    G.add_edge(skills[i], skills[j], weight=0.5)
        
        # Connect skills with similar proficiency levels
        for i in range(len(skill_nodes)):
            for j in range(i + 1, len(skill_nodes)):
                node1 = skill_nodes[i]
                node2 = skill_nodes[j]
                
                # If proficiency levels are similar, add an edge
                prof_diff = abs(node1['proficiency_score'] - node2['proficiency_score'])
                if prof_diff < 0.15:  # Threshold for similarity
                    if not G.has_edge(node1['name'], node2['name']):
                        G.add_edge(node1['name'], node2['name'], weight=0.3)
        
        return G
    
    def _generate_skill_graph_visualization(self, graph: nx.Graph, student_id: str) -> str:
        """Generate a visualization of the skill graph"""
        try:
            if not graph.nodes():
                return ""
            
            plt.figure(figsize=(12, 10))
            
            # Define node colors based on proficiency
            node_colors = []
            for node in graph.nodes():
                proficiency = graph.nodes[node]['proficiency']
                if proficiency == 'mastered':
                    node_colors.append('#2ecc71')  # Green
                elif proficiency == 'proficient':
                    node_colors.append('#3498db')  # Blue
                elif proficiency == 'developing':
                    node_colors.append('#f39c12')  # Orange
                else:
                    node_colors.append('#e74c3c')  # Red
            
            # Define node sizes based on proficiency score
            node_sizes = [graph.nodes[node]['score'] * 1000 + 500 for node in graph.nodes()]
            
            # Define edge weights based on connection strength
            edge_weights = [graph[u][v]['weight'] * 2 for u, v in graph.edges()]
            
            # Use spring layout for positioning
            pos = nx.spring_layout(graph, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                                 node_size=node_sizes, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.5)
            
            # Draw labels
            nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', 
                          markersize=15, label='Mastered'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', 
                          markersize=15, label='Proficient'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', 
                          markersize=15, label='Developing'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', 
                          markersize=15, label='Beginner')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            # Set title and remove axes
            plt.title('Skill Proficiency Map')
            plt.axis('off')
            
            # Save figure
            output_path = os.path.join(self.learning_paths_dir, f'skill_graph_{student_id}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            log_system_event(f"Error generating skill graph visualization: {str(e)}")
            return ""
    
    def _generate_resource_recommendations(self, skill_nodes: List[Dict], 
                                         resources: List[Dict], 
                                         courses: List[Dict]) -> List[Dict]:
        """Generate personalized resource recommendations"""
        recommendations = []
        
        # Identify skills that need improvement
        improvement_skills = [node for node in skill_nodes if node['needs_improvement']]
        
        # If custom resources are provided, use them
        if resources:
            for skill_node in improvement_skills:
                skill_name = skill_node['name']
                # Find relevant resources
                matching_resources = [
                    r for r in resources 
                    if skill_name.lower() in r.get('tags', []) or 
                       skill_name.lower() in r.get('title', '').lower()
                ]
                
                if matching_resources:
                    # Limit to top 3 resources
                    for resource in matching_resources[:3]:
                        recommendations.append({
                            "type": "resource",
                            "title": resource.get('title', 'Learning Resource'),
                            "url": resource.get('url', ''),
                            "source": resource.get('source', 'Custom'),
                            "format": resource.get('format', 'Other'),
                            "for_skill": skill_name,
                            "proficiency": skill_node['proficiency']
                        })
        
        # Generate generic recommendations based on skill categories
        for skill_node in improvement_skills:
            if any(r['for_skill'] == skill_node['name'] for r in recommendations):
                continue  # Skip if we already have recommendations for this skill
                
            category = skill_node['category']
            skill_name = skill_node['name']
            
            if category == 'programming':
                recommendations.append({
                    "type": "resource",
                    "title": f"Practice exercises for {skill_name}",
                    "description": f"Complete coding exercises focusing on {skill_name} concepts",
                    "for_skill": skill_name,
                    "proficiency": skill_node['proficiency']
                })
                
                # Add tutorial recommendation
                recommendations.append({
                    "type": "resource",
                    "title": f"{skill_name} tutorials on Codecademy",
                    "url": f"https://www.codecademy.com/search?query={skill_name.replace(' ', '%20')}",
                    "source": "Codecademy",
                    "format": "Tutorial",
                    "for_skill": skill_name,
                    "proficiency": skill_node['proficiency']
                })
                
            elif category == 'academic':
                # Find relevant course materials
                course_materials = []
                for course in courses:
                    if 'materials' in course:
                        for material in course['materials']:
                            if skill_name.lower() in material.get('tags', []) or \
                               skill_name.lower() in material.get('title', '').lower():
                                course_materials.append({
                                    "title": material.get('title', 'Course Material'),
                                    "url": material.get('url', ''),
                                    "course_name": course.get('name', 'Course')
                                })
                
                if course_materials:
                    for material in course_materials[:2]:
                        recommendations.append({
                            "type": "course_material",
                            "title": material['title'],
                            "url": material['url'],
                            "course": material['course_name'],
                            "for_skill": skill_name,
                            "proficiency": skill_node['proficiency']
                        })
                else:
                    # Generic recommendation
                    recommendations.append({
                        "type": "practice",
                        "title": f"Review {skill_name} concepts",
                        "description": "Create flashcards or summary notes on this topic",
                        "for_skill": skill_name,
                        "proficiency": skill_node['proficiency']
                    })
            
            # Add Khan Academy as a generic resource for most skills
            recommendations.append({
                "type": "resource",
                "title": f"{skill_name} on Khan Academy",
                "url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={skill_name.replace(' ', '+')}",
                "source": "Khan Academy",
                "format": "Video lessons",
                "for_skill": skill_name,
                "proficiency": skill_node['proficiency']
            })
        
        return recommendations
    
    def _generate_milestones(self, skill_nodes: List[Dict], 
                           submissions: List[Dict]) -> List[Dict]:
        """Generate milestone checklist for skill progression"""
        milestones = []
        
        # Create short-term milestones for skills needing improvement
        improvement_skills = [node for node in skill_nodes if node['needs_improvement']]
        for skill in improvement_skills[:3]:  # Focus on top 3 skills needing improvement
            milestones.append({
                "title": f"Improve {skill['name']} proficiency",
                "description": f"Complete practice exercises focusing on {skill['name']}",
                "category": "short_term",
                "current_level": skill['proficiency'],
                "target_level": "proficient" if skill['proficiency'] == "developing" else "developing",
                "progress": skill['proficiency_score'] * 100,
                "estimated_completion": "2 weeks"
            })
        
        # Create medium-term milestones for overall improvement
        if len(skill_nodes) > 0:
            avg_proficiency = sum(node['proficiency_score'] for node in skill_nodes) / len(skill_nodes)
            
            milestones.append({
                "title": "Raise overall proficiency",
                "description": "Improve average proficiency across all skills",
                "category": "medium_term",
                "current_level": f"{avg_proficiency:.2f}",
                "target_level": f"{min(avg_proficiency + 0.15, 1.0):.2f}",
                "progress": avg_proficiency * 100,
                "estimated_completion": "1 month"
            })
        
        # Create long-term milestone for mastery
        mastered_count = len([node for node in skill_nodes if node['proficiency'] == 'mastered'])
        if skill_nodes:
            mastery_percentage = (mastered_count / len(skill_nodes)) * 100
            
            milestones.append({
                "title": "Achieve mastery in key skills",
                "description": "Reach mastery level in most important skills",
                "category": "long_term",
                "current_level": f"{mastery_percentage:.1f}% mastered",
                "target_level": "75% mastered",
                "progress": mastery_percentage,
                "estimated_completion": "3 months"
            })
        
        # Create assignment completion milestone if there are pending assignments
        pending_assignments = [s for s in submissions if s.get('status') == 'pending' or s.get('status') == 'late']
        if pending_assignments:
            milestones.append({
                "title": "Complete pending assignments",
                "description": f"Submit {len(pending_assignments)} pending assignments",
                "category": "immediate",
                "current_level": "Incomplete",
                "target_level": "Complete",
                "progress": 0,
                "estimated_completion": "1 week"
            })
        
        return milestones 