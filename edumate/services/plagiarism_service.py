"""Advanced plagiarism detection service."""
import logging
import re
import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import requests
from pathlib import Path
import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlagiarismService:
    """Service for detecting plagiarism in student submissions."""
    
    def __init__(self):
        """Initialize the plagiarism detection service."""
        # Set up cache directory for previously analyzed content
        self.cache_dir = Path("data/cache/plagiarism")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load external API keys if available
        self.api_key = os.environ.get("PLAGIARISM_API_KEY")
        
        # Initialize vectorizer for text comparison
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            lowercase=True
        )
        
        # Load reference database if available
        self.reference_db = self._load_reference_db()
    
    def check_plagiarism(self, content: str, compare_against: Optional[List[str]] = None) -> Dict[str, Any]:
        """Check content for plagiarism against various sources.
        
        Args:
            content: The content to check for plagiarism
            compare_against: Additional content to compare against (optional)
            
        Returns:
            Dictionary with plagiarism score and analysis
        """
        if not content:
            return {"score": 0, "matches": [], "error": "Empty content"}
            
        # Preprocess content
        processed_content = self._preprocess_text(content)
        content_hash = self._get_content_hash(processed_content)
        
        # Check cache first
        cached_result = self._check_cache(content_hash)
        if cached_result:
            logger.info("Using cached plagiarism result")
            return cached_result
        
        # Local corpus comparison
        local_similarity = self._check_against_local_corpus(processed_content, compare_against)
        
        # Reference database comparison
        reference_similarity = self._check_against_reference_db(processed_content)
        
        # External API check (if available)
        api_similarity = self._check_external_api(processed_content) if self.api_key else {"score": 0, "sources": []}
        
        # Combine all results
        result = self._combine_results(local_similarity, reference_similarity, api_similarity)
        
        # Cache result
        self._cache_result(content_hash, result)
        
        return result
    
    def add_to_reference_db(self, content: str, metadata: Dict[str, str]) -> bool:
        """Add content to the reference database.
        
        Args:
            content: The content to add
            metadata: Information about the content (author, title, etc.)
            
        Returns:
            True if successfully added
        """
        if not content:
            return False
            
        # Preprocess content
        processed_content = self._preprocess_text(content)
        content_hash = self._get_content_hash(processed_content)
        
        # Add to reference database
        if not hasattr(self, 'reference_db'):
            self.reference_db = {'entries': []}
            
        # Check if already exists
        if any(entry.get('hash') == content_hash for entry in self.reference_db.get('entries', [])):
            logger.info(f"Content with hash {content_hash} already exists in reference database")
            return False
        
        # Add new entry
        entry = {
            'hash': content_hash,
            'content': processed_content,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        self.reference_db.setdefault('entries', []).append(entry)
        
        # Save database
        self._save_reference_db()
        
        return True
    
    def _preprocess_text(self, text: str) -> str:
        """Prepare text for plagiarism detection."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def _get_content_hash(self, content: str) -> str:
        """Generate a hash for the content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _check_cache(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Check if plagiarism result exists in cache."""
        cache_file = self.cache_dir / f"{content_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache: {str(e)}")
        return None
    
    def _cache_result(self, content_hash: str, result: Dict[str, Any]) -> None:
        """Cache plagiarism result."""
        cache_file = self.cache_dir / f"{content_hash}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")
    
    def _check_against_local_corpus(self, content: str, 
                                   compare_against: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare content against provided corpus.
        
        Args:
            content: The content to check
            compare_against: List of other content to compare against
            
        Returns:
            Dictionary with similarity score and matches
        """
        if not compare_against:
            return {"score": 0, "matches": []}
            
        # Preprocess comparison corpus
        processed_corpus = [self._preprocess_text(text) for text in compare_against]
        
        # Calculate similarity using TF-IDF and cosine similarity
        try:
            # Add the content to check at the beginning
            all_texts = [content] + processed_corpus
            
            # If only one document, similarity is 0
            if len(all_texts) < 2:
                return {"score": 0, "matches": []}
                
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between first document and all others
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Find matches above threshold
            matches = []
            max_score = 0
            for i, score in enumerate(similarity_scores):
                if score > 0.3:  # Similarity threshold
                    max_score = max(max_score, score)
                    matches.append({
                        "document_index": i,
                        "similarity": float(score),
                        "excerpt": self._extract_similar_excerpt(content, processed_corpus[i])
                    })
            
            return {
                "score": float(max_score),
                "matches": sorted(matches, key=lambda x: x["similarity"], reverse=True)
            }
            
        except Exception as e:
            logger.error(f"Error during local corpus comparison: {str(e)}")
            return {"score": 0, "matches": [], "error": str(e)}
    
    def _check_against_reference_db(self, content: str) -> Dict[str, Any]:
        """Compare content against reference database."""
        if not hasattr(self, 'reference_db') or not self.reference_db.get('entries'):
            return {"score": 0, "matches": []}
            
        try:
            # Extract content from reference database
            reference_texts = [entry.get('content', '') for entry in self.reference_db.get('entries', [])]
            
            # If database is empty, return no matches
            if not reference_texts:
                return {"score": 0, "matches": []}
                
            # Create TF-IDF matrix
            all_texts = [content] + reference_texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Find matches above threshold
            matches = []
            max_score = 0
            for i, score in enumerate(similarity_scores):
                if score > 0.3:  # Similarity threshold
                    max_score = max(max_score, score)
                    reference_entry = self.reference_db.get('entries', [])[i]
                    matches.append({
                        "reference_id": i,
                        "similarity": float(score),
                        "metadata": reference_entry.get('metadata', {}),
                        "excerpt": self._extract_similar_excerpt(content, reference_entry.get('content', ''))
                    })
            
            return {
                "score": float(max_score),
                "matches": sorted(matches, key=lambda x: x["similarity"], reverse=True)
            }
            
        except Exception as e:
            logger.error(f"Error during reference database comparison: {str(e)}")
            return {"score": 0, "matches": [], "error": str(e)}
    
    def _check_external_api(self, content: str) -> Dict[str, Any]:
        """Check plagiarism using external API (if available)."""
        if not self.api_key:
            return {"score": 0, "sources": []}
            
        try:
            # This is a placeholder for external API integration
            # In a real implementation, this would call an external service
            # For now, return mock result
            return {
                "score": 0.1,
                "sources": [{
                    "url": "https://example.com/sample",
                    "similarity": 0.1,
                    "title": "Sample Source"
                }]
            }
            
        except Exception as e:
            logger.error(f"Error during external API check: {str(e)}")
            return {"score": 0, "sources": [], "error": str(e)}
    
    def _combine_results(self, local_result: Dict[str, Any], 
                        reference_result: Dict[str, Any],
                        api_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from different plagiarism checks."""
        # Get highest similarity score from any source
        max_score = max(
            local_result.get('score', 0),
            reference_result.get('score', 0),
            api_result.get('score', 0)
        )
        
        # Combine matches
        all_matches = []
        
        # Add local matches
        for match in local_result.get('matches', []):
            all_matches.append({
                "source": "local_corpus",
                "similarity": match.get('similarity', 0),
                "excerpt": match.get('excerpt', ''),
                "document_index": match.get('document_index')
            })
            
        # Add reference matches
        for match in reference_result.get('matches', []):
            all_matches.append({
                "source": "reference_database",
                "similarity": match.get('similarity', 0),
                "excerpt": match.get('excerpt', ''),
                "metadata": match.get('metadata', {})
            })
            
        # Add API matches
        for source in api_result.get('sources', []):
            all_matches.append({
                "source": "external_api",
                "similarity": source.get('similarity', 0),
                "url": source.get('url', ''),
                "title": source.get('title', '')
            })
            
        # Sort matches by similarity
        all_matches.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Generate human-readable summary
        summary = self._generate_summary(max_score, all_matches)
        
        return {
            "score": max_score,
            "matches": all_matches[:10],  # Limit to top 10 matches
            "match_count": len(all_matches),
            "summary": summary,
            "severity": self._get_severity_level(max_score)
        }
    
    def _extract_similar_excerpt(self, text1: str, text2: str) -> str:
        """Extract similar excerpt between two texts."""
        # Split into sentences
        sentences1 = re.split(r'(?<=[.!?])\s+', text1)
        sentences2 = re.split(r'(?<=[.!?])\s+', text2)
        
        # Compare each sentence
        best_match = ""
        best_score = 0
        
        for s1 in sentences1:
            if len(s1) < 20:  # Skip very short sentences
                continue
                
            for s2 in sentences2:
                if len(s2) < 20:
                    continue
                    
                # Calculate similarity
                similarity = self._sentence_similarity(s1, s2)
                
                if similarity > 0.7 and similarity > best_score:
                    best_score = similarity
                    best_match = s2
        
        return best_match
    
    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two sentences."""
        # Simple approach: count common words
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'to', 'for', 'by', 'with', 'about'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def _generate_summary(self, score: float, matches: List[Dict[str, Any]]) -> str:
        """Generate human-readable summary of plagiarism check."""
        if score < 0.1:
            return "No significant similarity detected."
        elif score < 0.3:
            return f"Low similarity detected with {len(matches)} sources."
        elif score < 0.6:
            return f"Moderate similarity detected with {len(matches)} sources. Review recommended."
        else:
            return f"High similarity detected with {len(matches)} sources. Detailed review required."
    
    def _get_severity_level(self, score: float) -> str:
        """Get severity level based on similarity score."""
        if score < 0.1:
            return "none"
        elif score < 0.3:
            return "low"
        elif score < 0.6:
            return "moderate"
        else:
            return "high"
    
    def _load_reference_db(self) -> Dict[str, Any]:
        """Load reference database from disk."""
        db_file = Path("data/reference_db.json")
        if db_file.exists():
            try:
                with open(db_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading reference database: {str(e)}")
        
        return {"entries": []}
    
    def _save_reference_db(self) -> None:
        """Save reference database to disk."""
        db_file = Path("data/reference_db.json")
        try:
            with open(db_file, 'w') as f:
                json.dump(self.reference_db, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving reference database: {str(e)}")

# Utility functions
def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity between two texts."""
    # Preprocess
    text1 = re.sub(r'\s+', ' ', text1.lower()).strip()
    text2 = re.sub(r'\s+', ' ', text2.lower()).strip()
    
    # Calculate Jaccard similarity
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0 