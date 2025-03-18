"""Advanced plagiarism detection service for student submissions."""
import logging
import re
import os
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
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
    
    def __init__(self, reference_db_path: str = None, cache_dir: str = None, api_key: str = None):
        """Initialize the plagiarism detection service.
        
        Args:
            reference_db_path: Path to reference database for comparison
            cache_dir: Directory to store cache
            api_key: External API key for additional plagiarism detection
        """
        # Set up paths
        self.reference_db_path = reference_db_path or "data/reference_db.json"
        self.cache_dir = Path(cache_dir or "data/cache/plagiarism")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.reference_db_path)).mkdir(parents=True, exist_ok=True)
        
        # Load reference database
        self.reference_db = self._load_reference_db()
        
        # Set up API key for external service
        self.api_key = api_key or os.getenv('PLAGIARISM_API_KEY')
        
        # Initialize vectorizer for text comparison
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        logger.info("Plagiarism service initialized")
    
    def check_plagiarism(self, content: str, comparison_content: Optional[List[str]] = None, 
                        threshold: float = 0.8, check_external: bool = True) -> Dict[str, Any]:
        """Check content for plagiarism against references and optionally external sources.
        
        Args:
            content: Text content to check for plagiarism
            comparison_content: Optional specific content to compare against
            threshold: Similarity threshold for plagiarism detection (0.0 to 1.0)
            check_external: Whether to check against external API
            
        Returns:
            Dictionary with plagiarism score and analysis
        """
        # Validate and clean input
        if not content or not content.strip():
            return {
                "score": 0.0,
                "matches": [],
                "original": True,
                "message": "Empty content provided"
            }
            
        # Preprocess content
        processed_content = self._preprocess_text(content)
        content_hash = hashlib.md5(processed_content.encode('utf-8')).hexdigest()
        
        # Check cache first
        cache_result = self._check_cache(content_hash)
        if cache_result:
            logger.info("Returning cached plagiarism result")
            return cache_result
        
        # Initialize results
        results = {
            "score": 0.0,
            "matches": [],
            "original": True,
            "message": "Content appears to be original"
        }
        
        # Perform three types of checks and combine results
        local_result = self._check_local_corpus(processed_content, threshold)
        reference_result = self._check_reference_db(processed_content, threshold)
        
        # Combine local and reference results
        if local_result["score"] > 0 or reference_result["score"] > 0:
            max_score = max(local_result["score"], reference_result["score"])
            combined_matches = local_result["matches"] + reference_result["matches"]
            
            # Sort matches by similarity score (descending)
            combined_matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            results["score"] = max_score
            results["matches"] = combined_matches
            results["original"] = max_score < threshold
            
            if max_score >= threshold:
                results["message"] = f"Potential plagiarism detected ({max_score:.2f} similarity)"
        
        # Check against specific comparison content if provided
        if comparison_content:
            comparison_result = self._compare_with_content(processed_content, comparison_content, threshold)
            
            if comparison_result["score"] > results["score"]:
                results["score"] = comparison_result["score"]
                results["matches"] = comparison_result["matches"] + results["matches"]
                results["original"] = comparison_result["score"] < threshold
                
                if comparison_result["score"] >= threshold:
                    results["message"] = f"Potential plagiarism detected in direct comparison ({comparison_result['score']:.2f} similarity)"
        
        # Check against external API if enabled
        if check_external and self.api_key:
            try:
                api_result = self._check_external_api(content)
                
                # If API check found higher plagiarism, update results
                if api_result["score"] > results["score"]:
                    results["score"] = api_result["score"]
                    results["matches"] = api_result["matches"] + results["matches"]
                    results["original"] = api_result["score"] < threshold
                    results["message"] = api_result["message"]
                    
            except Exception as e:
                logger.error(f"Error checking external API: {e}")
                # Continue with local results if API fails
        
        # Add summary information
        results["summary"] = self._generate_summary(results)
        
        # Cache results
        self._cache_result(content_hash, results)
        
        return results
    
    def add_to_reference_db(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add content to the reference database.
        
        Args:
            content: Content to add to reference database
            metadata: Information about the content (author, title, etc.)
            
        Returns:
            Boolean indicating success
        """
        if not content or not content.strip():
            return False
            
        # Process content
        processed_content = self._preprocess_text(content)
        content_hash = hashlib.md5(processed_content.encode('utf-8')).hexdigest()
        
        # Check if already exists
        if content_hash in self.reference_db:
            logger.info(f"Content already exists in reference DB with hash {content_hash}")
            return False
            
        # Add to database
        self.reference_db[content_hash] = {
            "content": processed_content,
            "metadata": metadata,
            "added_at": time.time()
        }
        
        # Save database
        self._save_reference_db()
        logger.info(f"Added content to reference DB with hash {content_hash}")
        
        return True
    
    def _check_local_corpus(self, content: str, threshold: float) -> Dict[str, Any]:
        """Compare content against local corpus using TF-IDF and cosine similarity.
        
        Args:
            content: Preprocessed content to check
            threshold: Similarity threshold
            
        Returns:
            Dictionary with score and matches
        """
        result = {
            "score": 0.0,
            "matches": []
        }
        
        # If reference database is empty, return empty result
        if not self.reference_db:
            return result
            
        try:
            # Create corpus with content to check and reference texts
            corpus = [content]
            reference_hashes = []
            
            for doc_hash, doc_data in self.reference_db.items():
                corpus.append(doc_data["content"])
                reference_hashes.append(doc_hash)
            
            # Skip if only the input content exists
            if len(corpus) <= 1:
                return result
                
            # Calculate TF-IDF and similarity
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Find matches above threshold
            max_similarity = 0.0
            for i, similarity in enumerate(cosine_similarities):
                if similarity >= threshold:
                    doc_hash = reference_hashes[i]
                    doc_data = self.reference_db[doc_hash]
                    
                    match = {
                        "similarity": float(similarity),
                        "source": doc_data["metadata"].get("title", "Unknown Source"),
                        "author": doc_data["metadata"].get("author", "Unknown Author"),
                        "type": "reference_db",
                        "source_id": doc_hash
                    }
                    
                    result["matches"].append(match)
                
                max_similarity = max(max_similarity, similarity)
            
            result["score"] = float(max_similarity)
            
        except Exception as e:
            logger.error(f"Error in local corpus check: {e}")
            
        return result
    
    def _check_reference_db(self, content: str, threshold: float) -> Dict[str, Any]:
        """Check content against reference database for direct matches.
        
        Args:
            content: Preprocessed content to check
            threshold: Similarity threshold
            
        Returns:
            Dictionary with score and matches
        """
        result = {
            "score": 0.0,
            "matches": []
        }
        
        # Extract sentences from content to check
        content_sentences = self._extract_sentences(content)
        
        # Process each reference document
        max_similarity = 0.0
        
        for doc_hash, doc_data in self.reference_db.items():
            ref_content = doc_data["content"]
            ref_sentences = self._extract_sentences(ref_content)
            
            # Skip empty documents
            if not ref_sentences:
                continue
                
            # Compare sentences for matches
            matches = self._find_matching_segments(content_sentences, ref_sentences)
            
            if matches:
                similarity = matches[0]["similarity"]  # Highest similarity match
                
                if similarity >= threshold:
                    match = {
                        "similarity": similarity,
                        "source": doc_data["metadata"].get("title", "Unknown Source"),
                        "author": doc_data["metadata"].get("author", "Unknown Author"),
                        "type": "exact_match",
                        "source_id": doc_hash,
                        "matched_text": matches[0]["text"]
                    }
                    
                    result["matches"].append(match)
                
                max_similarity = max(max_similarity, similarity)
        
        result["score"] = max_similarity
        return result
    
    def _check_external_api(self, content: str) -> Dict[str, Any]:
        """Check content for plagiarism using external API.
        
        Args:
            content: Original content to check
            
        Returns:
            Dictionary with score and matches from external API
        """
        result = {
            "score": 0.0,
            "matches": [],
            "message": "No external plagiarism detected"
        }
        
        if not self.api_key:
            return result
            
        try:
            # Sample API call - replace with actual API integration
            # This is a placeholder - implement with your chosen plagiarism API
            api_url = "https://api.plagiarismchecker.example/v1/check"
            
            payload = {
                "text": content[:10000],  # API may have text size limits
                "api_key": self.api_key
            }
            
            # Uncomment to use a real API
            # response = requests.post(api_url, json=payload)
            # api_result = response.json()
            
            # Placeholder API result for testing
            api_result = {
                "plagiarism_score": 0.0,
                "matches": []
            }
            
            # Process API response
            api_score = float(api_result.get("plagiarism_score", 0.0))
            
            # Transform API matches to our format
            for api_match in api_result.get("matches", []):
                match = {
                    "similarity": float(api_match.get("similarity", 0.0)),
                    "source": api_match.get("source_url", "External Source"),
                    "matched_text": api_match.get("matched_text", ""),
                    "type": "external_api"
                }
                
                result["matches"].append(match)
            
            result["score"] = api_score
            
            if api_score > 0.5:  # Use appropriate threshold
                result["message"] = f"External plagiarism check found {api_score:.2f} similarity score"
                
        except Exception as e:
            logger.error(f"Error with external API: {e}")
            
        return result
    
    def _compare_with_content(self, content: str, comparison_texts: List[str], 
                             threshold: float) -> Dict[str, Any]:
        """Compare content directly against provided comparison texts.
        
        Args:
            content: Preprocessed content to check
            comparison_texts: List of texts to compare against
            threshold: Similarity threshold
            
        Returns:
            Dictionary with score and matches
        """
        result = {
            "score": 0.0,
            "matches": []
        }
        
        if not comparison_texts:
            return result
            
        try:
            # Create corpus with content and comparison texts
            corpus = [content] + [self._preprocess_text(text) for text in comparison_texts]
            
            # Calculate TF-IDF and similarity
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Process similarities
            max_similarity = 0.0
            
            for i, similarity in enumerate(cosine_similarities):
                if similarity >= threshold:
                    match = {
                        "similarity": float(similarity),
                        "source": f"Provided comparison text #{i+1}",
                        "type": "direct_comparison"
                    }
                    
                    result["matches"].append(match)
                
                max_similarity = max(max_similarity, similarity)
            
            result["score"] = float(max_similarity)
            
        except Exception as e:
            logger.error(f"Error in direct comparison: {e}")
        
        return result
    
    def _find_matching_segments(self, content_sentences: List[str], 
                               ref_sentences: List[str]) -> List[Dict[str, Any]]:
        """Find matching text segments between content and reference.
        
        Args:
            content_sentences: List of sentences from content
            ref_sentences: List of sentences from reference
            
        Returns:
            List of matching segments with similarity scores
        """
        matches = []
        
        # Skip if either list is empty
        if not content_sentences or not ref_sentences:
            return matches
            
        # Compare each content sentence against reference sentences
        for i, content_sent in enumerate(content_sentences):
            content_sent = content_sent.strip()
            
            # Skip short sentences (likely common phrases)
            if len(content_sent.split()) < 7:
                continue
                
            # Compare against each reference sentence
            for j, ref_sent in enumerate(ref_sentences):
                ref_sent = ref_sent.strip()
                
                # Calculate similarity
                similarity = self._calculate_similarity(content_sent, ref_sent)
                
                if similarity > 0.8:  # Threshold for sentence matching
                    matches.append({
                        "text": content_sent,
                        "ref_text": ref_sent,
                        "content_idx": i,
                        "ref_idx": j,
                        "similarity": similarity
                    })
        
        # Sort matches by similarity (descending)
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        return matches
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text segments.
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # For short texts, check for exact match or containment
        if text1 == text2:
            return 1.0
            
        if text1 in text2 or text2 in text1:
            return 0.9
            
        # For longer texts, use cosine similarity
        try:
            # Create temporary corpus with both texts
            corpus = [text1, text2]
            
            # Calculate TF-IDF and similarity
            temp_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = temp_vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()[0]
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            
            # Fallback: simple word overlap ratio
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
                
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return overlap / union if union > 0 else 0.0
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for consistent comparison.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters and numbers (optional)
        # text = re.sub(r'[^\w\s]', '', text)
        # text = re.sub(r'\d+', '', text)
        
        return text
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text.
        
        Args:
            text: Text to extract sentences from
            
        Returns:
            List of sentences
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty or very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary of plagiarism results.
        
        Args:
            results: Plagiarism check results
            
        Returns:
            Human-readable summary
        """
        score = results["score"]
        matches = results["matches"]
        
        if score < 0.3:
            return "The content appears to be original with no significant matches found."
            
        elif score < 0.5:
            return f"The content contains some similarity ({score:.2f}) with existing sources, but is likely original work with proper citation."
            
        elif score < 0.7:
            match_count = len(matches)
            sources = ", ".join(set(m["source"] for m in matches[:3]))
            return f"Moderate similarity ({score:.2f}) detected with {match_count} potential matches from sources including {sources}. Review suggested."
            
        else:
            match_count = len(matches)
            sources = ", ".join(set(m["source"] for m in matches[:3]))
            return f"High similarity ({score:.2f}) detected with {match_count} significant matches from sources including {sources}. Detailed review required."
    
    def _load_reference_db(self) -> Dict[str, Any]:
        """Load reference database from file.
        
        Returns:
            Dictionary containing reference documents
        """
        try:
            if os.path.exists(self.reference_db_path):
                with open(self.reference_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.info(f"Reference database not found at {self.reference_db_path}, creating new database")
                return {}
        except Exception as e:
            logger.error(f"Error loading reference database: {e}")
            return {}
    
    def _save_reference_db(self) -> bool:
        """Save reference database to file.
        
        Returns:
            Boolean indicating success
        """
        try:
            with open(self.reference_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.reference_db, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving reference database: {e}")
            return False
    
    def _check_cache(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Check if result exists in cache.
        
        Args:
            content_hash: Hash of content
            
        Returns:
            Cached result or None
        """
        cache_file = self.cache_dir / f"{content_hash}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache: {str(e)}")
                
        return None
    
    def _cache_result(self, content_hash: str, result: Dict[str, Any]) -> None:
        """Cache a plagiarism check result.
        
        Args:
            content_hash: Hash of content
            result: Result to cache
        """
        cache_file = self.cache_dir / f"{content_hash}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")

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