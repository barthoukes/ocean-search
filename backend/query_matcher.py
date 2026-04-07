"""
Query matching and text highlighting utilities.
"""

import re
from typing import List, Dict, Any


class QueryMatcher:
    """Utility class to find and highlight matching text in documents."""
    
    # Common stop words to ignore
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for',
        'in', 'is', 'it', 'of', 'on', 'or', 'the', 'to', 'with', 'what',
        'when', 'where', 'who', 'why', 'how', 'the', 'this', 'that'
    }
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Split and clean words
        words = re.findall(r'\b[a-zA-Z0-9_]{3,}\b', query.lower())
        keywords = [w for w in words if w not in QueryMatcher.STOP_WORDS]
        
        # If no keywords found, use the original query words
        if not keywords:
            keywords = re.findall(r'\b[a-zA-Z0-9_]+\b', query.lower())
        
        return keywords
    
    @staticmethod
    def find_matching_snippets(text: str, query: str, context_chars: int = 100) -> List[Dict[str, Any]]:
        """
        Find snippets of text that match the query.
        
        Args:
            text: Document text to search
            query: Search query
            context_chars: Number of characters to show around matches
            
        Returns:
            List of dicts with 'snippet', 'score', and 'matched_terms'
        """
        text_lower = text.lower()
        query_lower = query.lower()
        keywords = QueryMatcher.extract_keywords(query)
        
        if not keywords:
            return []
        
        matches = []
        
        # Find exact phrase matches first (highest priority)
        if query_lower in text_lower:
            positions = [m.start() for m in re.finditer(re.escape(query_lower), text_lower)]
            for pos in positions:
                start = max(0, pos - context_chars)
                end = min(len(text), pos + len(query) + context_chars)
                snippet = text[start:end]
                matches.append({
                    'snippet': snippet,
                    'score': 1.0,
                    'matched_terms': [query],
                    'match_type': 'exact_phrase'
                })
        
        # Find keyword matches
        keyword_matches = []
        for keyword in keywords:
            if keyword in text_lower:
                positions = [m.start() for m in re.finditer(re.escape(keyword), text_lower)]
                for pos in positions:
                    start = max(0, pos - context_chars)
                    end = min(len(text), pos + len(keyword) + context_chars)
                    snippet = text[start:end]
                    keyword_matches.append({
                        'snippet': snippet,
                        'keyword': keyword,
                        'position': pos
                    })
        
        # Group by snippet and calculate score
        if keyword_matches:
            # Group matches by snippet window
            snippets_dict = {}
            for match in keyword_matches:
                snippet_key = match['snippet'][:50]  # Use first 50 chars as key
                if snippet_key not in snippets_dict:
                    snippets_dict[snippet_key] = {
                        'snippet': match['snippet'],
                        'matched_terms': set(),
                        'count': 0
                    }
                snippets_dict[snippet_key]['matched_terms'].add(match['keyword'])
                snippets_dict[snippet_key]['count'] += 1
            
            for snippet_data in snippets_dict.values():
                # Score based on number of matching terms
                score = min(0.95, 0.5 + (snippet_data['count'] / len(keywords)) * 0.45)
                matches.append({
                    'snippet': snippet_data['snippet'],
                    'score': score,
                    'matched_terms': list(snippet_data['matched_terms']),
                    'match_type': 'keywords'
                })
        
        # Sort by score and remove duplicates
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top 3 unique snippets
        seen = set()
        unique_matches = []
        for match in matches:
            snippet_key = match['snippet'][:50]
            if snippet_key not in seen:
                seen.add(snippet_key)
                unique_matches.append(match)
            if len(unique_matches) >= 3:
                break
        
        return unique_matches
    
    @staticmethod
    def highlight_text(text: str, query: str) -> str:
        """Highlight matching terms in text with color."""
        if not text or not query:
            return text
        
        text_lower = text.lower()
        query_lower = query.lower()
        keywords = QueryMatcher.extract_keywords(query)
        
        # Colors for highlighting
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        
        highlighted = text
        
        # Highlight exact phrase matches (in green)
        if query_lower in text_lower:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            highlighted = pattern.sub(f'{GREEN}\\g<0>{RESET}', highlighted)
        
        # Highlight individual keywords (in yellow, if not already highlighted)
        for keyword in keywords:
            if keyword.lower() in text_lower:
                # Don't highlight if it's part of a larger highlighted phrase
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                highlighted = pattern.sub(f'{YELLOW}\\g<0>{RESET}', highlighted)
        
        return highlighted

