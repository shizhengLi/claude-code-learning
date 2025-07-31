"""
Token management for context compression.

This module provides token counting and management utilities
for efficient context handling.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from ..utils.helpers import ContextManagerError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TokenInfo:
    """Information about tokens."""
    count: int
    limit: int
    usage_ratio: float
    estimated_cost: float


class TokenManager:
    """Manages token counting and limits."""
    
    def __init__(self, token_limit: int = 4000):
        """
        Initialize token manager.
        
        Args:
            token_limit: Maximum number of tokens allowed
        """
        self.token_limit = token_limit
        self.current_tokens = 0
        self.token_history = []
        
        # Simple token estimation (can be replaced with proper tokenizer)
        self.word_to_token_ratio = 1.3  # Average words per token
        self.char_to_token_ratio = 4.0  # Average characters per token
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        # Simple approximation - in production, use proper tokenizer
        # This is a rough estimate based on character count
        return max(1, int(len(text) / self.char_to_token_ratio))
    
    def count_tokens_dict(self, data: Dict[str, Any]) -> int:
        """
        Count tokens in dictionary data.
        
        Args:
            data: Dictionary to count tokens for
            
        Returns:
            Number of tokens
        """
        total_tokens = 0
        
        for key, value in data.items():
            # Count key tokens
            total_tokens += self.count_tokens(key)
            
            # Count value tokens based on type
            if isinstance(value, str):
                total_tokens += self.count_tokens(value)
            elif isinstance(value, dict):
                total_tokens += self.count_tokens_dict(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (str, dict)):
                        total_tokens += self.count_tokens_dict(item) if isinstance(item, dict) else self.count_tokens(item)
            elif value is not None:
                total_tokens += self.count_tokens(str(value))
        
        return total_tokens
    
    def check_token_limit(self, text: str) -> TokenInfo:
        """
        Check if text is within token limit.
        
        Args:
            text: Text to check
            
        Returns:
            TokenInfo object
        """
        token_count = self.count_tokens(text)
        usage_ratio = token_count / self.token_limit if self.token_limit > 0 else 0
        
        return TokenInfo(
            count=token_count,
            limit=self.token_limit,
            usage_ratio=usage_ratio,
            estimated_cost=self._estimate_cost(token_count)
        )
    
    def truncate_to_token_limit(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens (uses default if None)
            
        Returns:
            Truncated text
        """
        limit = max_tokens or self.token_limit
        if limit <= 0:
            return text
        
        current_count = self.count_tokens(text)
        if current_count <= limit:
            return text
        
        # Simple truncation by character count
        target_chars = int(limit * self.char_to_token_ratio)
        return text[:target_chars]
    
    def get_optimization_suggestions(self, data: Dict[str, Any]) -> List[str]:
        """
        Get suggestions for token optimization.
        
        Args:
            data: Data to analyze
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        token_count = self.count_tokens_dict(data)
        
        if token_count > self.token_limit * 0.8:
            suggestions.append("Consider compressing large text fields")
        
        # Check for redundant data
        if len(data) > 20:
            suggestions.append("Consider grouping related fields")
        
        # Check for long strings
        long_strings = [k for k, v in data.items() if isinstance(v, str) and len(v) > 1000]
        if long_strings:
            suggestions.append(f"Consider summarizing long text fields: {', '.join(long_strings[:3])}")
        
        return suggestions
    
    def _estimate_cost(self, token_count: int) -> float:
        """
        Estimate cost based on token count.
        
        Args:
            token_count: Number of tokens
            
        Returns:
            Estimated cost in USD
        """
        # Simple cost estimation (adjust based on actual pricing)
        return token_count * 0.00002  # $0.02 per 1K tokens
    
    def reset(self) -> None:
        """Reset token manager state."""
        self.current_tokens = 0
        self.token_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token manager statistics."""
        return {
            'token_limit': self.token_limit,
            'current_tokens': self.current_tokens,
            'usage_ratio': self.current_tokens / self.token_limit if self.token_limit > 0 else 0,
            'history_length': len(self.token_history)
        }