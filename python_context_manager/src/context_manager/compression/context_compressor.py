"""
Context compression utilities.

This module provides utilities for compressing and optimizing
context data to reduce token usage.
"""

import json
import zlib
import gzip
import pickle
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict

from .token_manager import TokenManager
from ..utils.helpers import ContextManagerError, CompressionError
from ..utils.logging import get_logger


logger = get_logger(__name__)


class ContextCompressor:
    """Handles context compression and optimization."""
    
    def __init__(self, token_limit: int = 4000):
        """
        Initialize context compressor.
        
        Args:
            token_limit: Maximum number of tokens
        """
        self.token_manager = TokenManager(token_limit)
        self.compression_strategies = [
            self._compress_json,
            self._compress_text,
            self._compress_numbers,
            self._compress_booleans,
            self._compress_lists
        ]
    
    def compress_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress context data.
        
        Args:
            context: Context data to compress
            
        Returns:
            Compressed context
        """
        try:
            compressed = context.copy()
            
            # Apply compression strategies
            for strategy in self.compression_strategies:
                compressed = strategy(compressed)
            
            # Check token usage
            token_info = self.token_manager.check_token_limit(json.dumps(compressed))
            
            if token_info.usage_ratio > 0.9:
                logger.warning(f"Context is near token limit: {token_info.usage_ratio:.1%}")
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error compressing context: {e}")
            return context
    
    def decompress_context(self, compressed_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompress context data.
        
        Args:
            compressed_context: Compressed context data
            
        Returns:
            Decompressed context
        """
        try:
            decompressed = compressed_context.copy()
            
            # Apply decompression strategies
            decompressed = self._decompress_json(decompressed)
            decompressed = self._decompress_text(decompressed)
            
            return decompressed
            
        except Exception as e:
            logger.error(f"Error decompressing context: {e}")
            return compressed_context
    
    def optimize_for_tokens(self, context: Dict[str, Any], target_tokens: int) -> Dict[str, Any]:
        """
        Optimize context for specific token count.
        
        Args:
            context: Context to optimize
            target_tokens: Target token count
            
        Returns:
            Optimized context
        """
        try:
            current_tokens = self.token_manager.count_tokens_dict(context)
            
            if current_tokens <= target_tokens:
                return context
            
            # Optimization strategies
            optimized = context.copy()
            
            # Remove low-priority fields
            optimized = self._remove_low_priority_fields(optimized, target_tokens)
            
            # Summarize long text
            optimized = self._summarize_long_text(optimized, target_tokens)
            
            # Compress remaining data
            optimized = self.compress_context(optimized)
            
            return optimized
            
        except Exception as e:
            logger.error(f"Error optimizing context: {e}")
            return context
    
    def get_compression_stats(self, original: Dict[str, Any], compressed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Args:
            original: Original context
            compressed: Compressed context
            
        Returns:
            Compression statistics
        """
        original_tokens = self.token_manager.count_tokens_dict(original)
        compressed_tokens = self.token_manager.count_tokens_dict(compressed)
        
        original_size = len(json.dumps(original).encode('utf-8'))
        compressed_size = len(json.dumps(compressed).encode('utf-8'))
        
        return {
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'token_reduction': original_tokens - compressed_tokens,
            'token_reduction_ratio': (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'size_reduction': original_size - compressed_size,
            'size_reduction_ratio': (original_size - compressed_size) / original_size if original_size > 0 else 0
        }
    
    def _compress_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress JSON data."""
        try:
            compressed = {}
            
            for key, value in data.items():
                if isinstance(value, dict):
                    # Recursively compress nested dictionaries
                    compressed[key] = self._compress_json(value)
                elif isinstance(value, (list, tuple)):
                    # Compress lists
                    compressed[key] = [self._compress_json(item) if isinstance(item, dict) else item for item in value]
                else:
                    compressed[key] = value
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error in JSON compression: {e}")
            return data
    
    def _compress_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress text fields."""
        try:
            compressed = {}
            
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 100:
                    # Compress long strings
                    compressed_bytes = zlib.compress(value.encode('utf-8'))
                    compressed[key] = f"COMPRESSED:{compressed_bytes.hex()}"
                elif isinstance(value, dict):
                    compressed[key] = self._compress_text(value)
                elif isinstance(value, list):
                    compressed[key] = [self._compress_text(item) if isinstance(item, dict) else item for item in value]
                else:
                    compressed[key] = value
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error in text compression: {e}")
            return data
    
    def _compress_numbers(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress numeric fields."""
        try:
            compressed = {}
            
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    # No compression needed for numbers, but could apply rounding
                    if isinstance(value, float):
                        compressed[key] = round(value, 6)  # Round to 6 decimal places
                    else:
                        compressed[key] = value
                elif isinstance(value, dict):
                    compressed[key] = self._compress_numbers(value)
                elif isinstance(value, list):
                    compressed[key] = [self._compress_numbers(item) if isinstance(item, dict) else item for item in value]
                else:
                    compressed[key] = value
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error in number compression: {e}")
            return data
    
    def _compress_booleans(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress boolean fields."""
        try:
            compressed = {}
            
            for key, value in data.items():
                if isinstance(value, bool):
                    # Convert to int for smaller representation
                    compressed[key] = 1 if value else 0
                elif isinstance(value, dict):
                    compressed[key] = self._compress_booleans(value)
                elif isinstance(value, list):
                    compressed[key] = [self._compress_booleans(item) if isinstance(item, dict) else item for item in value]
                else:
                    compressed[key] = value
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error in boolean compression: {e}")
            return data
    
    def _compress_lists(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress list fields."""
        try:
            compressed = {}
            
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 10:
                    # Compress long lists
                    list_str = json.dumps(value)
                    compressed_bytes = zlib.compress(list_str.encode('utf-8'))
                    compressed[key] = f"LIST_COMPRESSED:{compressed_bytes.hex()}"
                elif isinstance(value, dict):
                    compressed[key] = self._compress_lists(value)
                elif isinstance(value, list):
                    compressed[key] = [self._compress_lists(item) if isinstance(item, dict) else item for item in value]
                else:
                    compressed[key] = value
            
            return compressed
            
        except Exception as e:
            logger.error(f"Error in list compression: {e}")
            return data
    
    def _decompress_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress JSON data."""
        # For now, JSON compression is just copying
        return data.copy()
    
    def _decompress_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress text fields."""
        try:
            decompressed = {}
            
            for key, value in data.items():
                if isinstance(value, str) and value.startswith("COMPRESSED:"):
                    # Decompress compressed text
                    hex_data = value[11:]  # Remove "COMPRESSED:" prefix
                    compressed_bytes = bytes.fromhex(hex_data)
                    decompressed_text = zlib.decompress(compressed_bytes).decode('utf-8')
                    decompressed[key] = decompressed_text
                elif isinstance(value, str) and value.startswith("LIST_COMPRESSED:"):
                    # Decompress compressed list
                    hex_data = value[15:]  # Remove "LIST_COMPRESSED:" prefix
                    compressed_bytes = bytes.fromhex(hex_data)
                    list_str = zlib.decompress(compressed_bytes).decode('utf-8')
                    decompressed[key] = json.loads(list_str)
                elif isinstance(value, dict):
                    decompressed[key] = self._decompress_text(value)
                elif isinstance(value, list):
                    decompressed[key] = [self._decompress_text(item) if isinstance(item, dict) else item for item in value]
                else:
                    decompressed[key] = value
            
            return decompressed
            
        except Exception as e:
            logger.error(f"Error in text decompression: {e}")
            return data
    
    def _remove_low_priority_fields(self, data: Dict[str, Any], target_tokens: int) -> Dict[str, Any]:
        """Remove low-priority fields to meet token target."""
        current_tokens = self.token_manager.count_tokens_dict(data)
        
        if current_tokens <= target_tokens:
            return data
        
        # Define field priorities (lower number = higher priority)
        field_priorities = {
            'id': 1, 'name': 1, 'type': 1, 'status': 2,
            'content': 3, 'description': 3, 'details': 4,
            'metadata': 5, 'debug': 6, 'temp': 7
        }
        
        # Sort fields by priority (lowest first)
        sorted_fields = sorted(
            data.items(),
            key=lambda x: field_priorities.get(x[0].lower(), 10)
        )
        
        # Remove low-priority fields until under token limit
        optimized = {}
        for key, value in sorted_fields:
            optimized[key] = value
            current_tokens = self.token_manager.count_tokens_dict(optimized)
            if current_tokens <= target_tokens:
                break
        
        return optimized
    
    def _summarize_long_text(self, data: Dict[str, Any], target_tokens: int) -> Dict[str, Any]:
        """Summarize long text fields."""
        current_tokens = self.token_manager.count_tokens_dict(data)
        
        if current_tokens <= target_tokens:
            return data
        
        optimized = {}
        
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 500:
                # Truncate long text
                max_length = int(500 * (target_tokens / current_tokens))
                optimized[key] = value[:max_length] + "..." if len(value) > max_length else value
            else:
                optimized[key] = value
        
        return optimized