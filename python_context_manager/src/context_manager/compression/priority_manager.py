"""
Priority management for context optimization.

This module provides utilities for managing context priorities
and optimizing based on importance.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from ..utils.helpers import ContextManagerError
from ..utils.logging import get_logger


logger = get_logger(__name__)


class PriorityLevel(Enum):
    """Priority levels for context items."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


@dataclass
class PriorityRule:
    """Rule for determining priority."""
    field_name: str
    priority: PriorityLevel
    condition: Optional[str] = None  # Condition for applying this rule
    weight: float = 1.0


@dataclass
class ContextItem:
    """Item in context with priority information."""
    key: str
    value: Any
    priority: PriorityLevel
    last_accessed: float
    access_count: int = 0
    size: int = 0


class PriorityManager:
    """Manages context priorities and optimization."""
    
    def __init__(self):
        """Initialize priority manager."""
        self.priority_rules: List[PriorityRule] = []
        self.context_items: Dict[str, ContextItem] = {}
        self.access_history: List[Tuple[str, float]] = []
        
        # Default priority rules
        self._setup_default_rules()
    
    def add_priority_rule(self, field_name: str, priority: PriorityLevel, 
                          condition: Optional[str] = None, weight: float = 1.0) -> None:
        """
        Add a priority rule.
        
        Args:
            field_name: Name of the field
            priority: Priority level
            condition: Optional condition for applying the rule
            weight: Weight of the rule (higher = more important)
        """
        rule = PriorityRule(field_name, priority, condition, weight)
        self.priority_rules.append(rule)
        
        # Sort rules by weight (highest first)
        self.priority_rules.sort(key=lambda x: x.weight, reverse=True)
    
    def calculate_priority(self, key: str, value: Any) -> PriorityLevel:
        """
        Calculate priority for a context item.
        
        Args:
            key: Item key
            value: Item value
            
        Returns:
            Priority level
        """
        priority = PriorityLevel.MEDIUM  # Default priority
        
        for rule in self.priority_rules:
            if self._matches_rule(key, value, rule):
                priority = rule.priority
                break
        
        # Adjust based on value characteristics
        priority = self._adjust_priority_based_on_value(key, value, priority)
        
        return priority
    
    def track_access(self, key: str) -> None:
        """
        Track access to a context item.
        
        Args:
            key: Item key
        """
        current_time = time.time()
        
        if key in self.context_items:
            item = self.context_items[key]
            item.last_accessed = current_time
            item.access_count += 1
        else:
            logger.warning(f"Attempted to track access for unknown key: {key}")
        
        # Add to access history
        self.access_history.append((key, current_time))
        
        # Keep only recent history (last 1000 accesses)
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]
    
    def get_prioritized_items(self, context: Dict[str, Any]) -> List[ContextItem]:
        """
        Get context items sorted by priority.
        
        Args:
            context: Context data
            
        Returns:
            List of context items sorted by priority
        """
        items = []
        
        for key, value in context.items():
            priority = self.calculate_priority(key, value)
            
            # Get existing item or create new one
            if key in self.context_items:
                item = self.context_items[key]
                item.value = value
                item.priority = priority
            else:
                item = ContextItem(
                    key=key,
                    value=value,
                    priority=priority,
                    last_accessed=time.time(),
                    size=self._calculate_size(value)
                )
                self.context_items[key] = item
            
            items.append(item)
        
        # Sort by priority (highest first), then by access recency
        items.sort(key=lambda x: (x.priority.value, x.last_accessed), reverse=True)
        
        return items
    
    def optimize_context(self, context: Dict[str, Any], max_items: int, 
                        max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize context based on priorities.
        
        Args:
            context: Context to optimize
            max_items: Maximum number of items to keep
            max_tokens: Maximum tokens (optional)
            
        Returns:
            Optimized context
        """
        prioritized_items = self.get_prioritized_items(context)
        
        # Select top items
        selected_items = prioritized_items[:max_items]
        
        # Build optimized context
        optimized = {}
        for item in selected_items:
            optimized[item.key] = item.value
        
        # Apply token limit if specified
        if max_tokens:
            optimized = self._apply_token_limit(optimized, max_tokens)
        
        return optimized
    
    def get_priority_stats(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get priority statistics for context.
        
        Args:
            context: Context data
            
        Returns:
            Priority statistics
        """
        items = self.get_prioritized_items(context)
        
        stats = {
            'total_items': len(items),
            'priority_distribution': defaultdict(int),
            'most_accessed': [],
            'least_accessed': [],
            'highest_priority': [],
            'lowest_priority': []
        }
        
        for item in items:
            stats['priority_distribution'][item.priority.name] += 1
        
        # Get most and least accessed items
        sorted_by_access = sorted(items, key=lambda x: x.access_count, reverse=True)
        stats['most_accessed'] = [(item.key, item.access_count) for item in sorted_by_access[:5]]
        stats['least_accessed'] = [(item.key, item.access_count) for item in sorted_by_access[-5:]]
        
        # Get highest and lowest priority items
        stats['highest_priority'] = [item.key for item in items if item.priority == PriorityLevel.CRITICAL]
        stats['lowest_priority'] = [item.key for item in items if item.priority == PriorityLevel.MINIMAL]
        
        return dict(stats)
    
    def cleanup_old_items(self, max_age: float = 86400) -> None:
        """
        Clean up old items from tracking.
        
        Args:
            max_age: Maximum age in seconds (default: 24 hours)
        """
        current_time = time.time()
        old_keys = []
        
        for key, item in self.context_items.items():
            if current_time - item.last_accessed > max_age:
                old_keys.append(key)
        
        for key in old_keys:
            del self.context_items[key]
        
        if old_keys:
            logger.info(f"Cleaned up {len(old_keys)} old items from priority tracking")
    
    def _setup_default_rules(self) -> None:
        """Setup default priority rules."""
        default_rules = [
            # Critical items
            PriorityRule("id", PriorityLevel.CRITICAL, weight=2.0),
            PriorityRule("user_id", PriorityLevel.CRITICAL, weight=2.0),
            PriorityRule("session_id", PriorityLevel.CRITICAL, weight=2.0),
            PriorityRule("message", PriorityLevel.CRITICAL, weight=2.0),
            PriorityRule("content", PriorityLevel.CRITICAL, weight=2.0),
            
            # High priority items
            PriorityRule("name", PriorityLevel.HIGH, weight=1.5),
            PriorityRule("type", PriorityLevel.HIGH, weight=1.5),
            PriorityRule("status", PriorityLevel.HIGH, weight=1.5),
            PriorityRule("action", PriorityLevel.HIGH, weight=1.5),
            PriorityRule("result", PriorityLevel.HIGH, weight=1.5),
            
            # Medium priority items
            PriorityRule("description", PriorityLevel.MEDIUM, weight=1.0),
            PriorityRule("metadata", PriorityLevel.MEDIUM, weight=1.0),
            PriorityRule("parameters", PriorityLevel.MEDIUM, weight=1.0),
            PriorityRule("timestamp", PriorityLevel.MEDIUM, weight=1.0),
            
            # Low priority items
            PriorityRule("debug", PriorityLevel.LOW, weight=0.5),
            PriorityRule("log", PriorityLevel.LOW, weight=0.5),
            PriorityRule("temp", PriorityLevel.LOW, weight=0.5),
            
            # Minimal priority items
            PriorityRule("cache", PriorityLevel.MINIMAL, weight=0.2),
            PriorityRule("history", PriorityLevel.MINIMAL, weight=0.2),
        ]
        
        self.priority_rules.extend(default_rules)
    
    def _matches_rule(self, key: str, value: Any, rule: PriorityRule) -> bool:
        """Check if item matches a priority rule."""
        if rule.field_name != "*" and key.lower() != rule.field_name.lower():
            return False
        
        if rule.condition:
            # Simple condition checking (can be expanded)
            try:
                return eval(rule.condition, {"key": key, "value": value})
            except Exception:
                return False
        
        return True
    
    def _adjust_priority_based_on_value(self, key: str, value: Any, 
                                       base_priority: PriorityLevel) -> PriorityLevel:
        """Adjust priority based on value characteristics."""
        # Boost priority for important content
        if isinstance(value, str):
            if any(keyword in value.lower() for keyword in ["error", "exception", "critical", "urgent"]):
                return PriorityLevel(max(base_priority.value, PriorityLevel.HIGH.value))
            elif len(value) > 1000:
                return PriorityLevel(max(base_priority.value, PriorityLevel.MEDIUM.value))
        
        # Boost priority for non-empty values
        if value and not isinstance(value, (dict, list)):
            return PriorityLevel(max(base_priority.value, PriorityLevel.MEDIUM.value))
        
        return base_priority
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            import pickle
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value).encode('utf-8'))
    
    def _apply_token_limit(self, context: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Apply token limit to context."""
        # Simple token counting (can be replaced with proper tokenizer)
        def count_tokens(text: str) -> int:
            return len(text.split()) * 1.3  # Rough estimate
        
        def count_context_tokens(ctx: Dict[str, Any]) -> int:
            total = 0
            for k, v in ctx.items():
                total += count_tokens(k)
                if isinstance(v, str):
                    total += count_tokens(v)
                elif isinstance(v, dict):
                    total += count_context_tokens(v)
                else:
                    total += count_tokens(str(v))
            return total
        
        current_tokens = count_context_tokens(context)
        
        if current_tokens <= max_tokens:
            return context
        
        # Remove lowest priority items until under limit
        prioritized_items = self.get_prioritized_items(context)
        optimized = {}
        
        for item in prioritized_items:
            optimized[item.key] = item.value
            if count_context_tokens(optimized) <= max_tokens:
                break
        
        return optimized