"""
Tool system for the context manager.

This module provides a comprehensive tool system that supports:
- Tool registration and discovery
- Tool execution with timeout and error handling
- Tool result management
- Tool dependencies and chains
- Tool security and validation
"""

import asyncio
import inspect
import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union, Type
from dataclasses import dataclass, field
from functools import wraps
import uuid

from .models import ToolStatus, ToolResult
from ..utils.logging import get_logger
from ..utils.error_handling import ToolError
from ..utils.helpers import Timer, retry, RateLimiter


logger = get_logger(__name__)


class ToolCategory(Enum):
    """Categories of tools."""
    UTILITY = "utility"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    CUSTOM = "custom"


class ToolPermission(Enum):
    """Permission levels for tools."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    ADMIN = "admin"


@dataclass
class ToolMetadata:
    """Metadata for a tool."""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    permission: ToolPermission = ToolPermission.PUBLIC
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit: Optional[int] = None
    rate_window: float = 60.0
    dependencies: List[str] = field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_execution_time: float = 0.0
    last_executed: Optional[datetime] = None


@dataclass
class ToolExecutionContext:
    """Context for tool execution."""
    execution_id: str
    tool_name: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context_window: Optional[Any] = None
    memory_manager: Optional[Any] = None
    start_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolInterface(ABC):
    """Base interface for all tools."""
    
    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self.rate_limiter = None
        if metadata.rate_limit:
            self.rate_limiter = RateLimiter(
                max_calls=metadata.rate_limit,
                time_window=metadata.rate_window
            )
    
    @abstractmethod
    async def execute(self, context: ToolExecutionContext) -> ToolResult:
        """Execute the tool with the given context."""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters."""
        pass
    
    async def pre_execute(self, context: ToolExecutionContext) -> bool:
        """Hook called before execution."""
        return True
    
    async def post_execute(self, context: ToolExecutionContext, result: ToolResult) -> ToolResult:
        """Hook called after execution."""
        return result
    
    def update_stats(self, execution_time: float, success: bool):
        """Update tool execution statistics."""
        self.metadata.execution_count += 1
        self.metadata.last_executed = datetime.now()
        
        if success:
            self.metadata.success_count += 1
        else:
            self.metadata.failure_count += 1
        
        # Update average execution time
        total_time = self.metadata.average_execution_time * (self.metadata.execution_count - 1)
        total_time += execution_time
        self.metadata.average_execution_time = total_time / self.metadata.execution_count


class FunctionTool(ToolInterface):
    """Tool that wraps a Python function."""
    
    def __init__(self, func: Callable, metadata: ToolMetadata):
        super().__init__(metadata)
        self.func = func
        self._validate_function_signature()
    
    def _validate_function_signature(self):
        """Validate that the function signature is compatible."""
        sig = inspect.signature(self.func)
        params = list(sig.parameters.keys())
        
        if not params:
            raise ToolError(f"Function {self.func.__name__} must accept at least one parameter")
    
    async def execute(self, context: ToolExecutionContext) -> ToolResult:
        """Execute the wrapped function."""
        if not self.validate_parameters(context.parameters):
            return ToolResult(
                success=False,
                error="Invalid parameters",
                tool_name=self.metadata.name,
                parameters=context.parameters,
                status=ToolStatus.FAILED
            )
        
        # Check rate limit
        if self.rate_limiter and not self.rate_limiter.is_allowed():
            wait_time = self.rate_limiter.get_wait_time()
            return ToolResult(
                success=False,
                error=f"Rate limit exceeded. Wait {wait_time:.1f}s",
                tool_name=self.metadata.name,
                parameters=context.parameters,
                status=ToolStatus.FAILED
            )
        
        # Pre-execute hook
        if not await self.pre_execute(context):
            return ToolResult(
                success=False,
                error="Pre-execution hook failed",
                tool_name=self.metadata.name,
                parameters=context.parameters,
                status=ToolStatus.FAILED
            )
        
        start_time = time.time()
        
        try:
            # Execute function
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(context.parameters)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.func, context.parameters
                )
            
            execution_time = time.time() - start_time
            
            tool_result = ToolResult(
                success=True,
                result=result,
                execution_time=execution_time,
                tool_name=self.metadata.name,
                parameters=context.parameters,
                status=ToolStatus.COMPLETED
            )
            
            # Update stats
            self.update_stats(execution_time, True)
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool {self.metadata.name} execution failed: {e}")
            
            tool_result = ToolResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                tool_name=self.metadata.name,
                parameters=context.parameters,
                status=ToolStatus.FAILED
            )
            
            # Update stats
            self.update_stats(execution_time, False)
        
        # Post-execute hook
        tool_result = await self.post_execute(context, tool_result)
        
        return tool_result
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate function parameters."""
        try:
            sig = inspect.signature(self.func)
            bound_args = sig.bind(**parameters)
            bound_args.apply_defaults()
            return True
        except Exception as e:
            logger.warning(f"Parameter validation failed for {self.metadata.name}: {e}")
            return False


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self.tools: Dict[str, ToolInterface] = {}
        self.categories: Dict[ToolCategory, Set[str]] = {}
        self.tags: Dict[str, Set[str]] = {}
        self.dependencies: Dict[str, Set[str]] = {}
    
    def register_tool(self, tool: ToolInterface) -> bool:
        """Register a tool."""
        if tool.metadata.name in self.tools:
            logger.warning(f"Tool {tool.metadata.name} already registered, overwriting")
        
        self.tools[tool.metadata.name] = tool
        
        # Update categories
        if tool.metadata.category not in self.categories:
            self.categories[tool.metadata.category] = set()
        self.categories[tool.metadata.category].add(tool.metadata.name)
        
        # Update tags
        for tag in tool.metadata.tags:
            if tag not in self.tags:
                self.tags[tag] = set()
            self.tags[tag].add(tool.metadata.name)
        
        # Update dependencies
        self.dependencies[tool.metadata.name] = set(tool.metadata.dependencies)
        
        logger.info(f"Registered tool: {tool.metadata.name}")
        return True
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name not in self.tools:
            return False
        
        tool = self.tools[tool_name]
        
        # Remove from categories
        if tool.metadata.category in self.categories:
            self.categories[tool.metadata.category].discard(tool_name)
        
        # Remove from tags
        for tag in tool.metadata.tags:
            if tag in self.tags:
                self.tags[tag].discard(tool_name)
        
        # Remove from dependencies
        self.dependencies.pop(tool_name, None)
        
        # Remove tool
        self.tools.pop(tool_name, None)
        
        logger.info(f"Unregistered tool: {tool_name}")
        return True
    
    def get_tool(self, tool_name: str) -> Optional[ToolInterface]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolInterface]:
        """Get all tools in a category."""
        tool_names = self.categories.get(category, set())
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_tools_by_tag(self, tag: str) -> List[ToolInterface]:
        """Get all tools with a specific tag."""
        tool_names = self.tags.get(tag, set())
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def search_tools(self, query: str) -> List[ToolInterface]:
        """Search tools by name or description."""
        query = query.lower()
        results = []
        
        for tool in self.tools.values():
            if (query in tool.metadata.name.lower() or 
                query in tool.metadata.description.lower() or
                any(query in tag.lower() for tag in tool.metadata.tags)):
                results.append(tool)
        
        return results
    
    def get_all_tools(self) -> List[ToolInterface]:
        """Get all registered tools."""
        return list(self.tools.values())
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about registered tools."""
        stats = {
            "total_tools": len(self.tools),
            "active_tools": sum(1 for tool in self.tools.values() if tool.metadata.is_active),
            "categories": {cat.value: len(tools) for cat, tools in self.categories.items()},
            "top_tools": sorted(
                [(tool.metadata.name, tool.metadata.execution_count) 
                 for tool in self.tools.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        return stats


class ToolExecutor:
    """Executes tools with various strategies."""
    
    def __init__(self, registry: ToolRegistry, max_workers: int = 5):
        self.registry = registry
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          context: Optional[ToolExecutionContext] = None) -> ToolResult:
        """Execute a single tool."""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool {tool_name} not found",
                tool_name=tool_name,
                parameters=parameters,
                status=ToolStatus.FAILED
            )
        
        if not tool.metadata.is_active:
            return ToolResult(
                success=False,
                error=f"Tool {tool_name} is not active",
                tool_name=tool_name,
                parameters=parameters,
                status=ToolStatus.FAILED
            )
        
        # Create execution context if not provided
        if context is None:
            context = ToolExecutionContext(
                execution_id=str(uuid.uuid4()),
                tool_name=tool_name,
                parameters=parameters
            )
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(context),
                timeout=tool.metadata.timeout
            )
        except asyncio.TimeoutError:
            result = ToolResult(
                success=False,
                error=f"Tool {tool_name} execution timed out",
                tool_name=tool_name,
                parameters=parameters,
                status=ToolStatus.TIMEOUT
            )
        
        # Record execution
        self._record_execution(context, result)
        
        return result
    
    async def execute_tools_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute multiple tools in parallel."""
        tasks = []
        
        for call in tool_calls:
            task = self.execute_tool(
                call["tool_name"],
                call.get("parameters", {}),
                call.get("context")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ToolResult
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ToolResult(
                        success=False,
                        error=str(result),
                        tool_name=tool_calls[i]["tool_name"],
                        parameters=tool_calls[i].get("parameters", {}),
                        status=ToolStatus.FAILED
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_tool_chain(self, chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a chain of tools where output of one feeds into the next."""
        context = {}
        results = []
        
        for i, step in enumerate(chain):
            tool_name = step["tool_name"]
            parameters = step.get("parameters", {})
            
            # Merge context with parameters
            merged_params = {**context, **parameters}
            
            # Execute tool
            result = await self.execute_tool(tool_name, merged_params)
            results.append(result)
            
            if not result.success:
                # Stop chain on failure
                break
            
            # Update context with result
            if isinstance(result.result, dict):
                context.update(result.result)
            else:
                context[f"step_{i}_result"] = result.result
        
        return {
            "chain_results": results,
            "final_context": context,
            "success": all(r.success for r in results)
        }
    
    def _record_execution(self, context: ToolExecutionContext, result: ToolResult):
        """Record execution in history."""
        record = {
            "execution_id": context.execution_id,
            "tool_name": context.tool_name,
            "parameters": context.parameters,
            "result": result.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.execution_history.append(record)
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_execution_history(self, tool_name: Optional[str] = None, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history."""
        history = self.execution_history
        
        if tool_name:
            history = [record for record in history if record["tool_name"] == tool_name]
        
        return history[-limit:]
    
    def clear_execution_history(self):
        """Clear execution history."""
        self.execution_history.clear()


def tool(name: str, description: str, category: ToolCategory = ToolCategory.CUSTOM,
         **kwargs):
    """Decorator to register a function as a tool."""
    def decorator(func):
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            **kwargs
        )
        
        # Create function tool
        tool_instance = FunctionTool(func, metadata)
        
        # Register tool (this would be handled by the tool manager)
        func._tool_instance = tool_instance
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator