"""
Tool Manager for coordinating tool operations.

This module provides the main interface for tool management, including:
- Tool registration and discovery
- Tool execution coordination
- Tool lifecycle management
- Tool performance monitoring
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .tool_system import (
    ToolInterface, ToolRegistry, ToolExecutor, ToolMetadata,
    ToolCategory, ToolPermission, ToolExecutionContext,
    FunctionTool, tool
)
from .models import ToolResult, ToolStatus
from .config import ContextManagerConfig
from ..utils.logging import get_logger
from ..utils.error_handling import ToolError


logger = get_logger(__name__)


@dataclass
class ToolExecutionRequest:
    """Request for tool execution."""
    tool_name: str
    parameters: Dict[str, Any]
    execution_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: str = "normal"
    timeout: Optional[float] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ToolExecutionResponse:
    """Response from tool execution."""
    success: bool
    result: Any
    execution_id: str
    tool_name: str
    execution_time: float
    error: Optional[str] = None
    status: ToolStatus = ToolStatus.COMPLETED
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolManager:
    """Main tool management class."""
    
    def __init__(self, config: ContextManagerConfig):
        self.config = config
        self.registry = ToolRegistry()
        self.executor = ToolExecutor(self.registry, max_workers=config.max_workers)
        
        # Built-in tools
        self._register_builtin_tools()
        
        # Tool execution tracking
        self.active_executions: Dict[str, ToolExecutionContext] = {}
        self.execution_queue: List[ToolExecutionRequest] = []
        
        # Performance monitoring
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "tools_usage": {},
            "hourly_stats": {}
        }
        
        logger.info("ToolManager initialized")
    
    def _register_builtin_tools(self):
        """Register built-in tools."""
        
        @tool(
            name="echo",
            description="Echo back the input parameters",
            category=ToolCategory.UTILITY,
            timeout=5.0,
            tags=["utility", "echo"]
        )
        async def echo_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Echo tool for testing."""
            return {
                "echo": parameters,
                "timestamp": datetime.now().isoformat()
            }
        
        @tool(
            name="get_system_info",
            description="Get system information",
            category=ToolCategory.SYSTEM,
            timeout=10.0,
            tags=["system", "info"]
        )
        async def get_system_info_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Get system information."""
            import platform
            import psutil
            
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": {
                    "total": psutil.disk_usage('/').total,
                    "used": psutil.disk_usage('/').used,
                    "free": psutil.disk_usage('/').free
                }
            }
        
        @tool(
            name="calculate",
            description="Perform mathematical calculations",
            category=ToolCategory.UTILITY,
            timeout=5.0,
            tags=["utility", "math", "calculation"]
        )
        async def calculate_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate mathematical expressions."""
            expression = parameters.get("expression", "")
            
            try:
                # Safe evaluation of mathematical expressions
                allowed_names = {
                    "abs": abs, "round": round, "min": min, "max": max,
                    "sum": sum, "len": len, "pow": pow
                }
                
                # Create safe environment
                safe_env = {"__builtins__": {}, **allowed_names}
                
                # Evaluate expression
                result = eval(expression, safe_env)
                
                return {
                    "expression": expression,
                    "result": result,
                    "type": type(result).__name__
                }
            except Exception as e:
                return {
                    "expression": expression,
                    "error": str(e)
                }
        
        @tool(
            name="format_text",
            description="Format and manipulate text",
            category=ToolCategory.UTILITY,
            timeout=5.0,
            tags=["utility", "text", "formatting"]
        )
        async def format_text_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Format text with various options."""
            text = parameters.get("text", "")
            operation = parameters.get("operation", "uppercase")
            
            if operation == "uppercase":
                result = text.upper()
            elif operation == "lowercase":
                result = text.lower()
            elif operation == "capitalize":
                result = text.capitalize()
            elif operation == "reverse":
                result = text[::-1]
            elif operation == "word_count":
                result = len(text.split())
            elif operation == "character_count":
                result = len(text)
            else:
                result = text
            
            return {
                "original": text,
                "operation": operation,
                "result": result
            }
        
        # Register the built-in tools
        for func in [echo_tool, get_system_info_tool, calculate_tool, format_text_tool]:
            if hasattr(func, '_tool_instance'):
                self.registry.register_tool(func._tool_instance)
    
    def register_tool(self, tool: ToolInterface) -> bool:
        """Register a custom tool."""
        return self.registry.register_tool(tool)
    
    def register_function_tool(self, name: str, func: Callable, 
                               description: str = "", 
                               category: ToolCategory = ToolCategory.CUSTOM,
                               **kwargs) -> bool:
        """
        Register a function as a tool (convenience method).
        
        Args:
            name: Tool name
            func: Function to register as tool
            description: Tool description
            category: Tool category
            **kwargs: Additional tool metadata
            
        Returns:
            True if registration was successful
        """
        metadata = ToolMetadata(
            name=name,
            description=description or f"Tool: {name}",
            category=category,
            **kwargs
        )
        
        tool_instance = FunctionTool(func, metadata)
        return self.registry.register_tool(tool_instance)
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        return self.registry.unregister_tool(tool_name)
    
    def get_tool(self, tool_name: str) -> Optional[ToolInterface]:
        """Get a tool by name."""
        return self.registry.get_tool(tool_name)
    
    def list_tools(self, category: Optional[ToolCategory] = None, 
                  tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools."""
        if category:
            tools = self.registry.get_tools_by_category(category)
        elif tag:
            tools = self.registry.get_tools_by_tag(tag)
        else:
            tools = self.registry.get_all_tools()
        
        return [
            {
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "category": tool.metadata.category.value,
                "version": tool.metadata.version,
                "tags": tool.metadata.tags,
                "permission": tool.metadata.permission.value,
                "timeout": tool.metadata.timeout,
                "is_active": tool.metadata.is_active,
                "execution_count": tool.metadata.execution_count,
                "success_rate": (
                    tool.metadata.success_count / tool.metadata.execution_count
                    if tool.metadata.execution_count > 0 else 0.0
                )
            }
            for tool in tools
        ]
    
    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search for tools."""
        tools = self.registry.search_tools(query)
        
        return [
            {
                "name": tool.metadata.name,
                "description": tool.metadata.description,
                "category": tool.metadata.category.value,
                "tags": tool.metadata.tags,
                "relevance_score": self._calculate_relevance_score(query, tool)
            }
            for tool in tools
        ]
    
    def _calculate_relevance_score(self, query: str, tool: ToolInterface) -> float:
        """Calculate relevance score for search."""
        query = query.lower()
        score = 0.0
        
        # Name match (highest weight)
        if query in tool.metadata.name.lower():
            score += 0.5
        
        # Description match
        if query in tool.metadata.description.lower():
            score += 0.3
        
        # Tag matches
        for tag in tool.metadata.tags:
            if query in tag.lower():
                score += 0.1
        
        return min(score, 1.0)
    
    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResponse:
        """Execute a tool."""
        # Generate execution ID if not provided
        if not request.execution_id:
            import uuid
            request.execution_id = str(uuid.uuid4())
        
        # Create execution context
        context = ToolExecutionContext(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            parameters=request.parameters,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata=request.context or {}
        )
        
        # Track active execution
        self.active_executions[request.execution_id] = context
        
        try:
            # Execute tool
            result = await self.executor.execute_tool(
                request.tool_name,
                request.parameters,
                context
            )
            
            # Update performance metrics
            self._update_performance_metrics(request.tool_name, result)
            
            # Create response
            response = ToolExecutionResponse(
                success=result.success,
                result=result.result,
                execution_id=request.execution_id,
                tool_name=request.tool_name,
                execution_time=result.execution_time,
                error=result.error,
                status=result.status
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            
            response = ToolExecutionResponse(
                success=False,
                result=None,
                execution_id=request.execution_id,
                tool_name=request.tool_name,
                execution_time=0.0,
                error=str(e),
                status=ToolStatus.FAILED
            )
        
        finally:
            # Remove from active executions
            self.active_executions.pop(request.execution_id, None)
        
        return response
    
    async def execute_tool_simple(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool with simple keyword arguments (convenience method).
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        request = ToolExecutionRequest(
            tool_name=tool_name,
            parameters=kwargs
        )
        
        response = await self.execute_tool(request)
        
        # Convert response to ToolResult
        return ToolResult(
            success=response.success,
            result=response.result,
            error=response.error,
            execution_time=response.execution_time,
            tool_name=response.tool_name,
            parameters=kwargs,  # Use the original parameters
            status=response.status
        )
    
    async def execute_tools_parallel(self, requests: List[ToolExecutionRequest]) -> List[ToolExecutionResponse]:
        """Execute multiple tools in parallel."""
        # Create execution tasks
        tasks = [self.execute_tool(request) for request in requests]
        
        # Execute in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                processed_responses.append(
                    ToolExecutionResponse(
                        success=False,
                        result=None,
                        execution_id=requests[i].execution_id or "unknown",
                        tool_name=requests[i].tool_name,
                        execution_time=0.0,
                        error=str(response),
                        status=ToolStatus.FAILED
                    )
                )
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    async def execute_tool_chain(self, chain: List[ToolExecutionRequest]) -> Dict[str, Any]:
        """Execute a chain of tools."""
        chain_results = []
        context = {}
        
        for request in chain:
            # Merge context with parameters
            merged_params = {**context, **request.parameters}
            request.parameters = merged_params
            
            # Execute tool
            response = await self.execute_tool(request)
            chain_results.append(response)
            
            if not response.success:
                # Stop chain on failure
                break
            
            # Update context with result
            if isinstance(response.result, dict):
                context.update(response.result)
            else:
                context[f"step_{len(chain_results)}_result"] = response.result
        
        return {
            "chain_results": [r.__dict__ for r in chain_results],
            "final_context": context,
            "success": all(r.success for r in chain_results)
        }
    
    def _update_performance_metrics(self, tool_name: str, result: ToolResult):
        """Update performance metrics."""
        self.performance_metrics["total_executions"] += 1
        
        if result.success:
            self.performance_metrics["successful_executions"] += 1
        else:
            self.performance_metrics["failed_executions"] += 1
        
        # Update tool usage
        if tool_name not in self.performance_metrics["tools_usage"]:
            self.performance_metrics["tools_usage"][tool_name] = {
                "count": 0,
                "success_count": 0,
                "total_time": 0.0
            }
        
        usage = self.performance_metrics["tools_usage"][tool_name]
        usage["count"] += 1
        usage["total_time"] += result.execution_time
        
        if result.success:
            usage["success_count"] += 1
        
        # Update average execution time
        total_time = self.performance_metrics["average_execution_time"] * (
            self.performance_metrics["total_executions"] - 1
        )
        total_time += result.execution_time
        self.performance_metrics["average_execution_time"] = (
            total_time / self.performance_metrics["total_executions"]
        )
        
        # Update hourly stats
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        if current_hour not in self.performance_metrics["hourly_stats"]:
            self.performance_metrics["hourly_stats"][current_hour] = {
                "executions": 0,
                "success_rate": 0.0
            }
        
        self.performance_metrics["hourly_stats"][current_hour]["executions"] += 1
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool statistics."""
        return {
            "performance_metrics": self.performance_metrics,
            "registry_stats": self.registry.get_tool_stats(),
            "active_executions": len(self.active_executions),
            "queue_size": len(self.execution_queue)
        }
    
    def get_execution_history(self, tool_name: Optional[str] = None, 
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.executor.get_execution_history(tool_name, limit)
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get currently active executions."""
        return [
            {
                "execution_id": exec_id,
                "tool_name": context.tool_name,
                "start_time": context.start_time.isoformat(),
                "user_id": context.user_id,
                "session_id": context.session_id
            }
            for exec_id, context in self.active_executions.items()
        ]
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id in self.active_executions:
            # Note: This is a simplified implementation
            # In a real system, you'd need to handle task cancellation
            self.active_executions.pop(exec_id, None)
            logger.info(f"Cancelled execution: {execution_id}")
            return True
        return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the tool system."""
        return {
            "status": "healthy",
            "registered_tools": len(self.registry.tools),
            "active_executions": len(self.active_executions),
            "queue_size": len(self.execution_queue),
            "performance": {
                "total_executions": self.performance_metrics["total_executions"],
                "success_rate": (
                    self.performance_metrics["successful_executions"] / 
                    max(1, self.performance_metrics["total_executions"])
                ),
                "average_execution_time": self.performance_metrics["average_execution_time"]
            }
        }