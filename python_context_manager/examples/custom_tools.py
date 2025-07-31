"""
Custom tools example for the Python Context Manager.

This example demonstrates how to create and use custom tools:
- Creating custom tool functions
- Tool registration and management
- Tool execution with parameters
- Tool chaining and composition
- Tool error handling
- Tool permissions and security
"""

import asyncio
import tempfile
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from context_manager.core.enhanced_context_manager import EnhancedContextManager
from context_manager.core.config import ContextManagerConfig
from context_manager.core.tool_system import ToolCategory, ToolPermission, tool


# Custom tool examples
@tool(
    name="text_analyzer",
    description="Analyze text and provide statistics",
    category=ToolCategory.ANALYSIS,
    tags=["text", "analysis", "statistics"],
    timeout=10.0
)
async def text_analyzer(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze text and provide various statistics.
    
    Parameters:
        text: The text to analyze
        include_word_count: Whether to include word count (default: True)
        include_char_count: Whether to include character count (default: True)
        include_sentiment: Whether to include sentiment analysis (default: False)
    
    Returns:
        Dictionary with analysis results
    """
    text = parameters.get("text", "")
    include_word_count = parameters.get("include_word_count", True)
    include_char_count = parameters.get("include_char_count", True)
    include_sentiment = parameters.get("include_sentiment", False)
    
    result = {"text_length": len(text)}
    
    if include_word_count:
        words = text.split()
        result["word_count"] = len(words)
        result["average_word_length"] = sum(len(word) for word in words) / len(words) if words else 0
    
    if include_char_count:
        result["character_count"] = len(text)
        result["character_count_no_spaces"] = len(text.replace(" ", ""))
    
    if include_sentiment:
        # Simple sentiment analysis (in real implementation, use a proper NLP library)
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disgusting", "hate"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        result["sentiment"] = {
            "positive_words": positive_count,
            "negative_words": negative_count,
            "overall": "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"
        }
    
    return result


@tool(
    name="data_transformer",
    description="Transform data between different formats",
    category=ToolCategory.DATA_PROCESSING,
    tags=["data", "transform", "json", "csv"],
    timeout=15.0
)
async def data_transformer(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform data between different formats.
    
    Parameters:
        data: The data to transform
        input_format: Input format ('json', 'dict', 'list')
        output_format: Output format ('json', 'csv', 'xml')
        pretty_print: Whether to pretty print the output (default: False)
    
    Returns:
        Transformed data
    """
    data = parameters.get("data", {})
    input_format = parameters.get("input_format", "dict")
    output_format = parameters.get("output_format", "json")
    pretty_print = parameters.get("pretty_print", False)
    
    # Parse input based on format
    if input_format == "json":
        try:
            if isinstance(data, str):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input"}
    elif input_format == "dict":
        parsed_data = data
    elif input_format == "list":
        parsed_data = data
    else:
        return {"error": f"Unsupported input format: {input_format}"}
    
    # Transform to output format
    if output_format == "json":
        if pretty_print:
            result = json.dumps(parsed_data, indent=2, ensure_ascii=False)
        else:
            result = json.dumps(parsed_data, ensure_ascii=False)
    elif output_format == "csv":
        if isinstance(parsed_data, list) and parsed_data:
            import csv
            import io
            output = io.StringIO()
            if isinstance(parsed_data[0], dict):
                writer = csv.DictWriter(output, fieldnames=parsed_data[0].keys())
                writer.writeheader()
                writer.writerows(parsed_data)
            else:
                writer = csv.writer(output)
                for row in parsed_data:
                    writer.writerow(row)
            result = output.getvalue()
        else:
            result = "No data to convert to CSV"
    elif output_format == "xml":
        # Simple XML conversion
        if isinstance(parsed_data, dict):
            xml_parts = ["<root>"]
            for key, value in parsed_data.items():
                xml_parts.append(f"<{key}>{value}</{key}>")
            xml_parts.append("</root>")
            result = "\n".join(xml_parts)
        else:
            result = f"<data>{str(parsed_data)}</data>"
    else:
        return {"error": f"Unsupported output format: {output_format}"}
    
    return {"transformed_data": result, "original_format": input_format, "target_format": output_format}


@tool(
    name="conversation_summarizer",
    description="Summarize conversation content",
    category=ToolCategory.ANALYSIS,
    tags=["conversation", "summary", "nlp"],
    timeout=20.0
)
async def conversation_summarizer(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize conversation content.
    
    Parameters:
        messages: List of messages to summarize
        max_length: Maximum length of summary (default: 200)
        style: Summary style ('brief', 'detailed', 'bullets') (default: 'brief')
    
    Returns:
        Conversation summary
    """
    messages = parameters.get("messages", [])
    max_length = parameters.get("max_length", 200)
    style = parameters.get("style", "brief")
    
    if not messages:
        return {"summary": "No messages to summarize", "message_count": 0}
    
    # Extract user and assistant messages
    user_messages = [msg for msg in messages if msg.get("role") == "user"]
    assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
    
    # Create summary based on style
    if style == "brief":
        summary = f"Conversation with {len(user_messages)} user messages and {len(assistant_messages)} assistant responses."
    elif style == "detailed":
        user_topics = set()
        for msg in user_messages:
            content = msg.get("content", "").lower()
            if "python" in content:
                user_topics.add("Python")
            if "help" in content:
                user_topics.add("help")
            if "question" in content:
                user_topics.add("questions")
        
        summary = f"Detailed conversation covering topics: {', '.join(user_topics) if user_topics else 'general discussion'}. "
        summary += f"Contains {len(user_messages)} user inputs and {len(assistant_messages)} assistant responses."
    elif style == "bullets":
        summary = "• Conversation summary:\n"
        summary += f"  • {len(user_messages)} user messages\n"
        summary += f"  • {len(assistant_messages)} assistant responses\n"
        if user_messages:
            summary += "  • Topics discussed: general conversation"
    else:
        summary = "Summary with unknown style"
    
    # Truncate if too long
    if len(summary) > max_length:
        summary = summary[:max_length-3] + "..."
    
    return {
        "summary": summary,
        "message_count": len(messages),
        "user_message_count": len(user_messages),
        "assistant_message_count": len(assistant_messages),
        "style": style
    }


@tool(
    name="code_executor",
    description="Safely execute Python code (for demonstration purposes)",
    category=ToolCategory.SYSTEM,
    tags=["code", "python", "execution"],
    timeout=30.0,
    permission=ToolPermission.RESTRICTED
)
async def code_executor(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely execute Python code in a restricted environment.
    
    Parameters:
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 10)
    
    Returns:
        Execution result
    """
    code = parameters.get("code", "")
    timeout = parameters.get("timeout", 10)
    
    # Security check - only allow safe operations
    forbidden_keywords = ["import", "exec", "eval", "open", "file", "os", "sys", "subprocess"]
    for keyword in forbidden_keywords:
        if keyword in code.lower():
            return {"error": f"Code contains forbidden keyword: {keyword}"}
    
    try:
        # Create a safe execution environment
        safe_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round,
            },
            "math": __import__("math"),
            "datetime": datetime,
            "result": None
        }
        
        # Execute the code
        exec(code, safe_globals)
        
        return {
            "success": True,
            "result": safe_globals.get("result", "Code executed successfully"),
            "executed_code": code
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "executed_code": code
        }


@tool(
    name="web_search_simulator",
    description="Simulate web search functionality",
    category=ToolCategory.COMMUNICATION,
    tags=["search", "web", "simulation"],
    timeout=5.0
)
async def web_search_simulator(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate web search for demonstration purposes.
    
    Parameters:
        query: Search query
        max_results: Maximum number of results (default: 5)
    
    Returns:
        Simulated search results
    """
    query = parameters.get("query", "")
    max_results = parameters.get("max_results", 5)
    
    # Simulate search results based on query
    query_lower = query.lower()
    
    # Mock search results
    mock_results = [
        {
            "title": f"Search result for '{query}' - Result 1",
            "url": "https://example.com/result1",
            "snippet": f"This is a mock search result for the query: {query}. It contains relevant information about the topic."
        },
        {
            "title": f"Information about '{query}' - Result 2",
            "url": "https://example.com/result2",
            "snippet": f"Another mock result discussing {query} and related concepts in detail."
        },
        {
            "title": f"Learn more about '{query}' - Result 3",
            "url": "https://example.com/result3",
            "snippet": f"Comprehensive guide and resources about {query} and best practices."
        }
    ]
    
    # Filter results based on query content
    if "python" in query_lower:
        mock_results[0]["title"] = "Python Programming - Official Documentation"
        mock_results[0]["url"] = "https://docs.python.org/"
        mock_results[0]["snippet"] = "The official Python documentation with tutorials, library references, and more."
    
    if "machine learning" in query_lower:
        mock_results[1]["title"] = "Machine Learning Tutorial - Complete Guide"
        mock_results[1]["url"] = "https://example.com/ml-tutorial"
        mock_results[1]["snippet"] = "A comprehensive tutorial covering machine learning concepts and implementations."
    
    return {
        "query": query,
        "results": mock_results[:max_results],
        "total_results": len(mock_results)
    }


async def basic_tool_example():
    """Demonstrate basic tool usage."""
    print("=== Basic Tool Usage Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=1000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Register tools
            print("\n1. Registering tools")
            print("-" * 30)
            
            # Register tools using the tool instances created by the decorator
            context_manager.tool_manager.registry.register_tool(text_analyzer._tool_instance)
            context_manager.tool_manager.registry.register_tool(data_transformer._tool_instance)
            context_manager.tool_manager.registry.register_tool(conversation_summarizer._tool_instance)
            
            # List registered tools
            tools = context_manager.tool_manager.list_tools()
            print(f"Registered {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool['description']}")
            
            # Execute text analyzer
            print("\n2. Text analyzer tool")
            print("-" * 30)
            
            result = await context_manager.execute_tool(
                "text_analyzer",
                text="Hello world! This is a test message for text analysis.",
                include_word_count=True,
                include_sentiment=True
            )
            
            print(f"Tool execution success: {result.success}")
            if result.success:
                print(f"Analysis result: {result.result}")
            
            # Execute data transformer
            print("\n3. Data transformer tool")
            print("-" * 30)
            
            test_data = {"name": "John", "age": 30, "city": "New York"}
            result = await context_manager.execute_tool(
                "data_transformer",
                data=test_data,
                input_format="dict",
                output_format="json",
                pretty_print=True
            )
            
            print(f"Tool execution success: {result.success}")
            if result.success:
                print(f"Transformed data: {result.result['transformed_data']}")
            
        finally:
            await context_manager.cleanup()
    
    print("=== Basic Tool Usage Example Complete ===")


async def tool_chaining_example():
    """Demonstrate tool chaining and composition."""
    print("\n=== Tool Chaining Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=2000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Register tools using the tool instances created by the decorator
            context_manager.tool_manager.registry.register_tool(text_analyzer._tool_instance)
            context_manager.tool_manager.registry.register_tool(web_search_simulator._tool_instance)
            context_manager.tool_manager.registry.register_tool(conversation_summarizer._tool_instance)
            
            # Add some conversation messages
            messages = [
                {"role": "user", "content": "I need help with Python programming"},
                {"role": "assistant", "content": "I can help you with Python programming. What specific topic do you need help with?"},
                {"role": "user", "content": "I'm trying to understand decorators and generators"},
                {"role": "assistant", "content": "Decorators and generators are advanced Python features. Decorators modify function behavior, while generators create iterators."}
            ]
            
            for msg in messages:
                await context_manager.add_message(msg["role"], msg["content"])
            
            # Chain 1: Analyze conversation, then summarize
            print("\n1. Conversation analysis chain")
            print("-" * 40)
            
            # Get conversation from context
            conversation_data = [
                {"role": msg.role, "content": msg.content}
                for msg in context_manager.context_window.messages
            ]
            
            # Step 1: Summarize conversation
            summary_result = await context_manager.execute_tool(
                "conversation_summarizer",
                messages=conversation_data,
                style="detailed"
            )
            
            print(f"Summary result: {summary_result.success}")
            if summary_result.success:
                print(f"Summary: {summary_result.result['summary']}")
                
                # Step 2: Analyze the summary text
                analysis_result = await context_manager.execute_tool(
                    "text_analyzer",
                    text=summary_result.result['summary'],
                    include_word_count=True,
                    include_sentiment=True
                )
                
                print(f"Analysis of summary: {analysis_result.success}")
                if analysis_result.success:
                    print(f"Word count: {analysis_result.result.get('word_count', 0)}")
                    print(f"Sentiment: {analysis_result.result.get('sentiment', {}).get('overall', 'neutral')}")
            
            # Chain 2: Search and analyze
            print("\n2. Search and analysis chain")
            print("-" * 40)
            
            # Step 1: Simulate web search
            search_result = await context_manager.execute_tool(
                "web_search_simulator",
                query="Python decorators best practices",
                max_results=3
            )
            
            print(f"Search result: {search_result.success}")
            if search_result.success:
                # Extract search result snippets
                snippets = [result['snippet'] for result in search_result.result['results']]
                
                # Step 2: Analyze search results
                combined_text = " ".join(snippets)
                analysis_result = await context_manager.execute_tool(
                    "text_analyzer",
                    text=combined_text,
                    include_word_count=True
                )
                
                print(f"Analysis of search results: {analysis_result.success}")
                if analysis_result.success:
                    print(f"Total words in search results: {analysis_result.result.get('word_count', 0)}")
            
            # Tool composition example
            print("\n3. Tool composition example")
            print("-" * 40)
            
            # Create a composite tool that uses multiple tools
            async def analyze_and_search(query: str) -> Dict[str, Any]:
                """Search for information and analyze the results."""
                
                # Step 1: Search
                search_result = await context_manager.execute_tool(
                    "web_search_simulator",
                    query=query
                )
                
                if not search_result.success:
                    return {"error": "Search failed", "details": search_result.error}
                
                # Step 2: Extract and combine text
                snippets = [result['snippet'] for result in search_result.result['results']]
                combined_text = " ".join(snippets)
                
                # Step 3: Analyze
                analysis_result = await context_manager.execute_tool(
                    "text_analyzer",
                    text=combined_text,
                    include_word_count=True,
                    include_sentiment=True
                )
                
                return {
                    "query": query,
                    "search_results": search_result.result['results'],
                    "analysis": analysis_result.result if analysis_result.success else None,
                    "success": analysis_result.success
                }
            
            # Execute composite operation
            composite_result = await analyze_and_search("async programming in Python")
            print(f"Composite operation success: {composite_result['success']}")
            if composite_result['success']:
                print(f"Found {len(composite_result['search_results'])} search results")
                if composite_result['analysis']:
                    print(f"Analysis word count: {composite_result['analysis'].get('word_count', 0)}")
            
        finally:
            await context_manager.cleanup()
    
    print("=== Tool Chaining Example Complete ===")


async def advanced_tool_example():
    """Demonstrate advanced tool features."""
    print("\n=== Advanced Tool Features Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=2000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Register advanced tools using the tool instances created by the decorator
            context_manager.tool_manager.registry.register_tool(code_executor._tool_instance)
            context_manager.tool_manager.registry.register_tool(data_transformer._tool_instance)
            
            # Tool with error handling
            print("\n1. Tool error handling")
            print("-" * 40)
            
            # Try to execute code with forbidden keywords
            result = await context_manager.execute_tool(
                "code_executor",
                code="import os\nprint('Hello from restricted code')"
            )
            
            print(f"Code execution result: {result.success}")
            if not result.success:
                print(f"Error captured: {result.error}")
            
            # Execute safe code
            result = await context_manager.execute_tool(
                "code_executor",
                code="result = [x**2 for x in range(10)]\nprint('Generated squares:', result)"
            )
            
            print(f"Safe code execution: {result.success}")
            if result.success:
                print(f"Result: {result.result}")
            
            # Tool with timeout handling
            print("\n2. Tool timeout handling")
            print("-" * 40)
            
            # Create a slow tool
            async def slow_tool(parameters):
                await asyncio.sleep(2)  # Simulate slow operation
                return {"result": "Slow operation completed"}
            
            # Create and register the slow tool manually (not decorated)
            from context_manager.core.tool_system import ToolMetadata, FunctionTool, ToolCategory
            slow_tool_metadata = ToolMetadata(
                name="slow_tool",
                description="Slow operation tool",
                category=ToolCategory.SYSTEM,
                timeout=1.0
            )
            slow_tool_instance = FunctionTool(slow_tool, slow_tool_metadata)
            context_manager.tool_manager.registry.register_tool(slow_tool_instance)
            
            # This should timeout
            result = await context_manager.execute_tool("slow_tool")
            print(f"Slow tool result (should timeout): {result.success}")
            if not result.success:
                print(f"Timeout error: {result.error}")
            
            # Tool statistics and monitoring
            print("\n3. Tool statistics")
            print("-" * 40)
            
            stats = context_manager.tool_manager.get_tool_stats()
            registry_stats = stats.get('registry_stats', {})
            print(f"Total tools: {registry_stats.get('total_tools', 'N/A')}")
            print(f"Active tools: {registry_stats.get('active_tools', 'N/A')}")
            print(f"Total executions: {stats.get('performance_metrics', {}).get('total_executions', 'N/A')}")
            
            if registry_stats.get('top_tools'):
                print("Top tools by usage:")
                for tool_name, count in registry_stats['top_tools'][:3]:
                    print(f"  {tool_name}: {count} executions")
            
            # Tool categories
            print(f"Tool categories: {registry_stats.get('categories', 'N/A')}")
            
            # Parallel tool execution
            print("\n4. Parallel tool execution")
            print("-" * 40)
            
            # Execute multiple tools in parallel
            tool_calls = [
                {"tool_name": "text_analyzer", "parameters": {"text": "Hello world", "include_word_count": True}},
                {"tool_name": "data_transformer", "parameters": {"data": {"test": "value"}, "input_format": "dict", "output_format": "json"}},
                {"tool_name": "web_search_simulator", "parameters": {"query": "Python programming", "max_results": 2}}
            ]
            
            # Register missing tools using the tool instances created by the decorator
            context_manager.tool_manager.registry.register_tool(text_analyzer._tool_instance)
            context_manager.tool_manager.registry.register_tool(web_search_simulator._tool_instance)
            
            # Execute tools in parallel
            results = await context_manager.tool_manager.executor.execute_tools_parallel(tool_calls)
            
            print(f"Parallel execution completed: {len(results)} results")
            for i, result in enumerate(results):
                print(f"  Tool {i+1}: {'Success' if result.success else 'Failed'}")
                if result.success:
                    print(f"    Tool: {result.tool_name}")
            
        finally:
            await context_manager.cleanup()
    
    print("=== Advanced Tool Features Example Complete ===")


async def custom_tool_creation_example():
    """Demonstrate how to create custom tools."""
    print("\n=== Custom Tool Creation Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ContextManagerConfig(
            max_tokens=1000,
            cache_path=temp_dir
        )
        
        context_manager = EnhancedContextManager(config=config)
        await context_manager.initialize()
        
        try:
            # Example 1: Simple calculator tool
            print("\n1. Creating a calculator tool")
            print("-" * 40)
            
            @tool(
                name="calculator",
                description="Perform mathematical calculations",
                category=ToolCategory.UTILITY,
                tags=["math", "calculator", "calculation"]
            )
            async def calculator(parameters: Dict[str, Any]) -> Dict[str, Any]:
                """Perform basic mathematical operations."""
                operation = parameters.get("operation", "add")
                a = parameters.get("a", 0)
                b = parameters.get("b", 0)
                
                if operation == "add":
                    result = a + b
                elif operation == "subtract":
                    result = a - b
                elif operation == "multiply":
                    result = a * b
                elif operation == "divide":
                    if b == 0:
                        return {"error": "Division by zero"}
                    result = a / b
                else:
                    return {"error": f"Unknown operation: {operation}"}
                
                return {
                    "operation": operation,
                    "operands": [a, b],
                    "result": result
                }
            
            # Register the calculator tool using the tool instance created by the decorator
            context_manager.tool_manager.registry.register_tool(calculator._tool_instance)
            
            # Test the calculator
            result = await context_manager.execute_tool(
                "calculator",
                operation="add",
                a=10,
                b=5
            )
            
            print(f"Calculator result: {result.success}")
            if result.success:
                print(f"10 + 5 = {result.result['result']}")
            
            # Example 2: Data validation tool
            print("\n2. Creating a data validation tool")
            print("-" * 40)
            
            @tool(
                name="data_validator",
                description="Validate data against schema",
                category=ToolCategory.DATA_PROCESSING,
                tags=["validation", "schema", "data"]
            )
            async def data_validator(parameters: Dict[str, Any]) -> Dict[str, Any]:
                """Validate data against a simple schema."""
                data = parameters.get("data", {})
                schema = parameters.get("schema", {})
                
                errors = []
                validated_data = {}
                
                for field, field_schema in schema.items():
                    field_type = field_schema.get("type")
                    required = field_schema.get("required", False)
                    
                    if required and field not in data:
                        errors.append(f"Required field '{field}' is missing")
                        continue
                    
                    if field in data:
                        value = data[field]
                        
                        # Type validation
                        if field_type == "string" and not isinstance(value, str):
                            errors.append(f"Field '{field}' must be a string")
                        elif field_type == "number" and not isinstance(value, (int, float)):
                            errors.append(f"Field '{field}' must be a number")
                        elif field_type == "boolean" and not isinstance(value, bool):
                            errors.append(f"Field '{field}' must be a boolean")
                        
                        validated_data[field] = value
                
                return {
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "validated_data": validated_data
                }
            
            # Register the data validator using the tool instance created by the decorator
            context_manager.tool_manager.registry.register_tool(data_validator._tool_instance)
            
            # Test the data validator
            test_schema = {
                "name": {"type": "string", "required": True},
                "age": {"type": "number", "required": True},
                "email": {"type": "string", "required": False}
            }
            
            test_data = {
                "name": "John Doe",
                "age": 30,
                "email": "john@example.com"
            }
            
            result = await context_manager.execute_tool(
                "data_validator",
                data=test_data,
                schema=test_schema
            )
            
            print(f"Data validation result: {result.success}")
            if result.success:
                print(f"Valid: {result.result['valid']}")
                if result.result['errors']:
                    print(f"Errors: {result.result['errors']}")
                else:
                    print("Data is valid!")
            
            # Example 3: File processing tool
            print("\n3. Creating a file processing tool")
            print("-" * 40)
            
            @tool(
                name="file_processor",
                description="Process file content",
                category=ToolCategory.UTILITY,
                tags=["file", "processing", "text"],
                permission=ToolPermission.RESTRICTED
            )
            async def file_processor(parameters: Dict[str, Any]) -> Dict[str, Any]:
                """Process file content (simulated)."""
                filename = parameters.get("filename", "")
                operation = parameters.get("operation", "read")
                content = parameters.get("content", "")
                
                # Simulate file operations
                if operation == "read":
                    # Return simulated file content
                    return {
                        "filename": filename,
                        "content": f"Simulated content of {filename}\nThis is a test file content.",
                        "size": len(f"Simulated content of {filename}\nThis is a test file content."),
                        "operation": "read"
                    }
                elif operation == "write":
                    # Simulate writing content
                    return {
                        "filename": filename,
                        "content": content,
                        "size": len(content),
                        "operation": "write",
                        "status": "written"
                    }
                elif operation == "info":
                    # Simulate file info
                    return {
                        "filename": filename,
                        "size": 1024,
                        "modified": datetime.now().isoformat(),
                        "operation": "info"
                    }
                else:
                    return {"error": f"Unknown operation: {operation}"}
            
            # Register the file processor using the tool instance created by the decorator
            context_manager.tool_manager.registry.register_tool(file_processor._tool_instance)
            
            # Test the file processor
            result = await context_manager.execute_tool(
                "file_processor",
                filename="test.txt",
                operation="read"
            )
            
            print(f"File processing result: {result.success}")
            if result.success:
                print(f"Filename: {result.result['filename']}")
                print(f"Content preview: {result.result['content'][:50]}...")
            
        finally:
            await context_manager.cleanup()
    
    print("=== Custom Tool Creation Example Complete ===")


async def main():
    """Run all custom tool examples."""
    print("Python Context Manager - Custom Tools Examples")
    print("=" * 60)
    
    await basic_tool_example()
    await tool_chaining_example()
    await advanced_tool_example()
    await custom_tool_creation_example()
    
    print("\n" + "=" * 60)
    print("All custom tool examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())