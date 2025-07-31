# Tool System Documentation

## Overview

The Tool System provides a comprehensive framework for managing and executing tools within the context manager. It supports tool registration, execution coordination, performance monitoring, and security features.

## Features

### 1. Tool Management
- **Tool Registration**: Dynamic registration of tools with metadata
- **Tool Discovery**: Search and filtering by category, tags, and keywords
- **Tool Lifecycle**: Activation, deactivation, and removal of tools
- **Tool Dependencies**: Support for tool dependencies and chains

### 2. Tool Execution
- **Synchronous & Asynchronous**: Support for both sync and async tool functions
- **Parallel Execution**: Execute multiple tools concurrently
- **Tool Chains**: Execute tools in sequence with result passing
- **Timeout Handling**: Configurable timeouts for tool execution
- **Error Handling**: Comprehensive error handling and recovery

### 3. Performance Monitoring
- **Execution Statistics**: Track success rates, execution times, and usage patterns
- **Rate Limiting**: Configurable rate limits for tool usage
- **Health Monitoring**: Real-time health checks and status reporting
- **Execution History**: Detailed history of tool executions

### 4. Security & Validation
- **Parameter Validation**: Automatic parameter validation before execution
- **Permission Levels**: Public, restricted, and admin permission levels
- **Safe Execution**: Secure execution environment with restricted capabilities
- **Input/Output Validation**: Schema-based validation for tool inputs and outputs

## Architecture

### Core Components

#### ToolInterface
Abstract base class for all tools. Defines the contract for tool implementation.

```python
class ToolInterface(ABC):
    @abstractmethod
    async def execute(self, context: ToolExecutionContext) -> ToolResult:
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        pass
```

#### FunctionTool
Concrete implementation that wraps Python functions as tools.

#### ToolRegistry
Manages tool registration, discovery, and metadata.

#### ToolExecutor
Handles tool execution with various strategies (parallel, chained, etc.).

#### ToolManager
Main coordination class that integrates all tool system components.

### Data Models

#### ToolMetadata
Contains comprehensive metadata about tools:
- Basic info: name, description, category, version
- Execution info: timeout, retries, rate limits
- Security info: permissions, dependencies
- Statistics: execution counts, success rates

#### ToolExecutionContext
Context for tool execution:
- Execution ID and parameters
- User and session information
- Context window and memory manager access
- Start time and metadata

#### ToolResult
Standardized result format:
- Success/failure status
- Result data or error message
- Execution time and metadata
- Tool name and parameters used

## Usage Examples

### Basic Tool Registration

```python
from context_manager.core.tool_system import tool, ToolCategory

@tool(
    name="weather_tool",
    description="Get weather information for a location",
    category=ToolCategory.UTILITY,
    timeout=30.0,
    tags=["weather", "location"]
)
async def get_weather(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather information."""
    location = parameters.get("location")
    # Weather API logic here
    return {
        "location": location,
        "temperature": 72,
        "conditions": "Sunny"
    }
```

### Tool Execution

```python
from context_manager.core.tool_manager import ToolManager, ToolExecutionRequest

# Initialize tool manager
config = ContextManagerConfig()
tool_manager = ToolManager(config)

# Create execution request
request = ToolExecutionRequest(
    tool_name="weather_tool",
    parameters={"location": "New York"},
    user_id="user123"
)

# Execute tool
response = await tool_manager.execute_tool(request)

if response.success:
    print(f"Weather: {response.result}")
else:
    print(f"Error: {response.error}")
```

### Parallel Tool Execution

```python
# Execute multiple tools in parallel
requests = [
    ToolExecutionRequest(
        tool_name="weather_tool",
        parameters={"location": "New York"}
    ),
    ToolExecutionRequest(
        tool_name="weather_tool",
        parameters={"location": "London"}
    )
]

responses = await tool_manager.execute_tools_parallel(requests)

for response in responses:
    if response.success:
        print(f"Weather result: {response.result}")
```

### Tool Chain Execution

```python
# Execute tools in sequence with result passing
requests = [
    ToolExecutionRequest(
        tool_name="calculate",
        parameters={"expression": "10 + 5"}
    ),
    ToolExecutionRequest(
        tool_name="format_text",
        parameters={"text": "Result: {result}", "operation": "format"}
    )
]

result = await tool_manager.execute_tool_chain(requests)
print(f"Final result: {result['final_context']}")
```

### Custom Tool Implementation

```python
from context_manager.core.tool_system import ToolInterface, ToolMetadata, ToolCategory

class CustomTool(ToolInterface):
    def __init__(self):
        metadata = ToolMetadata(
            name="custom_tool",
            description="Custom tool implementation",
            category=ToolCategory.CUSTOM,
            timeout=10.0
        )
        super().__init__(metadata)
    
    async def execute(self, context: ToolExecutionContext) -> ToolResult:
        # Custom tool logic
        try:
            result = await self._process_data(context.parameters)
            return ToolResult(
                success=True,
                result=result,
                tool_name=self.metadata.name,
                parameters=context.parameters
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.metadata.name,
                parameters=context.parameters
            )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        # Validate parameters
        return "required_param" in parameters
    
    async def _process_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        # Process data logic
        return {"processed": True, "data": parameters}
```

## Built-in Tools

The system includes several built-in tools:

### Echo Tool
- **Name**: `echo`
- **Description**: Echo back the input parameters
- **Category**: Utility
- **Usage**: Test and debugging

### Calculate Tool
- **Name**: `calculate`
- **Description**: Perform mathematical calculations
- **Category**: Utility
- **Usage**: Safe mathematical expression evaluation

### Format Text Tool
- **Name**: `format_text`
- **Description**: Format and manipulate text
- **Category**: Utility
- **Usage**: Text transformation operations

### System Info Tool
- **Name**: `get_system_info`
- **Description**: Get system information
- **Category**: System
- **Usage**: System monitoring and diagnostics

## Configuration

### Tool System Configuration
```python
# In ContextManagerConfig
tool_timeout: int = 30  # Default tool timeout
max_concurrent_tools: int = 10  # Maximum concurrent tool executions
max_workers: int = 5  # Number of worker threads
```

### Tool-Specific Configuration
```python
@tool(
    name="my_tool",
    description="My custom tool",
    category=ToolCategory.CUSTOM,
    timeout=60.0,  # Custom timeout
    max_retries=3,  # Number of retry attempts
    rate_limit=10,  # Max calls per time window
    rate_window=60.0,  # Time window in seconds
    dependencies=["tool1", "tool2"],  # Required tools
    permission=ToolPermission.RESTRICTED  # Permission level
)
async def my_tool(parameters: Dict[str, Any]) -> Dict[str, Any]:
    pass
```

## Performance Monitoring

### Tool Statistics
```python
# Get comprehensive tool statistics
stats = tool_manager.get_tool_stats()

print(f"Total executions: {stats['performance_metrics']['total_executions']}")
print(f"Success rate: {stats['performance_metrics']['success_rate']:.2%}")
print(f"Average execution time: {stats['performance_metrics']['average_execution_time']:.3f}s")

# Top tools by usage
for tool_name, count in stats['registry_stats']['top_tools']:
    print(f"{tool_name}: {count} executions")
```

### Execution History
```python
# Get execution history
history = tool_manager.get_execution_history(limit=50)

for record in history:
    print(f"{record['timestamp']}: {record['tool_name']} - {record['result']['success']}")

# Get history for specific tool
tool_history = tool_manager.get_execution_history("weather_tool", limit=10)
```

### Health Monitoring
```python
# System health check
health = tool_manager.health_check()

print(f"Status: {health['status']}")
print(f"Registered tools: {health['registered_tools']}")
print(f"Active executions: {health['active_executions']}")
print(f"Success rate: {health['performance']['success_rate']:.2%}")
```

## Security Features

### Parameter Validation
```python
def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
    # Custom validation logic
    required_fields = ["api_key", "endpoint"]
    return all(field in parameters for field in required_fields)
```

### Permission Levels
- **Public**: Available to all users
- **Restricted**: Requires specific permissions
- **Admin**: Only available to administrators

### Safe Execution
- Restricted execution environment
- Input/output sanitization
- Timeout protection
- Resource limits

## Best Practices

### Tool Design
1. **Keep tools focused**: Each tool should have a single responsibility
2. **Use descriptive names**: Clear, descriptive tool names and descriptions
3. **Validate inputs**: Always validate parameters before execution
4. **Handle errors gracefully**: Provide meaningful error messages
5. **Document usage**: Include examples and usage instructions

### Performance Optimization
1. **Set appropriate timeouts**: Balance responsiveness and completion
2. **Use rate limiting**: Prevent abuse and resource exhaustion
3. **Monitor performance**: Track execution times and success rates
4. **Optimize frequently used tools**: Profile and optimize critical tools
5. **Use parallel execution**: Execute independent tools concurrently

### Security Considerations
1. **Validate all inputs**: Never trust external input
2. **Use appropriate permissions**: Restrict access to sensitive tools
3. **Sanitize outputs**: Prevent information leakage
4. **Monitor usage**: Track tool usage for security auditing
5. **Update regularly**: Keep tools and dependencies updated

## Error Handling

### Common Error Scenarios
```python
try:
    response = await tool_manager.execute_tool(request)
    if not response.success:
        if response.status == ToolStatus.TIMEOUT:
            print("Tool execution timed out")
        elif response.status == ToolStatus.FAILED:
            print(f"Tool failed: {response.error}")
        else:
            print(f"Unknown error: {response.error}")
except Exception as e:
    print(f"System error: {e}")
```

### Tool Error Handling
```python
async def execute(self, context: ToolExecutionContext) -> ToolResult:
    try:
        result = await self._do_work(context.parameters)
        return ToolResult(
            success=True,
            result=result,
            tool_name=self.metadata.name,
            parameters=context.parameters
        )
    except ValueError as e:
        return ToolResult(
            success=False,
            error=f"Invalid parameters: {e}",
            tool_name=self.metadata.name,
            parameters=context.parameters,
            status=ToolStatus.FAILED
        )
    except Exception as e:
        logger.error(f"Unexpected error in {self.metadata.name}: {e}")
        return ToolResult(
            success=False,
            error="Internal tool error",
            tool_name=self.metadata.name,
            parameters=context.parameters,
            status=ToolStatus.FAILED
        )
```

## Integration with Context Manager

The Tool System integrates seamlessly with the Context Manager:

### Memory Integration
```python
# Tools can access memory manager through context
async def execute(self, context: ToolExecutionContext) -> ToolResult:
    if context.memory_manager:
        # Retrieve relevant memories
        memories = context.memory_manager.search_memories(
            query=context.parameters.get("query", "")
        )
        # Use memories in tool execution
        result = self._process_with_memory(context.parameters, memories)
```

### Context Window Integration
```python
# Tools can access and modify context window
async def execute(self, context: ToolExecutionContext) -> ToolResult:
    if context.context_window:
        # Add tool result to context
        context.context_window.add_message(
            role="system",
            content=f"Tool result: {result}"
        )
```

## Future Enhancements

### Planned Features
1. **Tool Marketplace**: Centralized repository for sharing tools
2. **Tool Versioning**: Support for multiple versions of tools
3. **Tool Dependencies**: Advanced dependency management
4. **Tool Analytics**: Enhanced analytics and insights
5. **Tool Testing**: Automated testing framework for tools
6. **Tool Documentation**: Auto-generated documentation

### Performance Improvements
1. **Tool Caching**: Cache results of expensive operations
2. **Tool Pooling**: Reuse tool instances for better performance
3. **Async Optimization**: Improved async execution patterns
4. **Resource Management**: Better resource allocation and cleanup

### Security Enhancements
1. **Tool Sandboxing**: Isolated execution environments
2. **Tool Signing**: Cryptographic tool verification
3. **Access Controls**: Fine-grained access control
4. **Audit Logging**: Comprehensive audit trails