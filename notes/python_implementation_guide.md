# Python实现指南

## 概述

本指南基于对Claude Code架构的深入分析，为Python实现上下文管理和记忆管理系统提供详细的技术指导。

## 项目结构设计

### 推荐的项目结构

```
python_context_manager/
├── src/
│   ├── context_manager/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── context_manager.py
│   │   │   ├── memory_manager.py
│   │   │   └── state_controller.py
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   ├── short_term.py
│   │   │   ├── medium_term.py
│   │   │   ├── long_term.py
│   │   │   └── memory_index.py
│   │   ├── compression/
│   │   │   ├── __init__.py
│   │   │   ├── token_manager.py
│   │   │   ├── context_compressor.py
│   │   │   └── priority_manager.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── tool_executor.py
│   │   │   ├── tool_registry.py
│   │   │   └── execution_context.py
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── cache_layer.py
│   │   │   ├── memory_layer.py
│   │   │   ├── disk_layer.py
│   │   │   └── archive_layer.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── logging.py
│   │       └── error_handling.py
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_context_manager.py
│   │   ├── test_memory_manager.py
│   │   ├── test_compression.py
│   │   └── test_tools.py
│   ├── examples/
│   │   ├── basic_usage.py
│   │   ├── advanced_features.py
│   │   └── performance_demo.py
│   └── docs/
│       ├── api_reference.md
│       ├── user_guide.md
│       └── developer_guide.md
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## 核心组件实现

### 1. 上下文管理器实现

```python
# src/context_manager/core/context_manager.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
import time

class ContextPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Message:
    role: str
    content: str
    timestamp: float
    priority: ContextPriority = ContextPriority.MEDIUM
    metadata: Dict[str, Any] = None

@dataclass
class ContextWindow:
    messages: List[Message]
    max_tokens: int
    current_tokens: int
    compression_ratio: float = 0.8

class ContextManager:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.context_window = ContextWindow(
            messages=[], max_tokens=max_tokens, current_tokens=0
        )
        self.compression_engine = ContextCompressor()
        self.priority_manager = PriorityManager()
        self.token_manager = TokenManager()
        
    async def add_message(self, message: Message) -> bool:
        """添加消息到上下文窗口"""
        token_count = await self.token_manager.count_tokens(message.content)
        
        if self.context_window.current_tokens + token_count > self.max_tokens:
            await self._compress_context()
            
        if self.context_window.current_tokens + token_count > self.max_tokens:
            await self._evict_messages()
            
        self.context_window.messages.append(message)
        self.context_window.current_tokens += token_count
        return True
        
    async def _compress_context(self) -> None:
        """压缩上下文内容"""
        if len(self.context_window.messages) < 2:
            return
            
        compressed = await self.compression_engine.compress_messages(
            self.context_window.messages
        )
        
        token_count = await self.token_manager.count_tokens(
            json.dumps(compressed)
        )
        
        self.context_window.messages = compressed
        self.context_window.current_tokens = token_count
        
    async def _evict_messages(self) -> None:
        """基于优先级淘汰消息"""
        sorted_messages = self.priority_manager.sort_by_priority(
            self.context_window.messages
        )
        
        while (self.context_window.current_tokens > 
               self.max_tokens * self.context_window.compression_ratio and
               sorted_messages):
            
            message = sorted_messages.pop(0)
            self.context_window.messages.remove(message)
            self.context_window.current_tokens -= await self.token_manager.count_tokens(
                message.content
            )
            
    async def get_context(self) -> List[Dict[str, Any]]:
        """获取当前上下文"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "priority": msg.priority.value
            }
            for msg in self.context_window.messages
        ]
```

### 2. 记忆管理器实现

```python
# src/context_manager/memory/memory_manager.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pickle
import asyncio
from abc import ABC, abstractmethod

@dataclass
class Memory:
    content: str
    timestamp: datetime
    importance: float
    tags: List[str]
    metadata: Dict[str, Any]

class MemoryLayer(ABC):
    @abstractmethod
    async def store(self, memory: Memory) -> bool:
        pass
    
    @abstractmethod
    async def retrieve(self, query: str, limit: int = 10) -> List[Memory]:
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        pass

class ShortTermMemory(MemoryLayer):
    def __init__(self, max_items: int = 100):
        self.max_items = max_items
        self.memories: List[Memory] = []
        
    async def store(self, memory: Memory) -> bool:
        self.memories.append(memory)
        if len(self.memories) > self.max_items:
            self.memories.pop(0)
        return True
        
    async def retrieve(self, query: str, limit: int = 10) -> List[Memory]:
        # 简单的字符串匹配检索
        relevant_memories = []
        for memory in self.memories:
            if query.lower() in memory.content.lower():
                relevant_memories.append(memory)
                
        return sorted(relevant_memories, 
                     key=lambda x: x.importance, 
                     reverse=True)[:limit]
                     
    async def clear(self) -> bool:
        self.memories.clear()
        return True

class MediumTermMemory(MemoryLayer):
    def __init__(self, storage_path: str = "medium_term_memory.pkl"):
        self.storage_path = storage_path
        self.memories: List[Memory] = []
        self._load_memories()
        
    def _load_memories(self):
        try:
            with open(self.storage_path, 'rb') as f:
                self.memories = pickle.load(f)
        except FileNotFoundError:
            self.memories = []
            
    def _save_memories(self):
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.memories, f)
            
    async def store(self, memory: Memory) -> bool:
        self.memories.append(memory)
        self._save_memories()
        return True
        
    async def retrieve(self, query: str, limit: int = 10) -> List[Memory]:
        relevant_memories = []
        for memory in self.memories:
            if query.lower() in memory.content.lower():
                relevant_memories.append(memory)
                
        return sorted(relevant_memories, 
                     key=lambda x: x.importance, 
                     reverse=True)[:limit]
                     
    async def clear(self) -> bool:
        self.memories.clear()
        self._save_memories()
        return True

class LongTermMemory(MemoryLayer):
    def __init__(self, storage_path: str = "long_term_memory.json"):
        self.storage_path = storage_path
        self.memories: List[Memory] = []
        self._load_memories()
        
    def _load_memories(self):
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.memories = [
                    Memory(
                        content=item['content'],
                        timestamp=datetime.fromisoformat(item['timestamp']),
                        importance=item['importance'],
                        tags=item['tags'],
                        metadata=item['metadata']
                    )
                    for item in data
                ]
        except FileNotFoundError:
            self.memories = []
            
    def _save_memories(self):
        data = [
            {
                'content': memory.content,
                'timestamp': memory.timestamp.isoformat(),
                'importance': memory.importance,
                'tags': memory.tags,
                'metadata': memory.metadata
            }
            for memory in self.memories
        ]
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    async def store(self, memory: Memory) -> bool:
        self.memories.append(memory)
        self._save_memories()
        return True
        
    async def retrieve(self, query: str, limit: int = 10) -> List[Memory]:
        relevant_memories = []
        for memory in self.memories:
            if query.lower() in memory.content.lower():
                relevant_memories.append(memory)
                
        return sorted(relevant_memories, 
                     key=lambda x: x.importance, 
                     reverse=True)[:limit]
                     
    async def clear(self) -> bool:
        self.memories.clear()
        self._save_memories()
        return True

class MemoryManager:
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.medium_term = MediumTermMemory()
        self.long_term = LongTermMemory()
        self.memory_index = MemoryIndex()
        
    async def store_memory(self, content: str, importance: float = 0.5, 
                          tags: List[str] = None, metadata: Dict[str, Any] = None) -> bool:
        """存储记忆到合适的层级"""
        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # 根据重要性决定存储层级
        if importance >= 0.8:
            await self.long_term.store(memory)
        elif importance >= 0.5:
            await self.medium_term.store(memory)
        else:
            await self.short_term.store(memory)
            
        await self.memory_index.add_to_index(memory)
        return True
        
    async def retrieve_memory(self, query: str, limit: int = 10) -> List[Memory]:
        """跨层检索记忆"""
        results = []
        
        # 从短期记忆检索
        results.extend(await self.short_term.retrieve(query, limit // 3))
        
        # 从中期记忆检索
        results.extend(await self.medium_term.retrieve(query, limit // 3))
        
        # 从长期记忆检索
        results.extend(await self.long_term.retrieve(query, limit // 3))
        
        # 去重并排序
        unique_results = {}
        for memory in results:
            key = memory.content + str(memory.timestamp)
            if key not in unique_results:
                unique_results[key] = memory
                
        return sorted(unique_results.values(), 
                     key=lambda x: x.importance, 
                     reverse=True)[:limit]
                     
    async def consolidate_memories(self) -> None:
        """定期整合记忆"""
        # 将短期记忆中的重要内容迁移到中期记忆
        short_term_memories = await self.short_term.retrieve("", 100)
        for memory in short_term_memories:
            if memory.importance >= 0.5:
                await self.medium_term.store(memory)
                
        # 将中期记忆中的重要内容迁移到长期记忆
        medium_term_memories = await self.medium_term.retrieve("", 100)
        for memory in medium_term_memories:
            if memory.importance >= 0.8:
                await self.long_term.store(memory)
```

### 3. 压缩引擎实现

```python
# src/context_manager/compression/context_compressor.py
from typing import List, Dict, Any
import re
import json
from dataclasses import dataclass
import asyncio

@dataclass
class CompressionResult:
    compressed_content: str
    compression_ratio: float
    original_size: int
    compressed_size: int

class ContextCompressor:
    def __init__(self):
        self.compression_strategies = [
            self._remove_redundancy,
            self._summarize_content,
            self._remove_low_priority,
            self._merge_similar
        ]
        
    async def compress_messages(self, messages: List[Message]) -> List[Message]:
        """压缩消息列表"""
        if len(messages) <= 2:
            return messages
            
        compressed = messages.copy()
        
        for strategy in self.compression_strategies:
            compressed = await strategy(compressed)
            
        return compressed
        
    async def _remove_redundancy(self, messages: List[Message]) -> List[Message]:
        """移除重复内容"""
        seen_content = set()
        unique_messages = []
        
        for message in messages:
            content_key = message.content.strip().lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_messages.append(message)
                
        return unique_messages
        
    async def _summarize_content(self, messages: List[Message]) -> List[Message]:
        """总结长内容"""
        summarized = []
        
        for message in messages:
            if len(message.content) > 500:  # 超过500字符的内容进行总结
                summary = self._create_summary(message.content)
                summarized_message = Message(
                    role=message.role,
                    content=summary,
                    timestamp=message.timestamp,
                    priority=message.priority,
                    metadata={**message.metadata, "original_length": len(message.content)}
                )
                summarized.append(summarized_message)
            else:
                summarized.append(message)
                
        return summarized
        
    def _create_summary(self, content: str) -> str:
        """创建内容摘要"""
        # 简单的摘要算法：取首句和尾句
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            return content
            
        return f"{sentences[0]}. [内容摘要] {sentences[-1]}."
        
    async def _remove_low_priority(self, messages: List[Message]) -> List[Message]:
        """移除低优先级内容"""
        high_priority_messages = [
            msg for msg in messages 
            if msg.priority in [ContextPriority.HIGH, ContextPriority.MEDIUM]
        ]
        
        return high_priority_messages if high_priority_messages else messages
        
    async def _merge_similar(self, messages: List[Message]) -> List[Message]:
        """合并相似内容"""
        merged = []
        i = 0
        
        while i < len(messages):
            current = messages[i]
            
            # 查找相似消息
            similar_messages = [current]
            j = i + 1
            
            while j < len(messages):
                if self._is_similar(current.content, messages[j].content):
                    similar_messages.append(messages[j])
                    j += 1
                else:
                    break
                    
            if len(similar_messages) > 1:
                # 合并相似消息
                merged_content = self._merge_messages(similar_messages)
                merged_message = Message(
                    role=current.role,
                    content=merged_content,
                    timestamp=current.timestamp,
                    priority=current.priority,
                    metadata={**current.metadata, "merged_count": len(similar_messages)}
                )
                merged.append(merged_message)
                i = j
            else:
                merged.append(current)
                i += 1
                
        return merged
        
    def _is_similar(self, content1: str, content2: str, threshold: float = 0.7) -> bool:
        """判断两个内容是否相似"""
        # 简单的相似度计算
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold
        
    def _merge_messages(self, messages: List[Message]) -> str:
        """合并多个消息的内容"""
        if len(messages) == 1:
            return messages[0].content
            
        # 提取关键信息
        all_content = " ".join([msg.content for msg in messages])
        
        # 简单的合并策略
        return f"[合并了{len(messages)}条相似消息] {all_content[:200]}..."
```

### 4. 工具执行器实现

```python
# src/context_manager/tools/tool_executor.py
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import json
import inspect

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    return_type: str
    timeout: int = 30

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        
    def register_tool(self, tool: Tool) -> bool:
        """注册工具"""
        self.tools[tool.name] = tool
        return True
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)
        
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self.tools.keys())
        
    def unregister_tool(self, name: str) -> bool:
        """注销工具"""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

class ExecutionContext:
    def __init__(self, context_manager, memory_manager):
        self.context_manager = context_manager
        self.memory_manager = memory_manager
        self.execution_id: str = ""
        self.start_time: float = 0
        self.end_time: float = 0
        self.status: str = "pending"
        self.result: Any = None
        self.error: Optional[Exception] = None
        
    async def start_execution(self) -> None:
        """开始执行"""
        self.execution_id = f"exec_{int(asyncio.get_event_loop().time())}"
        self.start_time = asyncio.get_event_loop().time()
        self.status = "running"
        
    async def complete_execution(self, result: Any) -> None:
        """完成执行"""
        self.result = result
        self.end_time = asyncio.get_event_loop().time()
        self.status = "completed"
        
    async def fail_execution(self, error: Exception) -> None:
        """执行失败"""
        self.error = error
        self.end_time = asyncio.get_event_loop().time()
        self.status = "failed"

class ToolExecutor:
    def __init__(self, context_manager, memory_manager):
        self.registry = ToolRegistry()
        self.context_manager = context_manager
        self.memory_manager = memory_manager
        self.execution_history: List[ExecutionContext] = []
        
    def register_tool(self, name: str, description: str, function: Callable, 
                     parameters: Dict[str, Any] = None, timeout: int = 30) -> bool:
        """注册工具"""
        tool = Tool(
            name=name,
            description=description,
            function=function,
            parameters=parameters or {},
            return_type=self._get_return_type(function),
            timeout=timeout
        )
        return self.registry.register_tool(tool)
        
    def _get_return_type(self, function: Callable) -> str:
        """获取函数返回类型"""
        sig = inspect.signature(function)
        if sig.return_annotation == inspect.Signature.empty:
            return "any"
        return str(sig.return_annotation)
        
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具"""
        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        # 创建执行上下文
        context = ExecutionContext(self.context_manager, self.memory_manager)
        await context.start_execution()
        
        try:
            # 验证参数
            self._validate_parameters(tool, parameters)
            
            # 执行工具
            result = await asyncio.wait_for(
                self._execute_tool_function(tool, parameters),
                timeout=tool.timeout
            )
            
            await context.complete_execution(result)
            self.execution_history.append(context)
            
            return {
                "success": True,
                "result": result,
                "execution_id": context.execution_id,
                "execution_time": context.end_time - context.start_time
            }
            
        except Exception as e:
            await context.fail_execution(e)
            self.execution_history.append(context)
            
            return {
                "success": False,
                "error": str(e),
                "execution_id": context.execution_id,
                "execution_time": context.end_time - context.start_time
            }
            
    def _validate_parameters(self, tool: Tool, parameters: Dict[str, Any]) -> None:
        """验证工具参数"""
        # 检查必需参数
        for param_name, param_info in tool.parameters.items():
            if param_info.get("required", False) and param_name not in parameters:
                raise ValueError(f"Missing required parameter: {param_name}")
                
        # 检查参数类型
        for param_name, param_value in parameters.items():
            if param_name in tool.parameters:
                expected_type = tool.parameters[param_name].get("type")
                if expected_type and not isinstance(param_value, eval(expected_type)):
                    raise ValueError(f"Parameter '{param_name}' must be of type {expected_type}")
                    
    async def _execute_tool_function(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """执行工具函数"""
        if asyncio.iscoroutinefunction(tool.function):
            return await tool.function(**parameters)
        else:
            return tool.function(**parameters)
            
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return [
            {
                "execution_id": ctx.execution_id,
                "status": ctx.status,
                "execution_time": ctx.end_time - ctx.start_time,
                "has_error": ctx.error is not None
            }
            for ctx in self.execution_history[-limit:]
        ]
```

## 配置管理

```python
# src/context_manager/utils/config.py
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class ContextManagerConfig:
    max_tokens: int = 4000
    compression_ratio: float = 0.8
    short_term_memory_size: int = 100
    medium_term_memory_path: str = "medium_term_memory.pkl"
    long_term_memory_path: str = "long_term_memory.json"
    tool_timeout: int = 30
    log_level: str = "INFO"
    enable_caching: bool = True
    cache_size: int = 1000

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> ContextManagerConfig:
        """加载配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ContextManagerConfig(**data)
            except Exception as e:
                print(f"Error loading config: {e}")
                
        return ContextManagerConfig()
        
    def save_config(self) -> None:
        """保存配置"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
            
    def get_config(self) -> ContextManagerConfig:
        """获取配置"""
        return self.config
        
    def update_config(self, **kwargs) -> None:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config()
```

## 使用示例

### 基础使用示例

```python
# examples/basic_usage.py
import asyncio
from src.context_manager.core.context_manager import ContextManager, Message, ContextPriority
from src.context_manager.memory.memory_manager import MemoryManager
from src.context_manager.tools.tool_executor import ToolExecutor

async def basic_example():
    # 初始化组件
    context_manager = ContextManager(max_tokens=2000)
    memory_manager = MemoryManager()
    tool_executor = ToolExecutor(context_manager, memory_manager)
    
    # 注册示例工具
    def calculate_sum(a: int, b: int) -> int:
        return a + b
    
    tool_executor.register_tool(
        name="calculate_sum",
        description="计算两个数的和",
        function=calculate_sum,
        parameters={
            "a": {"type": "int", "required": True},
            "b": {"type": "int", "required": True}
        }
    )
    
    # 添加消息到上下文
    message1 = Message(
        role="user",
        content="请计算 15 + 27 的结果",
        timestamp=asyncio.get_event_loop().time(),
        priority=ContextPriority.HIGH
    )
    
    await context_manager.add_message(message1)
    
    # 存储到记忆
    await memory_manager.store_memory(
        content="用户询问数学计算问题",
        importance=0.7,
        tags=["数学", "计算"],
        metadata={"question_type": "arithmetic"}
    )
    
    # 执行工具
    result = await tool_executor.execute_tool(
        "calculate_sum",
        {"a": 15, "b": 27}
    )
    
    print(f"工具执行结果: {result}")
    
    # 获取当前上下文
    context = await context_manager.get_context()
    print(f"当前上下文: {context}")
    
    # 检索相关记忆
    memories = await memory_manager.retrieve_memory("数学")
    print(f"相关记忆: {[mem.content for mem in memories]}")

if __name__ == "__main__":
    asyncio.run(basic_example())
```

### 高级功能示例

```python
# examples/advanced_features.py
import asyncio
from src.context_manager.core.context_manager import ContextManager, Message, ContextPriority
from src.context_manager.memory.memory_manager import MemoryManager
from src.context_manager.tools.tool_executor import ToolExecutor
from src.context_manager.utils.config import ConfigManager

async def advanced_example():
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 初始化组件
    context_manager = ContextManager(max_tokens=config.max_tokens)
    memory_manager = MemoryManager()
    tool_executor = ToolExecutor(context_manager, memory_manager)
    
    # 注册多个工具
    def search_web(query: str) -> str:
        # 模拟网络搜索
        return f"搜索结果: {query}"
    
    def analyze_sentiment(text: str) -> dict:
        # 模拟情感分析
        return {"sentiment": "positive", "confidence": 0.85}
    
    def generate_summary(text: str) -> str:
        # 模拟文本摘要
        return f"摘要: {text[:100]}..."
    
    tools = [
        {
            "name": "search_web",
            "description": "网络搜索",
            "function": search_web,
            "parameters": {
                "query": {"type": "str", "required": True}
            }
        },
        {
            "name": "analyze_sentiment",
            "description": "情感分析",
            "function": analyze_sentiment,
            "parameters": {
                "text": {"type": "str", "required": True}
            }
        },
        {
            "name": "generate_summary",
            "description": "生成摘要",
            "function": generate_summary,
            "parameters": {
                "text": {"type": "str", "required": True}
            }
        }
    ]
    
    for tool_config in tools:
        tool_executor.register_tool(**tool_config)
    
    # 模拟多轮对话
    conversations = [
        "你好，我想了解一下人工智能的最新发展",
        "请帮我搜索一下人工智能的最新进展",
        "请分析一下这个话题的情感倾向",
        "请给我一个简短的总结"
    ]
    
    for i, message_text in enumerate(conversations):
        message = Message(
            role="user",
            content=message_text,
            timestamp=asyncio.get_event_loop().time(),
            priority=ContextPriority.HIGH if i < 2 else ContextPriority.MEDIUM
        )
        
        await context_manager.add_message(message)
        
        # 根据内容执行相应的工具
        if "搜索" in message_text:
            result = await tool_executor.execute_tool(
                "search_web",
                {"query": "人工智能最新进展"}
            )
            print(f"搜索结果: {result}")
            
        elif "情感" in message_text:
            result = await tool_executor.execute_tool(
                "analyze_sentiment",
                {"text": "人工智能发展很快"}
            )
            print(f"情感分析: {result}")
            
        elif "总结" in message_text:
            result = await tool_executor.execute_tool(
                "generate_summary",
                {"text": "人工智能是当今科技发展的热点领域"}
            )
            print(f"摘要生成: {result}")
        
        # 存储重要信息到记忆
        if i % 2 == 0:
            await memory_manager.store_memory(
                content=f"用户讨论了: {message_text[:50]}...",
                importance=0.6,
                tags=["对话", "AI"],
                metadata={"conversation_index": i}
            )
    
    # 演示上下文压缩
    print(f"压缩前消息数量: {len(await context_manager.get_context())}")
    
    # 添加更多消息触发压缩
    for i in range(10):
        message = Message(
            role="assistant",
            content=f"这是第{i+1}条回复消息",
            timestamp=asyncio.get_event_loop().time(),
            priority=ContextPriority.LOW
        )
        await context_manager.add_message(message)
    
    print(f"压缩后消息数量: {len(await context_manager.get_context())}")
    
    # 演示记忆整合
    await memory_manager.consolidate_memories()
    
    # 获取执行历史
    history = tool_executor.get_execution_history()
    print(f"执行历史: {len(history)} 条记录")

if __name__ == "__main__":
    asyncio.run(advanced_example())
```

## 性能优化建议

### 1. 异步处理
- 使用async/await提高并发性能
- 避免阻塞操作，使用异步IO
- 合理使用线程池处理CPU密集型任务

### 2. 内存管理
- 使用弱引用避免内存泄漏
- 定期清理不再需要的对象
- 实现对象池复用资源

### 3. 缓存策略
- 实现多级缓存机制
- 使用LRU缓存淘汰策略
- 考虑内存缓存和磁盘缓存结合

### 4. 数据库优化
- 使用连接池管理数据库连接
- 实现批量操作减少数据库访问
- 考虑使用ORM框架提高开发效率

## 部署建议

### 1. 容器化部署
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "src.main"]
```

### 2. 配置管理
- 使用环境变量管理敏感信息
- 实现配置热更新机制
- 支持多环境配置

### 3. 监控和日志
- 实现结构化日志记录
- 添加性能监控指标
- 设置告警机制

## 测试策略

### 1. 单元测试
- 测试核心组件的功能
- 使用mock对象隔离依赖
- 覆盖边界情况和异常处理

### 2. 集成测试
- 测试组件间的交互
- 验证数据流的一致性
- 测试工具调用链路

### 3. 性能测试
- 压力测试并发处理能力
- 内存使用情况测试
- 响应时间测试

## 总结

本实现指南提供了一个完整的Python上下文管理和记忆管理系统的实现方案，包括：

1. **模块化设计**：清晰的组件分离和职责划分
2. **异步处理**：支持高并发的异步架构
3. **三层记忆**：短期、中期、长期记忆的完整实现
4. **工具系统**：灵活的工具注册和执行机制
5. **压缩优化**：智能的上下文压缩算法
6. **配置管理**：灵活的配置系统
7. **完整示例**：基础和高级功能的使用示例

这个实现可以作为构建智能对话系统的基础，提供了Claude Code核心功能的Python版本实现。

---

*创建时间：2025-07-31*
*最后更新：2025-07-31*