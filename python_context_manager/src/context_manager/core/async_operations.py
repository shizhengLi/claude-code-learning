"""
Async operations and concurrency control system.

This module provides robust async operations management, concurrency control,
and resource management for the context manager system.
"""

import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

from ..utils.logging import get_logger
from ..utils.error_handling import ContextManagerError, ConcurrencyError


logger = get_logger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ResourceLimit(Enum):
    """Resource limit types."""
    MAX_CONCURRENT_TASKS = "max_concurrent_tasks"
    MAX_THREAD_POOL_SIZE = "max_thread_pool_size"
    MAX_ASYNC_QUEUE_SIZE = "max_async_queue_size"
    TASK_TIMEOUT = "task_timeout"
    MEMORY_LIMIT = "memory_limit"
    CPU_LIMIT = "cpu_limit"


@dataclass
class Task:
    """Represents an async task."""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    progress: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count,
            'progress': self.progress,
            'error': str(self.error) if self.error else None
        }


@dataclass
class ResourceUsage:
    """Resource usage statistics."""
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'active_tasks': self.active_tasks,
            'queued_tasks': self.queued_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.average_execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent
        }


class ConcurrencyController:
    """
    Concurrency control and resource management system.
    
    Provides:
    - Task scheduling with priorities
    - Resource limit enforcement
    - Thread pool management
    - Async queue management
    - Task retry mechanisms
    - Progress tracking
    """
    
    def __init__(self, 
                 max_concurrent_tasks: int = 10,
                 max_thread_pool_size: int = 4,
                 max_async_queue_size: int = 100,
                 task_timeout: float = 300.0):
        """
        Initialize concurrency controller.
        
        Args:
            max_concurrent_tasks: Maximum concurrent tasks
            max_thread_pool_size: Maximum thread pool size
            max_async_queue_size: Maximum async queue size
            task_timeout: Default task timeout in seconds
        """
        # Resource limits
        self.resource_limits = {
            ResourceLimit.MAX_CONCURRENT_TASKS: max_concurrent_tasks,
            ResourceLimit.MAX_THREAD_POOL_SIZE: max_thread_pool_size,
            ResourceLimit.MAX_ASYNC_QUEUE_SIZE: max_async_queue_size,
            ResourceLimit.TASK_TIMEOUT: task_timeout
        }
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.pending_queue = asyncio.PriorityQueue()
        self.running_tasks: Set[str] = set()
        self.completed_tasks: Dict[str, Task] = {}
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_thread_pool_size)
        
        # Async task management
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.lock = asyncio.Lock()
        
        # Resource tracking
        self.resource_usage = ResourceUsage()
        self.execution_times = deque(maxlen=1000)
        
        # Event loop management
        try:
            self.event_loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop available yet, will be set later
            self.event_loop = None
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(f"Concurrency controller initialized with limits: {self.resource_limits}")
    
    async def start(self) -> None:
        """Start the concurrency controller."""
        if self._scheduler_task and not self._scheduler_task.done():
            logger.warning("Concurrency controller already started")
            return
        
        # Set event loop if not already set
        if self.event_loop is None:
            self.event_loop = asyncio.get_event_loop()
        
        # Start background tasks
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Concurrency controller started")
    
    async def stop(self) -> None:
        """Stop the concurrency controller."""
        self.shutdown_event.set()
        
        # Cancel pending tasks
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
        
        # Stop background tasks
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Concurrency controller stopped")
    
    async def submit_task(self, 
                         name: str,
                         func: Callable,
                         *args,
                         priority: Union[TaskPriority, str] = TaskPriority.NORMAL,
                         timeout: Optional[float] = None,
                         max_retries: int = 3,
                         **kwargs) -> str:
        """
        Submit a task for execution.
        
        Args:
            name: Task name
            func: Function to execute
            *args: Function arguments
            priority: Task priority (enum or string)
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        # Convert string priority to enum if needed
        if isinstance(priority, str):
            priority_map = {
                'low': TaskPriority.LOW,
                'normal': TaskPriority.NORMAL,
                'high': TaskPriority.HIGH,
                'critical': TaskPriority.CRITICAL
            }
            priority = priority_map.get(priority.lower(), TaskPriority.NORMAL)
        
        # Check queue limits
        if len(self.pending_queue._queue) >= self.resource_limits[ResourceLimit.MAX_ASYNC_QUEUE_SIZE]:
            raise ConcurrencyError("Task queue is full")
        
        # Create task
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.resource_limits[ResourceLimit.TASK_TIMEOUT],
            max_retries=max_retries
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Add to pending queue
        await self.pending_queue.put((-priority.value, task_id))
        
        logger.debug(f"Task submitted: {name} (ID: {task_id})")
        return task_id
    
    async def execute_task(self, 
                          name: str,
                          func: Callable,
                          *args,
                          priority: TaskPriority = TaskPriority.NORMAL,
                          timeout: Optional[float] = None,
                          max_retries: int = 3,
                          **kwargs) -> Any:
        """
        Execute a task and wait for completion.
        
        Args:
            name: Task name
            func: Function to execute
            *args: Function arguments
            priority: Task priority
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Function keyword arguments
            
        Returns:
            Task result
        """
        task_id = await self.submit_task(
            name,
            func,
            *args,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        
        # Wait for completion
        return await self.wait_for_task(task_id)
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout: Wait timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            ConcurrencyError: If task fails or times out
        """
        if task_id not in self.tasks:
            raise ConcurrencyError(f"Task not found: {task_id}")
        
        task = self.tasks[task_id]
        
        # Wait for task completion
        start_time = time.time()
        while True:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                break
            
            if timeout and (time.time() - start_time) > timeout:
                raise ConcurrencyError(f"Task wait timeout: {task_id}")
            
            await asyncio.sleep(0.1)
        
        # Handle task result
        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise ConcurrencyError(f"Task failed: {task.error}")
        elif task.status == TaskStatus.CANCELLED:
            raise ConcurrencyError(f"Task cancelled: {task_id}")
        elif task.status == TaskStatus.TIMEOUT:
            raise ConcurrencyError(f"Task timeout: {task_id}")
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information."""
        task = self.tasks.get(task_id)
        return task.to_dict() if task else None
    
    def get_all_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """Get all tasks, optionally filtered by status."""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [task for task in tasks if task.status == status]
        
        return [task.to_dict() for task in tasks]
    
    def get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        return self.resource_usage
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        task.status = TaskStatus.CANCELLED
        logger.info(f"Task cancelled: {task_id}")
        return True
    
    async def _scheduler_loop(self) -> None:
        """Background task scheduler loop."""
        while not self.shutdown_event.is_set():
            try:
                # Get next task from queue
                priority, task_id = await self.pending_queue.get()
                
                if task_id not in self.tasks:
                    continue
                
                task = self.tasks[task_id]
                
                if task.status != TaskStatus.PENDING:
                    continue
                
                # Check concurrency limits
                if len(self.running_tasks) >= self.resource_limits[ResourceLimit.MAX_CONCURRENT_TASKS]:
                    # Re-queue task
                    await self.pending_queue.put((priority, task_id))
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                asyncio.create_task(self._execute_task(task))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_loop(self) -> None:
        """Background resource monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
                # Update resource usage
                await self._update_resource_usage()
                
                # Check for stuck tasks
                await self._check_stuck_tasks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
    
    async def _execute_task(self, task: Task) -> None:
        """Execute a single task."""
        async with self.semaphore:
            async with self.lock:
                self.running_tasks.add(task.id)
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                self.resource_usage.active_tasks = len(self.running_tasks)
            
            try:
                logger.debug(f"Executing task: {task.name} (ID: {task.id})")
                
                # Determine execution method
                if asyncio.iscoroutinefunction(task.func):
                    # Async function
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    # Sync function - run in thread pool
                    result = await asyncio.wait_for(
                        self.event_loop.run_in_executor(
                            self.thread_pool,
                            functools.partial(task.func, *task.args, **task.kwargs)
                        ),
                        timeout=task.timeout
                    )
                
                # Task completed successfully
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                # Update statistics
                self.resource_usage.completed_tasks += 1
                execution_time = (task.completed_at - task.started_at).total_seconds()
                self.execution_times.append(execution_time)
                self.resource_usage.total_execution_time += execution_time
                self.resource_usage.average_execution_time = (
                    sum(self.execution_times) / len(self.execution_times)
                )
                
                logger.debug(f"Task completed: {task.name} (ID: {task.id})")
                
            except asyncio.TimeoutError:
                task.status = TaskStatus.TIMEOUT
                task.error = TimeoutError(f"Task timeout: {task.timeout}s")
                task.completed_at = datetime.now()
                
                self.resource_usage.failed_tasks += 1
                logger.warning(f"Task timeout: {task.name} (ID: {task.id})")
                
            except Exception as e:
                task.error = e
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    
                    # Re-queue with same priority
                    await self.pending_queue.put((-task.priority.value, task.id))
                    logger.info(f"Task retry {task.retry_count}/{task.max_retries}: {task.name} (ID: {task.id})")
                else:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    self.resource_usage.failed_tasks += 1
                    logger.error(f"Task failed: {task.name} (ID: {task.id}) - {e}")
            
            finally:
                async with self.lock:
                    self.running_tasks.discard(task.id)
                    self.resource_usage.active_tasks = len(self.running_tasks)
    
    async def _update_resource_usage(self) -> None:
        """Update resource usage statistics."""
        try:
            # Update queue size
            self.resource_usage.queued_tasks = len(self.pending_queue._queue)
            
            # Update memory usage
            try:
                import psutil
                process = psutil.Process()
                self.resource_usage.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                self.resource_usage.cpu_usage_percent = process.cpu_percent()
            except ImportError:
                pass
            
        except Exception as e:
            logger.error(f"Error updating resource usage: {e}")
    
    async def _check_stuck_tasks(self) -> None:
        """Check for tasks that are stuck."""
        current_time = datetime.now()
        
        for task_id in list(self.running_tasks):
            task = self.tasks.get(task_id)
            if not task:
                continue
            
            # Check if task has been running too long
            if (task.started_at and 
                (current_time - task.started_at).total_seconds() > task.timeout * 2):
                
                # Cancel stuck task
                task.status = TaskStatus.TIMEOUT
                task.error = TimeoutError("Task stuck - force cancelled")
                task.completed_at = current_time
                
                self.resource_usage.failed_tasks += 1
                logger.warning(f"Stuck task cancelled: {task.name} (ID: {task.id})")


class AsyncOperationManager:
    """
    High-level async operation manager.
    
    Provides convenient methods for common async operations
    with proper error handling and resource management.
    """
    
    def __init__(self, concurrency_controller: ConcurrencyController):
        """
        Initialize async operation manager.
        
        Args:
            concurrency_controller: Concurrency controller instance
        """
        self.concurrency_controller = concurrency_controller
        self.operation_handlers = {}
        
        logger.info("Async operation manager initialized")
    
    async def execute_with_retry(self,
                               name: str,
                               func: Callable,
                               *args,
                               max_retries: int = 3,
                               retry_delay: float = 1.0,
                               backoff_factor: float = 2.0,
                               **kwargs) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            name: Operation name
            func: Function to execute
            *args: Function arguments
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay
            backoff_factor: Backoff multiplier
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = retry_delay * (backoff_factor ** (attempt - 1))
                    logger.info(f"Retry attempt {attempt}/{max_retries} for {name} (delay: {delay}s)")
                    await asyncio.sleep(delay)
                
                return await self.concurrency_controller.execute_task(
                    name=f"{name}_attempt_{attempt}",
                    func=func,
                    *args,
                    priority=TaskPriority.HIGH,
                    **kwargs
                )
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {name}: {e}")
                
                if attempt == max_retries:
                    break
        
        raise ConcurrencyError(f"Operation failed after {max_retries + 1} attempts: {last_error}")
    
    async def execute_parallel(self,
                             operations: List[Dict[str, Any]],
                             max_concurrent: Optional[int] = None) -> List[Any]:
        """
        Execute multiple operations in parallel.
        
        Args:
            operations: List of operation dictionaries
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of results
        """
        if not operations:
            return []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent or len(operations))
        
        async def execute_operation(operation: Dict[str, Any]) -> Any:
            async with semaphore:
                return await self.concurrency_controller.execute_task(
                    name=operation['name'],
                    func=operation['func'],
                    *operation.get('args', []),
                    **operation.get('kwargs', {})
                )
        
        # Execute all operations
        tasks = [execute_operation(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel operation {operations[i]['name']} failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_with_timeout(self,
                                 name: str,
                                 func: Callable,
                                 *args,
                                 timeout: float,
                                 **kwargs) -> Any:
        """
        Execute function with timeout.
        
        Args:
            name: Operation name
            func: Function to execute
            *args: Function arguments
            timeout: Timeout in seconds
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        try:
            return await asyncio.wait_for(
                self.concurrency_controller.execute_task(
                    name=name,
                    func=func,
                    *args,
                    priority=TaskPriority.HIGH,
                    timeout=timeout,
                    **kwargs
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise ConcurrencyError(f"Operation timeout: {name}")
    
    async def execute_with_progress(self,
                                  name: str,
                                  func: Callable,
                                  *args,
                                  progress_callback: Optional[Callable[[float], None]] = None,
                                  **kwargs) -> Any:
        """
        Execute function with progress reporting.
        
        Args:
            name: Operation name
            func: Function to execute
            *args: Function arguments
            progress_callback: Progress callback function
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Wrap function with progress tracking
        async def progress_wrapper():
            result = await func(*args, **kwargs)
            
            # Simulate progress (in real implementation, func would report progress)
            if progress_callback:
                for i in range(10):
                    progress = (i + 1) / 10.0
                    progress_callback(progress)
                    await asyncio.sleep(0.1)
            
            return result
        
        return await self.concurrency_controller.execute_task(
            name=name,
            func=progress_wrapper,
            priority=TaskPriority.NORMAL
        )
    
    def register_operation_handler(self, operation_type: str, handler: Callable) -> None:
        """
        Register an operation handler.
        
        Args:
            operation_type: Type of operation
            handler: Handler function
        """
        self.operation_handlers[operation_type] = handler
        logger.info(f"Registered operation handler: {operation_type}")
    
    async def handle_operation(self, operation_type: str, *args, **kwargs) -> Any:
        """
        Handle an operation using registered handler.
        
        Args:
            operation_type: Type of operation
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        handler = self.operation_handlers.get(operation_type)
        if not handler:
            raise ConcurrencyError(f"No handler registered for operation type: {operation_type}")
        
        return await self.concurrency_controller.execute_task(
            name=f"operation_{operation_type}",
            func=handler,
            *args,
            **kwargs
        )