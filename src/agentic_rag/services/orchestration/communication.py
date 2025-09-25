"""
Enhanced Agent Communication Framework

This module implements advanced communication and state management between agents.
It provides message passing with routing, shared state management, coordination protocols,
result aggregation mechanisms, and distributed workflow coordination.
"""

import asyncio
import json
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from uuid import uuid4
import weakref

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class MessageType(str, Enum):
    """Enhanced types of messages that can be sent between agents."""

    # Task coordination
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_PROGRESS = "task_progress"
    TASK_CANCELLATION = "task_cancellation"

    # Status and monitoring
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_METRICS = "performance_metrics"

    # Error handling
    ERROR_NOTIFICATION = "error_notification"
    WARNING_NOTIFICATION = "warning_notification"
    RECOVERY_REQUEST = "recovery_request"

    # Coordination protocols
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    LEADER_ELECTION = "leader_election"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESPONSE = "consensus_response"
    BARRIER_SYNC = "barrier_sync"

    # State management
    STATE_UPDATE = "state_update"
    STATE_QUERY = "state_query"
    STATE_LOCK_REQUEST = "state_lock_request"
    STATE_LOCK_RESPONSE = "state_lock_response"
    STATE_SYNC = "state_sync"

    # Resource management
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    RESOURCE_LOCK = "resource_lock"
    RESOURCE_RELEASE = "resource_release"

    # Workflow coordination
    WORKFLOW_START = "workflow_start"
    WORKFLOW_STEP = "workflow_step"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ABORT = "workflow_abort"

    # Result aggregation
    RESULT_PARTIAL = "result_partial"
    RESULT_FINAL = "result_final"
    RESULT_REQUEST = "result_request"

    # Lifecycle management
    AGENT_STARTUP = "agent_startup"
    AGENT_READY = "agent_ready"
    AGENT_SHUTDOWN = "agent_shutdown"
    HANDOFF_REQUEST = "handoff_request"
    HANDOFF_RESPONSE = "handoff_response"


class MessagePriority(str, Enum):
    """Message priority levels with enhanced granularity."""

    BACKGROUND = "background"    # Non-urgent background tasks
    LOW = "low"                 # Low priority messages
    NORMAL = "normal"           # Standard priority
    HIGH = "high"               # High priority messages
    URGENT = "urgent"           # Urgent messages requiring immediate attention
    CRITICAL = "critical"       # Critical system messages


class DeliveryGuarantee(str, Enum):
    """Message delivery guarantee levels."""

    AT_MOST_ONCE = "at_most_once"      # Fire and forget
    AT_LEAST_ONCE = "at_least_once"    # Retry until delivered
    EXACTLY_ONCE = "exactly_once"      # Guaranteed single delivery
    ORDERED = "ordered"                 # Maintain message order


class Message(BaseModel):
    """Enhanced message sent between agents with advanced routing and delivery features."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: MessageType = Field(..., description="Message type")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    delivery_guarantee: DeliveryGuarantee = Field(default=DeliveryGuarantee.AT_MOST_ONCE)

    # Routing and addressing
    sender_id: str = Field(..., description="Sender agent ID")
    recipient_id: Optional[str] = Field(None, description="Recipient agent ID (None for broadcast)")
    recipient_group: Optional[str] = Field(None, description="Recipient group for multicast")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request-response")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-message exchanges")

    # Content and serialization
    subject: str = Field(..., description="Message subject")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Message payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    content_type: str = Field(default="application/json", description="Content type")
    encoding: str = Field(default="utf-8", description="Content encoding")

    # Timing and lifecycle
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None, description="Message expiration time")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")

    # Delivery and acknowledgment tracking
    delivered: bool = Field(default=False)
    delivered_at: Optional[datetime] = None
    acknowledged: bool = Field(default=False)
    acknowledged_at: Optional[datetime] = None
    retry_count: int = Field(default=0, description="Number of delivery attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Routing history and tracing
    routing_path: List[str] = Field(default_factory=list, description="Message routing path")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

    # Security and validation
    checksum: Optional[str] = Field(None, description="Message integrity checksum")
    signature: Optional[str] = Field(None, description="Message signature for authentication")

    class Config:
        use_enum_values = True

    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at:
            return datetime.now(timezone.utc) > self.expires_at
        if self.ttl_seconds:
            age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
            return age > self.ttl_seconds
        return False

    def can_retry(self) -> bool:
        """Check if the message can be retried."""
        return self.retry_count < self.max_retries and not self.is_expired()

    def add_routing_hop(self, agent_id: str) -> None:
        """Add a routing hop to the message path."""
        self.routing_path.append(f"{agent_id}@{datetime.now(timezone.utc).isoformat()}")

    def serialize(self) -> str:
        """Serialize message to JSON string."""
        return self.json()

    @classmethod
    def deserialize(cls, data: str) -> 'Message':
        """Deserialize message from JSON string."""
        return cls.parse_raw(data)


class StateOperation(str, Enum):
    """Types of state operations."""

    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    LOCK = "lock"
    UNLOCK = "unlock"
    MERGE = "merge"


class StateLock(BaseModel):
    """Represents a lock on shared state."""

    lock_id: str = Field(default_factory=lambda: str(uuid4()))
    key: str = Field(..., description="State key being locked")
    owner_id: str = Field(..., description="Agent ID that owns the lock")
    lock_type: str = Field(default="exclusive", description="Lock type (exclusive, shared)")
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None, description="Lock expiration time")
    renewable: bool = Field(default=True, description="Whether lock can be renewed")

    def is_expired(self) -> bool:
        """Check if the lock has expired."""
        if self.expires_at:
            return datetime.now(timezone.utc) > self.expires_at
        return False


class StateVersion(BaseModel):
    """Represents a version of shared state."""

    version_id: str = Field(default_factory=lambda: str(uuid4()))
    key: str = Field(..., description="State key")
    value: Any = Field(..., description="State value")
    version_number: int = Field(..., description="Version number")
    created_by: str = Field(..., description="Agent ID that created this version")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_version: Optional[str] = Field(None, description="Parent version ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class SharedStateManager:
    """Thread-safe shared state manager with versioning and conflict resolution."""

    def __init__(self, persistence_enabled: bool = False):
        self._state: Dict[str, Any] = {}
        self._versions: Dict[str, List[StateVersion]] = defaultdict(list)
        self._locks: Dict[str, StateLock] = {}
        self._lock = threading.RLock()
        self._persistence_enabled = persistence_enabled
        self._change_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._logger = structlog.get_logger(__name__).bind(component="state_manager")

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value by key."""
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value: Any, agent_id: str, metadata: Optional[Dict] = None) -> StateVersion:
        """Set state value with versioning."""
        with self._lock:
            # Create new version
            current_versions = self._versions[key]
            version_number = len(current_versions) + 1

            parent_version = current_versions[-1].version_id if current_versions else None

            new_version = StateVersion(
                key=key,
                value=value,
                version_number=version_number,
                created_by=agent_id,
                parent_version=parent_version,
                metadata=metadata or {}
            )

            # Update state and version history
            self._state[key] = value
            self._versions[key].append(new_version)

            # Notify listeners
            self._notify_change_listeners(key, value, agent_id)

            self._logger.debug(f"State updated: {key} = {value} by {agent_id}")
            return new_version

    def update(self, key: str, updater: Callable[[Any], Any], agent_id: str) -> StateVersion:
        """Atomically update state value."""
        with self._lock:
            current_value = self._state.get(key)
            new_value = updater(current_value)
            return self.set(key, new_value, agent_id)

    def delete(self, key: str, agent_id: str) -> bool:
        """Delete state key."""
        with self._lock:
            if key in self._state:
                del self._state[key]
                self._notify_change_listeners(key, None, agent_id)
                self._logger.debug(f"State deleted: {key} by {agent_id}")
                return True
            return False

    def acquire_lock(self, key: str, agent_id: str, lock_type: str = "exclusive",
                    timeout_seconds: Optional[int] = None) -> Optional[StateLock]:
        """Acquire a lock on a state key."""
        with self._lock:
            # Check if key is already locked
            if key in self._locks:
                existing_lock = self._locks[key]
                if not existing_lock.is_expired():
                    if existing_lock.owner_id == agent_id:
                        # Renew existing lock
                        if existing_lock.renewable:
                            if timeout_seconds:
                                existing_lock.expires_at = (
                                    datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
                                )
                            return existing_lock
                    return None  # Lock held by another agent
                else:
                    # Remove expired lock
                    del self._locks[key]

            # Create new lock
            new_lock = StateLock(
                key=key,
                owner_id=agent_id,
                lock_type=lock_type,
                expires_at=(
                    datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
                    if timeout_seconds else None
                )
            )

            self._locks[key] = new_lock
            self._logger.debug(f"Lock acquired: {key} by {agent_id}")
            return new_lock

    def release_lock(self, key: str, agent_id: str) -> bool:
        """Release a lock on a state key."""
        with self._lock:
            if key in self._locks:
                lock = self._locks[key]
                if lock.owner_id == agent_id:
                    del self._locks[key]
                    self._logger.debug(f"Lock released: {key} by {agent_id}")
                    return True
            return False

    def get_version_history(self, key: str, limit: Optional[int] = None) -> List[StateVersion]:
        """Get version history for a state key."""
        with self._lock:
            versions = self._versions[key]
            if limit:
                return versions[-limit:]
            return versions.copy()

    def rollback_to_version(self, key: str, version_id: str, agent_id: str) -> bool:
        """Rollback state to a specific version."""
        with self._lock:
            versions = self._versions[key]
            for version in versions:
                if version.version_id == version_id:
                    self._state[key] = version.value
                    self._notify_change_listeners(key, version.value, agent_id)
                    self._logger.info(f"State rolled back: {key} to version {version_id} by {agent_id}")
                    return True
            return False

    def add_change_listener(self, key: str, listener: Callable[[str, Any, str], None]) -> None:
        """Add a change listener for a state key."""
        self._change_listeners[key].append(listener)

    def remove_change_listener(self, key: str, listener: Callable) -> None:
        """Remove a change listener for a state key."""
        if key in self._change_listeners:
            try:
                self._change_listeners[key].remove(listener)
            except ValueError:
                pass

    def _notify_change_listeners(self, key: str, value: Any, agent_id: str) -> None:
        """Notify change listeners."""
        for listener in self._change_listeners[key]:
            try:
                listener(key, value, agent_id)
            except Exception as e:
                self._logger.error(f"Error in change listener: {e}")

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current state."""
        with self._lock:
            return self._state.copy()

    def get_lock_status(self) -> Dict[str, StateLock]:
        """Get current lock status."""
        with self._lock:
            # Remove expired locks
            expired_keys = [
                key for key, lock in self._locks.items()
                if lock.is_expired()
            ]
            for key in expired_keys:
                del self._locks[key]

            return self._locks.copy()


class CommunicationChannel:
    """Enhanced communication channel with advanced message routing and delivery guarantees."""

    def __init__(self, channel_id: str, max_queue_size: int = 1000, enable_persistence: bool = False):
        self.channel_id = channel_id
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence

        # Message queues with priority support
        self._message_queues: Dict[str, Dict[MessagePriority, asyncio.Queue]] = {}
        self._subscribers: Set[str] = set()
        self._subscriber_groups: Dict[str, Set[str]] = defaultdict(set)

        # Message tracking and delivery
        self._pending_messages: Dict[str, Message] = {}
        self._message_history: deque = deque(maxlen=10000)
        self._delivery_confirmations: Dict[str, Set[str]] = defaultdict(set)

        # Routing and filtering
        self._message_filters: Dict[str, List[Callable[[Message], bool]]] = defaultdict(list)
        self._routing_rules: List[Callable[[Message], Optional[str]]] = []

        # Performance tracking
        self._message_stats = {
            'sent': 0,
            'delivered': 0,
            'failed': 0,
            'retried': 0
        }

        self._logger = structlog.get_logger(__name__).bind(channel_id=channel_id)
    
    async def subscribe(self, agent_id: str, groups: Optional[List[str]] = None) -> Dict[MessagePriority, asyncio.Queue]:
        """Subscribe an agent to this channel with priority queues."""
        if agent_id not in self._message_queues:
            # Create priority queues for the agent
            priority_queues = {}
            for priority in MessagePriority:
                priority_queues[priority] = asyncio.Queue(maxsize=self.max_queue_size)

            self._message_queues[agent_id] = priority_queues
            self._subscribers.add(agent_id)

            # Add to groups if specified
            if groups:
                for group in groups:
                    self._subscriber_groups[group].add(agent_id)

            self._logger.info(f"Agent {agent_id} subscribed to channel with groups: {groups or []}")

        return self._message_queues[agent_id]

    async def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from this channel."""
        if agent_id in self._subscribers:
            self._subscribers.remove(agent_id)

            # Remove from all groups
            for group_members in self._subscriber_groups.values():
                group_members.discard(agent_id)

            if agent_id in self._message_queues:
                # Clear any remaining messages from all priority queues
                priority_queues = self._message_queues[agent_id]
                for queue in priority_queues.values():
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                del self._message_queues[agent_id]

            self._logger.info(f"Agent {agent_id} unsubscribed from channel")

    def add_message_filter(self, agent_id: str, filter_func: Callable[[Message], bool]) -> None:
        """Add a message filter for an agent."""
        self._message_filters[agent_id].append(filter_func)

    def add_routing_rule(self, rule_func: Callable[[Message], Optional[str]]) -> None:
        """Add a routing rule for message forwarding."""
        self._routing_rules.append(rule_func)

    def _should_deliver_message(self, message: Message, agent_id: str) -> bool:
        """Check if a message should be delivered to an agent based on filters."""
        for filter_func in self._message_filters[agent_id]:
            try:
                if not filter_func(message):
                    return False
            except Exception as e:
                self._logger.warning(f"Message filter error for {agent_id}: {e}")
        return True

    def _get_priority_queue(self, agent_id: str, priority: MessagePriority) -> Optional[asyncio.Queue]:
        """Get the appropriate priority queue for an agent."""
        if agent_id in self._message_queues:
            return self._message_queues[agent_id].get(priority)
        return None
    
    async def send_message(self, message: Message) -> bool:
        """Send a message through this channel with enhanced routing and delivery guarantees."""
        try:
            self._message_stats['sent'] += 1
            message.add_routing_hop(self.channel_id)

            # Store message for tracking
            self._pending_messages[message.id] = message
            self._message_history.append(message)

            # Apply routing rules
            for rule in self._routing_rules:
                try:
                    route_to = rule(message)
                    if route_to and route_to != message.recipient_id:
                        message.recipient_id = route_to
                        break
                except Exception as e:
                    self._logger.warning(f"Routing rule error: {e}")

            # Handle direct messages
            if message.recipient_id:
                return await self._send_direct_message(message)

            # Handle group messages
            elif message.recipient_group:
                return await self._send_group_message(message)

            # Handle broadcast messages
            else:
                return await self._send_broadcast_message(message)

        except Exception as e:
            self._logger.error(f"Error sending message {message.id}: {e}")
            self._message_stats['failed'] += 1
            return False

    async def _send_direct_message(self, message: Message) -> bool:
        """Send a message to a specific recipient."""
        recipient_id = message.recipient_id

        if recipient_id not in self._subscribers:
            self._logger.warning(f"Recipient {recipient_id} not subscribed to channel")
            return False

        if not self._should_deliver_message(message, recipient_id):
            self._logger.debug(f"Message {message.id} filtered out for {recipient_id}")
            return False

        queue = self._get_priority_queue(recipient_id, message.priority)
        if not queue:
            self._logger.error(f"No queue found for {recipient_id}")
            return False

        try:
            await queue.put(message)
            message.delivered = True
            message.delivered_at = datetime.now(timezone.utc)
            self._message_stats['delivered'] += 1

            # Handle delivery guarantees
            if message.delivery_guarantee in [DeliveryGuarantee.AT_LEAST_ONCE, DeliveryGuarantee.EXACTLY_ONCE]:
                self._delivery_confirmations[message.id].add(recipient_id)

            self._logger.debug(f"Message {message.id} delivered to {recipient_id}")
            return True

        except asyncio.QueueFull:
            self._logger.warning(f"Message queue full for agent {recipient_id}")

            # Handle retry logic for guaranteed delivery
            if message.can_retry() and message.delivery_guarantee != DeliveryGuarantee.AT_MOST_ONCE:
                message.retry_count += 1
                self._message_stats['retried'] += 1
                # Schedule retry (simplified - in production would use proper retry scheduling)
                asyncio.create_task(self._retry_message_delivery(message))

            return False

    async def _send_group_message(self, message: Message) -> bool:
        """Send a message to a group of recipients."""
        group = message.recipient_group

        if group not in self._subscriber_groups:
            self._logger.warning(f"Group {group} not found")
            return False

        recipients = self._subscriber_groups[group]
        if not recipients:
            self._logger.warning(f"No subscribers in group {group}")
            return False

        delivered_count = 0
        for recipient_id in recipients:
            if recipient_id != message.sender_id:  # Don't send to sender
                if self._should_deliver_message(message, recipient_id):
                    queue = self._get_priority_queue(recipient_id, message.priority)
                    if queue:
                        try:
                            await queue.put(message)
                            delivered_count += 1

                            if message.delivery_guarantee in [DeliveryGuarantee.AT_LEAST_ONCE, DeliveryGuarantee.EXACTLY_ONCE]:
                                self._delivery_confirmations[message.id].add(recipient_id)

                        except asyncio.QueueFull:
                            self._logger.warning(f"Message queue full for agent {recipient_id}")

        if delivered_count > 0:
            message.delivered = True
            message.delivered_at = datetime.now(timezone.utc)
            self._message_stats['delivered'] += delivered_count

            self._logger.debug(f"Group message {message.id} delivered to {delivered_count} recipients")
            return True

        return False

    async def _send_broadcast_message(self, message: Message) -> bool:
        """Send a broadcast message to all subscribers."""
        delivered_count = 0

        for agent_id in self._subscribers:
            if agent_id != message.sender_id:  # Don't send to sender
                if self._should_deliver_message(message, agent_id):
                    queue = self._get_priority_queue(agent_id, message.priority)
                    if queue:
                        try:
                            await queue.put(message)
                            delivered_count += 1

                            if message.delivery_guarantee in [DeliveryGuarantee.AT_LEAST_ONCE, DeliveryGuarantee.EXACTLY_ONCE]:
                                self._delivery_confirmations[message.id].add(agent_id)

                        except asyncio.QueueFull:
                            self._logger.warning(f"Message queue full for agent {agent_id}")

        if delivered_count > 0:
            message.delivered = True
            message.delivered_at = datetime.now(timezone.utc)
            self._message_stats['delivered'] += delivered_count

            self._logger.debug(f"Broadcast message {message.id} delivered to {delivered_count} agents")
            return True

        return False

    async def _retry_message_delivery(self, message: Message) -> None:
        """Retry message delivery after a delay."""
        await asyncio.sleep(min(2 ** message.retry_count, 30))  # Exponential backoff
        await self.send_message(message)

    async def acknowledge_message(self, message_id: str, agent_id: str) -> bool:
        """Acknowledge receipt of a message."""
        if message_id in self._pending_messages:
            message = self._pending_messages[message_id]
            message.acknowledged = True
            message.acknowledged_at = datetime.now(timezone.utc)

            # For exactly-once delivery, remove from pending after acknowledgment
            if message.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
                del self._pending_messages[message_id]

            self._logger.debug(f"Message {message_id} acknowledged by {agent_id}")
            return True

        return False

    async def get_messages(self, agent_id: str, priority: Optional[MessagePriority] = None,
                          timeout: Optional[float] = None) -> List[Message]:
        """Get messages for an agent from priority queues."""
        if agent_id not in self._message_queues:
            return []

        messages = []
        priority_queues = self._message_queues[agent_id]

        # If specific priority requested
        if priority:
            queue = priority_queues.get(priority)
            if queue:
                try:
                    if timeout:
                        message = await asyncio.wait_for(queue.get(), timeout=timeout)
                        messages.append(message)
                    else:
                        while not queue.empty():
                            message = queue.get_nowait()
                            messages.append(message)
                except (asyncio.QueueEmpty, asyncio.TimeoutError):
                    pass
        else:
            # Get messages from all priority queues in order
            for priority_level in [MessagePriority.CRITICAL, MessagePriority.URGENT,
                                 MessagePriority.HIGH, MessagePriority.NORMAL,
                                 MessagePriority.LOW, MessagePriority.BACKGROUND]:
                queue = priority_queues.get(priority_level)
                if queue:
                    try:
                        while not queue.empty():
                            message = queue.get_nowait()
                            messages.append(message)
                    except asyncio.QueueEmpty:
                        continue

        return messages
    
    def get_channel_stats(self) -> Dict[str, Any]:
        """Get comprehensive channel statistics."""
        return {
            'channel_id': self.channel_id,
            'subscribers': len(self._subscribers),
            'groups': len(self._subscriber_groups),
            'pending_messages': len(self._pending_messages),
            'message_history_size': len(self._message_history),
            'message_stats': self._message_stats.copy(),
            'queue_sizes': {
                agent_id: {
                    priority.value: queue.qsize()
                    for priority, queue in queues.items()
                }
                for agent_id, queues in self._message_queues.items()
            }
        }

    def cleanup_expired_messages(self) -> int:
        """Clean up expired messages and return count of cleaned messages."""
        expired_count = 0

        # Clean up pending messages
        expired_message_ids = [
            msg_id for msg_id, msg in self._pending_messages.items()
            if msg.is_expired()
        ]

        for msg_id in expired_message_ids:
            del self._pending_messages[msg_id]
            expired_count += 1

        # Clean up delivery confirmations
        for msg_id in expired_message_ids:
            if msg_id in self._delivery_confirmations:
                del self._delivery_confirmations[msg_id]

        if expired_count > 0:
            self._logger.debug(f"Cleaned up {expired_count} expired messages")

        return expired_count


class CoordinationProtocol(str, Enum):
    """Types of coordination protocols."""

    LEADER_ELECTION = "leader_election"
    CONSENSUS = "consensus"
    BARRIER_SYNC = "barrier_sync"
    RESOURCE_LOCK = "resource_lock"
    WORKFLOW_COORDINATION = "workflow_coordination"


class ResultAggregationStrategy(str, Enum):
    """Result aggregation strategies."""

    FIRST_WINS = "first_wins"           # Use first result received
    MAJORITY_VOTE = "majority_vote"     # Use majority consensus
    WEIGHTED_AVERAGE = "weighted_average"  # Weight by confidence/performance
    BEST_CONFIDENCE = "best_confidence"    # Use result with highest confidence
    MERGE_ALL = "merge_all"             # Merge all results
    CUSTOM = "custom"                   # Custom aggregation function


class AgentCommunicationFramework:
    """Enhanced framework for managing agent communication, coordination, and state."""

    def __init__(self, enable_persistence: bool = False):
        self._channels: Dict[str, CommunicationChannel] = {}
        self._shared_state_manager = SharedStateManager(enable_persistence)

        # Coordination protocols
        self._coordination_sessions: Dict[str, Dict[str, Any]] = {}
        self._barriers: Dict[str, Dict[str, Any]] = {}
        self._elections: Dict[str, Dict[str, Any]] = {}

        # Result aggregation
        self._result_collectors: Dict[str, Dict[str, Any]] = {}
        self._aggregation_strategies: Dict[str, ResultAggregationStrategy] = {}

        # Performance tracking
        self._coordination_metrics = {
            'protocols_started': 0,
            'protocols_completed': 0,
            'protocols_failed': 0,
            'average_coordination_time_ms': 0.0
        }

        # Create default channel
        self._default_channel = CommunicationChannel("default", enable_persistence=enable_persistence)
        self._channels["default"] = self._default_channel

        self._logger = structlog.get_logger(__name__).bind(component="communication_framework")
    
    async def initialize(self) -> None:
        """Initialize the communication framework."""
        self._logger.info("Initializing agent communication framework")
    
    async def shutdown(self) -> None:
        """Shutdown the communication framework."""
        self._logger.info("Shutting down agent communication framework")

        # Cleanup channels
        for channel in self._channels.values():
            channel.cleanup_expired_messages()

        # Clear all channels and coordination state
        self._channels.clear()
        self._coordination_sessions.clear()
        self._barriers.clear()
        self._elections.clear()
        self._result_collectors.clear()
    
    def create_channel(self, channel_id: str, max_queue_size: int = 1000) -> CommunicationChannel:
        """Create a new communication channel."""
        if channel_id in self._channels:
            return self._channels[channel_id]
        
        channel = CommunicationChannel(channel_id, max_queue_size)
        self._channels[channel_id] = channel
        
        self._logger.info(f"Created communication channel: {channel_id}")
        return channel
    
    def get_channel(self, channel_id: str = "default") -> Optional[CommunicationChannel]:
        """Get a communication channel by ID."""
        return self._channels.get(channel_id)
    
    async def subscribe_agent(self, agent_id: str, channel_id: str = "default") -> Optional[asyncio.Queue]:
        """Subscribe an agent to a communication channel."""
        channel = self.get_channel(channel_id)
        if channel:
            return await channel.subscribe(agent_id)
        return None
    
    async def unsubscribe_agent(self, agent_id: str, channel_id: str = "default") -> None:
        """Unsubscribe an agent from a communication channel."""
        channel = self.get_channel(channel_id)
        if channel:
            await channel.unsubscribe(agent_id)
    
    async def send_message(
        self,
        message: Message,
        channel_id: str = "default"
    ) -> bool:
        """Send a message through a communication channel."""
        channel = self.get_channel(channel_id)
        if channel:
            return await channel.send_message(message)
        return False
    
    async def broadcast_message(
        self,
        sender_id: str,
        subject: str,
        payload: Dict[str, Any],
        channel_id: str = "default",
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Broadcast a message to all agents in a channel."""
        message = Message(
            type=MessageType.STATUS_UPDATE,
            priority=priority,
            sender_id=sender_id,
            subject=subject,
            payload=payload
        )
        
        return await self.send_message(message, channel_id)
    
    async def get_shared_state(self, key: str) -> Any:
        """Get a value from shared state."""
        return self._shared_state.get(key)
    
    async def set_shared_state(self, key: str, value: Any) -> None:
        """Set a value in shared state."""
        # Get or create lock for this key
        if key not in self._state_locks:
            self._state_locks[key] = asyncio.Lock()
        
        async with self._state_locks[key]:
            self._shared_state[key] = value
    
    async def update_shared_state(self, key: str, updater_func) -> Any:
        """Update shared state using an updater function."""
        if key not in self._state_locks:
            self._state_locks[key] = asyncio.Lock()
        
        async with self._state_locks[key]:
            current_value = self._shared_state.get(key)
            new_value = updater_func(current_value)
            self._shared_state[key] = new_value
            return new_value
    
    async def delete_shared_state(self, key: str) -> bool:
        """Delete a key from shared state."""
        if key not in self._state_locks:
            self._state_locks[key] = asyncio.Lock()
        
        async with self._state_locks[key]:
            if key in self._shared_state:
                del self._shared_state[key]
                return True
            return False

    # === State Management Methods ===

    def get_shared_state(self, key: str, default: Any = None) -> Any:
        """Get shared state value."""
        return self._shared_state_manager.get(key, default)

    def set_shared_state(self, key: str, value: Any, agent_id: str, metadata: Optional[Dict] = None) -> StateVersion:
        """Set shared state value."""
        return self._shared_state_manager.set(key, value, agent_id, metadata)

    def acquire_state_lock(self, key: str, agent_id: str, timeout_seconds: Optional[int] = None) -> Optional[StateLock]:
        """Acquire a lock on shared state."""
        return self._shared_state_manager.acquire_lock(key, agent_id, timeout_seconds=timeout_seconds)

    def release_state_lock(self, key: str, agent_id: str) -> bool:
        """Release a lock on shared state."""
        return self._shared_state_manager.release_lock(key, agent_id)

    # === Coordination Protocols ===

    async def create_barrier(self, barrier_id: str, participant_count: int, timeout_seconds: int = 60) -> bool:
        """Create a synchronization barrier for multiple agents."""
        barrier_data = {
            'barrier_id': barrier_id,
            'participant_count': participant_count,
            'arrived_agents': set(),
            'status': 'waiting',
            'created_at': datetime.now(timezone.utc),
            'timeout_at': datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
        }

        self._barriers[barrier_id] = barrier_data
        self._logger.info(f"Created barrier {barrier_id} for {participant_count} participants")
        return True

    async def wait_at_barrier(self, barrier_id: str, agent_id: str) -> bool:
        """Wait at a synchronization barrier."""
        if barrier_id not in self._barriers:
            return False

        barrier = self._barriers[barrier_id]

        # Check if barrier has timed out
        if datetime.now(timezone.utc) > barrier['timeout_at']:
            barrier['status'] = 'timeout'
            return False

        # Add agent to barrier
        barrier['arrived_agents'].add(agent_id)

        # Check if all participants have arrived
        if len(barrier['arrived_agents']) >= barrier['participant_count']:
            barrier['status'] = 'released'

            # Notify all waiting agents
            channel = self.get_channel()
            if channel:
                for participant in barrier['arrived_agents']:
                    message = Message(
                        type=MessageType.BARRIER_SYNC,
                        sender_id="framework",
                        recipient_id=participant,
                        subject=f"Barrier Released: {barrier_id}",
                        payload={'barrier_id': barrier_id, 'status': 'released'},
                        priority=MessagePriority.HIGH
                    )
                    await channel.send_message(message)

            self._logger.info(f"Barrier {barrier_id} released with {len(barrier['arrived_agents'])} participants")
            return True

        # Wait for other participants
        while (barrier['status'] == 'waiting' and
               datetime.now(timezone.utc) <= barrier['timeout_at']):
            await asyncio.sleep(0.1)

        return barrier['status'] == 'released'

    # === Result Aggregation ===

    def create_result_collector(self, collector_id: str, expected_results: int,
                               strategy: ResultAggregationStrategy = ResultAggregationStrategy.MAJORITY_VOTE,
                               timeout_seconds: int = 30) -> bool:
        """Create a result collector for aggregating agent results."""
        collector_data = {
            'collector_id': collector_id,
            'expected_results': expected_results,
            'strategy': strategy,
            'results': [],
            'status': 'collecting',
            'created_at': datetime.now(timezone.utc),
            'timeout_at': datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
        }

        self._result_collectors[collector_id] = collector_data
        self._aggregation_strategies[collector_id] = strategy

        self._logger.info(f"Created result collector {collector_id} expecting {expected_results} results")
        return True

    def submit_result(self, collector_id: str, agent_id: str, result: Any,
                     confidence: float = 1.0, metadata: Optional[Dict] = None) -> bool:
        """Submit a result to a collector."""
        if collector_id not in self._result_collectors:
            return False

        collector = self._result_collectors[collector_id]

        if collector['status'] != 'collecting':
            return False

        result_data = {
            'agent_id': agent_id,
            'result': result,
            'confidence': confidence,
            'metadata': metadata or {},
            'submitted_at': datetime.now(timezone.utc)
        }

        collector['results'].append(result_data)

        # Check if we have enough results
        if len(collector['results']) >= collector['expected_results']:
            collector['status'] = 'complete'
            aggregated_result = self._aggregate_results(collector_id)
            collector['aggregated_result'] = aggregated_result

            self._logger.info(f"Result collector {collector_id} completed with {len(collector['results'])} results")

        return True

    def _aggregate_results(self, collector_id: str) -> Any:
        """Aggregate results based on the specified strategy."""
        if collector_id not in self._result_collectors:
            return None

        collector = self._result_collectors[collector_id]
        results = collector['results']
        strategy = collector['strategy']

        if not results:
            return None

        if strategy == ResultAggregationStrategy.FIRST_WINS:
            return results[0]['result']

        elif strategy == ResultAggregationStrategy.BEST_CONFIDENCE:
            best_result = max(results, key=lambda r: r['confidence'])
            return best_result['result']

        elif strategy == ResultAggregationStrategy.MAJORITY_VOTE:
            # Simple majority vote for identical results
            result_counts = {}
            for result_data in results:
                result_str = str(result_data['result'])
                result_counts[result_str] = result_counts.get(result_str, 0) + 1

            if result_counts:
                majority_result_str = max(result_counts.items(), key=lambda x: x[1])[0]
                # Find the original result object
                for result_data in results:
                    if str(result_data['result']) == majority_result_str:
                        return result_data['result']

        elif strategy == ResultAggregationStrategy.WEIGHTED_AVERAGE:
            # For numeric results only
            try:
                total_weight = sum(r['confidence'] for r in results)
                if total_weight > 0:
                    weighted_sum = sum(r['result'] * r['confidence'] for r in results)
                    return weighted_sum / total_weight
            except (TypeError, ValueError):
                pass

        elif strategy == ResultAggregationStrategy.MERGE_ALL:
            return [r['result'] for r in results]

        # Default: return first result
        return results[0]['result']

    def get_aggregated_result(self, collector_id: str) -> Optional[Any]:
        """Get the aggregated result from a collector."""
        if collector_id not in self._result_collectors:
            return None

        collector = self._result_collectors[collector_id]
        return collector.get('aggregated_result')

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get comprehensive communication framework statistics."""
        channel_stats = {}
        for channel_id, channel in self._channels.items():
            channel_stats[channel_id] = channel.get_channel_stats()

        return {
            "channels": channel_stats,
            "coordination_metrics": self._coordination_metrics.copy(),
            "active_barriers": len([b for b in self._barriers.values() if b['status'] == 'waiting']),
            "active_elections": len([e for e in self._elections.values() if e['status'] == 'active']),
            "active_collectors": len([c for c in self._result_collectors.values() if c['status'] == 'collecting']),
            "shared_state_keys": len(self._shared_state_manager.get_state_snapshot()),
            "active_locks": len(self._shared_state_manager.get_lock_status())
        }


# Global communication framework instance
_communication_framework: Optional[AgentCommunicationFramework] = None


async def get_communication_framework() -> AgentCommunicationFramework:
    """Get or create the global communication framework instance."""
    global _communication_framework
    
    if _communication_framework is None:
        _communication_framework = AgentCommunicationFramework()
        await _communication_framework.initialize()
    
    return _communication_framework


async def close_communication_framework() -> None:
    """Close the global communication framework instance."""
    global _communication_framework
    
    if _communication_framework:
        await _communication_framework.shutdown()
        _communication_framework = None
