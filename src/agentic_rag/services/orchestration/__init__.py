"""
Agent Orchestration Framework

This module provides the core agent orchestration framework for the Agentic RAG system.
It includes base classes, interfaces, and orchestration patterns for coordinating
multiple specialized agents to handle complex queries.
"""

from .base import (
    Agent,
    AgentCapability,
    AgentStatus,
    Task,
    TaskStatus,
    TaskPriority,
    Context,
    Result,
    AgentError,
    AgentTimeoutError,
    AgentUnavailableError,
    WorkflowError,
)

from .registry import (
    AgentRegistry,
    ToolMetadata,
    AgentRegistration,
    get_agent_registry,
)

from .communication import (
    Message,
    MessageType,
    MessagePriority,
    CommunicationChannel,
    AgentCommunicationFramework,
    get_communication_framework,
)

from .workflow import (
    WorkflowStep,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowEngine,
    get_workflow_engine,
)

from .planner import (
    PlannerAgent,
    QueryAnalysis,
    ExecutionPlan,
    get_planner_agent,
)

from .orchestrator import (
    AgentOrchestrator,
    OrchestrationConfig,
    OrchestrationResult,
    get_agent_orchestrator,
)

__all__ = [
    # Base classes and types
    "Agent",
    "AgentCapability", 
    "AgentStatus",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "Context",
    "Result",
    "AgentError",
    "AgentTimeoutError",
    "AgentUnavailableError",
    "WorkflowError",
    
    # Registry
    "AgentRegistry",
    "ToolMetadata",
    "AgentRegistration",
    "get_agent_registry",
    
    # Communication
    "Message",
    "MessageType",
    "MessagePriority",
    "CommunicationChannel",
    "AgentCommunicationFramework",
    "get_communication_framework",
    
    # Workflow
    "WorkflowStep",
    "WorkflowDefinition",
    "WorkflowExecution",
    "WorkflowEngine",
    "get_workflow_engine",
    
    # Planner
    "PlannerAgent",
    "QueryAnalysis",
    "ExecutionPlan",
    "get_planner_agent",
    
    # Orchestrator
    "AgentOrchestrator",
    "OrchestrationConfig",
    "OrchestrationResult",
    "get_agent_orchestrator",
]
