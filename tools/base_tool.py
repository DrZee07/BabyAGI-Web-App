"""
Base tool interface for the BabyAGI tool system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    status: ToolStatus
    data: Any
    message: str = ""
    metadata: Dict[str, Any] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str, config: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.config = config or {}
        self.usage_count = 0
        self.success_count = 0
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool."""
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters before execution."""
        required_params = self.get_parameters().get("required", [])
        return all(param in kwargs for param in required_params)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool."""
        success_rate = self.success_count / self.usage_count if self.usage_count > 0 else 0
        return {
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": success_rate
        }
    
    def _update_stats(self, success: bool):
        """Update usage statistics."""
        self.usage_count += 1
        if success:
            self.success_count += 1

