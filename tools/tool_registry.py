"""
Tool Registry for BabyAGI
Manages registration, discovery, and execution of tools.
"""

import time
from typing import Dict, List, Any, Optional, Type
from .base_tool import BaseTool, ToolResult, ToolStatus


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_categories: Dict[str, List[str]] = {}
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_tool(self, tool: BaseTool, category: str = "general") -> bool:
        """Register a tool in the registry."""
        if tool.name in self.tools:
            return False  # Tool already exists
        
        self.tools[tool.name] = tool
        
        if category not in self.tool_categories:
            self.tool_categories[category] = []
        self.tool_categories[category].append(tool.name)
        
        return True
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry."""
        if tool_name not in self.tools:
            return False
        
        # Remove from tools
        del self.tools[tool_name]
        
        # Remove from categories
        for category, tools in self.tool_categories.items():
            if tool_name in tools:
                tools.remove(tool_name)
                if not tools:  # Remove empty categories
                    del self.tool_categories[category]
                break
        
        return True
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def list_tools(self, category: str = None) -> List[Dict[str, Any]]:
        """List available tools, optionally filtered by category."""
        if category and category in self.tool_categories:
            tool_names = self.tool_categories[category]
        else:
            tool_names = list(self.tools.keys())
        
        tools_info = []
        for name in tool_names:
            tool = self.tools[name]
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_parameters(),
                "usage_stats": tool.get_usage_stats()
            })
        
        return tools_info
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with given parameters."""
        if tool_name not in self.tools:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Tool '{tool_name}' not found"
            )
        
        tool = self.tools[tool_name]
        
        # Validate parameters
        if not tool.validate_parameters(**kwargs):
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Invalid parameters for tool '{tool_name}'"
            )
        
        # Execute tool
        start_time = time.time()
        result = tool.execute(**kwargs)
        
        # Record execution history
        self.execution_history.append({
            "tool_name": tool_name,
            "parameters": kwargs,
            "result_status": result.status.value,
            "execution_time": result.execution_time,
            "timestamp": start_time
        })
        
        # Keep only last 1000 executions
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        return result
    
    def get_tool_suggestions(self, query: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """Get tool suggestions based on a query."""
        suggestions = []
        query_lower = query.lower()
        
        for tool_name, tool in self.tools.items():
            score = 0
            
            # Check name match
            if query_lower in tool_name.lower():
                score += 10
            
            # Check description match
            if query_lower in tool.description.lower():
                score += 5
            
            # Check parameter descriptions
            params = tool.get_parameters()
            for param_name, param_info in params.get("properties", {}).items():
                if query_lower in param_name.lower():
                    score += 3
                if "description" in param_info and query_lower in param_info["description"].lower():
                    score += 2
            
            if score > 0:
                suggestions.append({
                    "tool_name": tool_name,
                    "description": tool.description,
                    "relevance_score": score,
                    "usage_stats": tool.get_usage_stats()
                })
        
        # Sort by relevance score and usage stats
        suggestions.sort(key=lambda x: (x["relevance_score"], x["usage_stats"]["success_rate"]), reverse=True)
        
        return suggestions[:max_suggestions]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get overall execution statistics."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "most_used_tools": [],
                "recent_failures": []
            }
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h["result_status"] == "success")
        success_rate = successful_executions / total_executions
        
        avg_execution_time = sum(h["execution_time"] for h in self.execution_history) / total_executions
        
        # Most used tools
        tool_usage = {}
        for h in self.execution_history:
            tool_name = h["tool_name"]
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        most_used_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Recent failures
        recent_failures = [
            {
                "tool_name": h["tool_name"],
                "timestamp": h["timestamp"],
                "parameters": h["parameters"]
            }
            for h in self.execution_history[-50:]  # Last 50 executions
            if h["result_status"] != "success"
        ][-10:]  # Last 10 failures
        
        return {
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "most_used_tools": most_used_tools,
            "recent_failures": recent_failures
        }
    
    def get_categories(self) -> Dict[str, int]:
        """Get tool categories with counts."""
        return {category: len(tools) for category, tools in self.tool_categories.items()}
    
    def clear_history(self):
        """Clear execution history."""
        self.execution_history.clear()
    
    def export_registry_info(self) -> Dict[str, Any]:
        """Export registry information for backup/analysis."""
        return {
            "tools": {
                name: {
                    "description": tool.description,
                    "parameters": tool.get_parameters(),
                    "usage_stats": tool.get_usage_stats()
                }
                for name, tool in self.tools.items()
            },
            "categories": self.tool_categories,
            "execution_stats": self.get_execution_stats(),
            "export_timestamp": time.time()
        }

