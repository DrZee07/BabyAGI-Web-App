"""
Multi-LLM Manager for Advanced BabyAGI
Supports multiple LLM providers with intelligent model selection.
"""

import time
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
from langchain.llms.base import BaseLLM
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    CREATIVE = "creative"


@dataclass
class LLMConfig:
    """Configuration for an LLM."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    cost_per_token: float = 0.0
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


@dataclass
class LLMUsageStats:
    """Usage statistics for an LLM."""
    total_requests: int = 0
    successful_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_used: float = 0.0


class LLMManager:
    """Manager for multiple LLM providers with intelligent selection."""
    
    def __init__(self):
        self.llms: Dict[str, BaseLLM] = {}
        self.configs: Dict[str, LLMConfig] = {}
        self.stats: Dict[str, LLMUsageStats] = {}
        self.fallback_order: List[str] = []
        
        # Task complexity to model mapping
        self.complexity_preferences = {
            TaskComplexity.SIMPLE: ["gpt-3.5-turbo", "claude-instant"],
            TaskComplexity.MEDIUM: ["gpt-4", "claude-2"],
            TaskComplexity.COMPLEX: ["gpt-4", "claude-2", "gpt-4-turbo"],
            TaskComplexity.CREATIVE: ["gpt-4", "claude-2", "gpt-3.5-turbo"]
        }
    
    def register_llm(self, name: str, config: LLMConfig) -> bool:
        """Register a new LLM."""
        try:
            llm = self._create_llm_instance(config)
            self.llms[name] = llm
            self.configs[name] = config
            self.stats[name] = LLMUsageStats()
            return True
        except Exception as e:
            print(f"Failed to register LLM {name}: {e}")
            return False
    
    def _create_llm_instance(self, config: LLMConfig) -> BaseLLM:
        """Create an LLM instance based on configuration."""
        if config.provider == LLMProvider.OPENAI:
            if "gpt-3.5" in config.model_name or "gpt-4" in config.model_name:
                return ChatOpenAI(
                    model_name=config.model_name,
                    openai_api_key=config.api_key,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
            else:
                return OpenAI(
                    model_name=config.model_name,
                    openai_api_key=config.api_key,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
        
        elif config.provider == LLMProvider.ANTHROPIC:
            try:
                from langchain.chat_models import ChatAnthropic
                return ChatAnthropic(
                    model=config.model_name,
                    anthropic_api_key=config.api_key,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
            except ImportError:
                raise ImportError("Anthropic support requires: pip install anthropic")
        
        elif config.provider == LLMProvider.LOCAL:
            try:
                from langchain.llms import LlamaCpp
                return LlamaCpp(
                    model_path=config.model_name,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
            except ImportError:
                raise ImportError("Local LLM support requires: pip install llama-cpp-python")
        
        elif config.provider == LLMProvider.HUGGINGFACE:
            try:
                from langchain.llms import HuggingFacePipeline
                return HuggingFacePipeline.from_model_id(
                    model_id=config.model_name,
                    task="text-generation",
                    model_kwargs={
                        "temperature": config.temperature,
                        "max_length": config.max_tokens
                    }
                )
            except ImportError:
                raise ImportError("HuggingFace support requires: pip install transformers torch")
        
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    def select_llm(self, task_complexity: TaskComplexity = TaskComplexity.MEDIUM,
                   required_capabilities: List[str] = None,
                   max_cost: float = None) -> Optional[str]:
        """Select the best LLM for a given task."""
        if not self.llms:
            return None
        
        required_capabilities = required_capabilities or []
        candidates = []
        
        # Get preferred models for complexity
        preferred_models = self.complexity_preferences.get(task_complexity, [])
        
        for name, config in self.configs.items():
            # Check if model meets capability requirements
            if required_capabilities and not all(cap in config.capabilities for cap in required_capabilities):
                continue
            
            # Check cost constraints
            if max_cost and config.cost_per_token > max_cost:
                continue
            
            # Calculate selection score
            score = 0
            
            # Prefer models suited for task complexity
            if any(preferred in name.lower() for preferred in [m.lower() for m in preferred_models]):
                score += 10
            
            # Consider success rate
            stats = self.stats[name]
            if stats.total_requests > 0:
                success_rate = stats.successful_requests / stats.total_requests
                score += success_rate * 5
            
            # Consider response time (lower is better)
            if stats.average_response_time > 0:
                score += max(0, 5 - stats.average_response_time)
            
            # Consider cost (lower is better)
            score += max(0, 3 - config.cost_per_token * 1000)
            
            candidates.append((name, score))
        
        if not candidates:
            # Fallback to any available LLM
            return list(self.llms.keys())[0]
        
        # Sort by score and return best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def generate(self, prompt: str, llm_name: str = None,
                 task_complexity: TaskComplexity = TaskComplexity.MEDIUM,
                 **kwargs) -> Dict[str, Any]:
        """Generate text using selected or specified LLM."""
        # Select LLM if not specified
        if llm_name is None:
            llm_name = self.select_llm(task_complexity)
        
        if llm_name not in self.llms:
            return {
                "success": False,
                "error": f"LLM '{llm_name}' not found",
                "response": None
            }
        
        llm = self.llms[llm_name]
        stats = self.stats[llm_name]
        
        start_time = time.time()
        
        try:
            # Generate response
            response = llm(prompt, **kwargs)
            
            # Update statistics
            execution_time = time.time() - start_time
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.last_used = time.time()
            
            # Update average response time
            if stats.average_response_time == 0:
                stats.average_response_time = execution_time
            else:
                stats.average_response_time = (stats.average_response_time + execution_time) / 2
            
            # Estimate tokens and cost
            estimated_tokens = len(prompt.split()) + len(response.split())
            stats.total_tokens += estimated_tokens
            stats.total_cost += estimated_tokens * self.configs[llm_name].cost_per_token
            
            return {
                "success": True,
                "response": response,
                "llm_used": llm_name,
                "execution_time": execution_time,
                "estimated_tokens": estimated_tokens
            }
            
        except Exception as e:
            # Update failure statistics
            stats.total_requests += 1
            execution_time = time.time() - start_time
            
            # Try fallback if available
            if self.fallback_order and llm_name != self.fallback_order[0]:
                fallback_llm = self.fallback_order[0]
                if fallback_llm in self.llms and fallback_llm != llm_name:
                    return self.generate(prompt, fallback_llm, task_complexity, **kwargs)
            
            return {
                "success": False,
                "error": str(e),
                "response": None,
                "llm_used": llm_name,
                "execution_time": execution_time
            }
    
    def set_fallback_order(self, llm_names: List[str]):
        """Set the fallback order for LLMs."""
        self.fallback_order = [name for name in llm_names if name in self.llms]
    
    def get_llm_stats(self, llm_name: str = None) -> Dict[str, Any]:
        """Get statistics for a specific LLM or all LLMs."""
        if llm_name:
            if llm_name not in self.stats:
                return {}
            
            stats = self.stats[llm_name]
            config = self.configs[llm_name]
            
            return {
                "name": llm_name,
                "provider": config.provider.value,
                "model": config.model_name,
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "success_rate": stats.successful_requests / stats.total_requests if stats.total_requests > 0 else 0,
                "total_tokens": stats.total_tokens,
                "total_cost": stats.total_cost,
                "average_response_time": stats.average_response_time,
                "last_used": stats.last_used
            }
        else:
            return {name: self.get_llm_stats(name) for name in self.llms.keys()}
    
    def list_llms(self) -> List[Dict[str, Any]]:
        """List all registered LLMs with their configurations."""
        llm_list = []
        for name, config in self.configs.items():
            stats = self.stats[name]
            llm_list.append({
                "name": name,
                "provider": config.provider.value,
                "model": config.model_name,
                "capabilities": config.capabilities,
                "cost_per_token": config.cost_per_token,
                "success_rate": stats.successful_requests / stats.total_requests if stats.total_requests > 0 else 0,
                "total_requests": stats.total_requests
            })
        
        return llm_list
    
    def remove_llm(self, name: str) -> bool:
        """Remove an LLM from the manager."""
        if name not in self.llms:
            return False
        
        del self.llms[name]
        del self.configs[name]
        del self.stats[name]
        
        # Remove from fallback order
        if name in self.fallback_order:
            self.fallback_order.remove(name)
        
        return True
    
    def reset_stats(self, llm_name: str = None):
        """Reset statistics for a specific LLM or all LLMs."""
        if llm_name:
            if llm_name in self.stats:
                self.stats[llm_name] = LLMUsageStats()
        else:
            for name in self.stats:
                self.stats[name] = LLMUsageStats()
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get cost analysis across all LLMs."""
        total_cost = sum(stats.total_cost for stats in self.stats.values())
        total_tokens = sum(stats.total_tokens for stats in self.stats.values())
        
        cost_breakdown = {}
        for name, stats in self.stats.items():
            if stats.total_cost > 0:
                cost_breakdown[name] = {
                    "cost": stats.total_cost,
                    "tokens": stats.total_tokens,
                    "percentage": (stats.total_cost / total_cost * 100) if total_cost > 0 else 0
                }
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "average_cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0,
            "cost_breakdown": cost_breakdown
        }

