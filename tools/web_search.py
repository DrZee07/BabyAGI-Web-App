"""
Web Search Tool for BabyAGI
Provides web search capabilities using multiple search engines.
"""

import time
import requests
from typing import Dict, List, Any
from .base_tool import BaseTool, ToolResult, ToolStatus


class WebSearchTool(BaseTool):
    """Tool for performing web searches."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="web_search",
            description="Search the web for information on any topic",
            config=config
        )
        self.search_engines = {
            "duckduckgo": self._duckduckgo_search,
            "serper": self._serper_search,
        }
        self.default_engine = config.get("default_engine", "duckduckgo")
        self.max_results = config.get("max_results", 10)
    
    def execute(self, query: str, engine: str = None, max_results: int = None) -> ToolResult:
        """Execute web search."""
        start_time = time.time()
        
        if not self.validate_parameters(query=query):
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message="Missing required parameter: query"
            )
        
        engine = engine or self.default_engine
        max_results = max_results or self.max_results
        
        try:
            if engine not in self.search_engines:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Unsupported search engine: {engine}"
                )
            
            results = self.search_engines[engine](query, max_results)
            execution_time = time.time() - start_time
            
            self._update_stats(True)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=results,
                message=f"Found {len(results)} results for '{query}'",
                metadata={"engine": engine, "query": query},
                execution_time=execution_time
            )
            
        except Exception as e:
            self._update_stats(False)
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Search failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform search using DuckDuckGo (simplified implementation)."""
        # This is a simplified implementation
        # In a real scenario, you'd use the duckduckgo-search library
        try:
            import duckduckgo_search
            ddg = duckduckgo_search.DDGS()
            results = []
            
            for result in ddg.text(query, max_results=max_results):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": "duckduckgo"
                })
            
            return results
            
        except ImportError:
            # Fallback to mock results if library not available
            return self._mock_search_results(query, max_results)
    
    def _serper_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform search using Serper API."""
        api_key = self.config.get("serper_api_key")
        if not api_key:
            raise ValueError("Serper API key not configured")
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": max_results
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "google"
            })
        
        return results
    
    def _mock_search_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock search results for testing."""
        return [
            {
                "title": f"Mock Result {i+1} for '{query}'",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is a mock search result snippet for query '{query}'. Result number {i+1}.",
                "source": "mock"
            }
            for i in range(min(max_results, 5))
        ]
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema."""
        return {
            "required": ["query"],
            "optional": ["engine", "max_results"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "engine": {
                    "type": "string",
                    "description": "Search engine to use",
                    "enum": list(self.search_engines.keys())
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "minimum": 1,
                    "maximum": 50
                }
            }
        }

