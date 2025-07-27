"""
Advanced Memory Management System for BabyAGI
Implements episodic, semantic, and working memory with sophisticated retrieval strategies.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore


class MemoryType(Enum):
    EPISODIC = "episodic"  # Specific experiences and events
    SEMANTIC = "semantic"  # General knowledge and facts
    WORKING = "working"    # Temporary, active information


@dataclass
class MemoryItem:
    """Represents a single memory item with metadata."""
    content: str
    memory_type: MemoryType
    timestamp: float
    importance: float  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: float = 0.0
    tags: List[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.context is None:
            self.context = {}
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp


class AdvancedMemoryManager:
    """Advanced memory management system with multiple memory types and retrieval strategies."""
    
    def __init__(self, embeddings: Embeddings, max_working_memory: int = 10):
        self.embeddings = embeddings
        self.max_working_memory = max_working_memory
        
        # Initialize different memory stores
        self.episodic_store: Optional[VectorStore] = None
        self.semantic_store: Optional[VectorStore] = None
        self.working_memory: List[MemoryItem] = []
        
        # Memory consolidation parameters
        self.consolidation_threshold = 0.7  # Importance threshold for long-term storage
        self.forgetting_curve_factor = 0.1  # Rate of importance decay
        
        # Initialize stores
        self._initialize_stores()
    
    def _initialize_stores(self):
        """Initialize the vector stores for different memory types."""
        # Create initial dummy documents to initialize FAISS stores
        dummy_texts = ["Initial memory store"]
        dummy_metadatas = [{"type": "initialization", "timestamp": time.time()}]
        
        self.episodic_store = FAISS.from_texts(
            dummy_texts, self.embeddings, metadatas=dummy_metadatas
        )
        self.semantic_store = FAISS.from_texts(
            dummy_texts, self.embeddings, metadatas=dummy_metadatas
        )
    
    def add_memory(self, content: str, memory_type: MemoryType, 
                   importance: float = 0.5, tags: List[str] = None, 
                   context: Dict[str, Any] = None) -> str:
        """Add a new memory item."""
        memory_item = MemoryItem(
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            importance=importance,
            tags=tags or [],
            context=context or {}
        )
        
        # Add to appropriate store
        if memory_type == MemoryType.WORKING:
            self._add_to_working_memory(memory_item)
        else:
            self._add_to_long_term_memory(memory_item)
        
        return f"memory_{int(memory_item.timestamp)}"
    
    def _add_to_working_memory(self, memory_item: MemoryItem):
        """Add item to working memory with size management."""
        self.working_memory.append(memory_item)
        
        # Manage working memory size
        if len(self.working_memory) > self.max_working_memory:
            # Move oldest items to long-term memory if important enough
            oldest = self.working_memory.pop(0)
            if oldest.importance >= self.consolidation_threshold:
                oldest.memory_type = MemoryType.EPISODIC
                self._add_to_long_term_memory(oldest)
    
    def _add_to_long_term_memory(self, memory_item: MemoryItem):
        """Add item to appropriate long-term memory store."""
        metadata = {
            "memory_type": memory_item.memory_type.value,
            "timestamp": memory_item.timestamp,
            "importance": memory_item.importance,
            "access_count": memory_item.access_count,
            "tags": json.dumps(memory_item.tags),
            "context": json.dumps(memory_item.context)
        }
        
        if memory_item.memory_type == MemoryType.EPISODIC:
            self.episodic_store.add_texts([memory_item.content], [metadata])
        elif memory_item.memory_type == MemoryType.SEMANTIC:
            self.semantic_store.add_texts([memory_item.content], [metadata])
    
    def retrieve_memories(self, query: str, memory_types: List[MemoryType] = None,
                         k: int = 5, importance_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using hybrid search strategy."""
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.WORKING]
        
        all_results = []
        
        # Search working memory
        if MemoryType.WORKING in memory_types:
            working_results = self._search_working_memory(query, k)
            all_results.extend(working_results)
        
        # Search long-term memory stores
        for memory_type in memory_types:
            if memory_type == MemoryType.WORKING:
                continue
                
            store = self.episodic_store if memory_type == MemoryType.EPISODIC else self.semantic_store
            if store:
                results = store.similarity_search_with_score(query, k=k)
                for doc, score in results:
                    if doc.metadata.get("importance", 0) >= importance_threshold:
                        all_results.append({
                            "content": doc.page_content,
                            "score": score,
                            "metadata": doc.metadata,
                            "memory_type": memory_type.value
                        })
        
        # Sort by relevance and importance
        all_results.sort(key=lambda x: (x["score"], x["metadata"].get("importance", 0)), reverse=True)
        
        # Update access patterns
        self._update_access_patterns([r["content"] for r in all_results[:k]])
        
        return all_results[:k]
    
    def _search_working_memory(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Search working memory using simple text matching."""
        results = []
        query_lower = query.lower()
        
        for item in self.working_memory:
            if query_lower in item.content.lower():
                results.append({
                    "content": item.content,
                    "score": 1.0,  # Simple binary relevance
                    "metadata": asdict(item),
                    "memory_type": MemoryType.WORKING.value
                })
        
        return results[:k]
    
    def _update_access_patterns(self, accessed_contents: List[str]):
        """Update access patterns for retrieved memories."""
        current_time = time.time()
        
        # Update working memory access patterns
        for item in self.working_memory:
            if item.content in accessed_contents:
                item.access_count += 1
                item.last_accessed = current_time
    
    def consolidate_memories(self):
        """Perform memory consolidation and forgetting."""
        current_time = time.time()
        
        # Apply forgetting curve to reduce importance over time
        for item in self.working_memory:
            time_diff = current_time - item.timestamp
            decay = np.exp(-self.forgetting_curve_factor * time_diff / 3600)  # Hourly decay
            item.importance *= decay
        
        # Remove very low importance items from working memory
        self.working_memory = [
            item for item in self.working_memory 
            if item.importance > 0.1
        ]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        episodic_count = len(self.episodic_store.docstore._dict) if self.episodic_store else 0
        semantic_count = len(self.semantic_store.docstore._dict) if self.semantic_store else 0
        working_count = len(self.working_memory)
        
        return {
            "episodic_memories": episodic_count,
            "semantic_memories": semantic_count,
            "working_memories": working_count,
            "total_memories": episodic_count + semantic_count + working_count,
            "working_memory_utilization": working_count / self.max_working_memory
        }
    
    def clear_working_memory(self):
        """Clear working memory (useful for new sessions)."""
        # Move important items to episodic memory
        for item in self.working_memory:
            if item.importance >= self.consolidation_threshold:
                item.memory_type = MemoryType.EPISODIC
                self._add_to_long_term_memory(item)
        
        self.working_memory.clear()
    
    def export_memories(self) -> Dict[str, Any]:
        """Export all memories for backup/analysis."""
        return {
            "working_memory": [asdict(item) for item in self.working_memory],
            "stats": self.get_memory_stats(),
            "export_timestamp": time.time()
        }

