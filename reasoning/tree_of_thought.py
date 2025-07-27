"""
Tree of Thought Reasoning for BabyAGI
Implements tree-based exploration of solution paths.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate


@dataclass
class ThoughtNode:
    """Represents a node in the tree of thoughts."""
    id: str
    content: str
    parent_id: Optional[str]
    children: List[str]
    score: float
    depth: int
    is_solution: bool = False
    
    def __post_init__(self):
        if not hasattr(self, 'children') or self.children is None:
            self.children = []


class TreeOfThoughtReasoner:
    """Implements tree-of-thought reasoning for complex problem solving."""
    
    def __init__(self, llm: BaseLLM, max_depth: int = 3, max_branches: int = 3):
        self.llm = llm
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.nodes: Dict[str, ThoughtNode] = {}
        
        self.generation_template = PromptTemplate(
            input_variables=["problem", "current_thought", "depth"],
            template="""
Problem: {problem}

Current thought path: {current_thought}

Generate {max_branches} different ways to continue solving this problem from the current state.
Each approach should be distinct and explore a different angle.

Depth: {depth}/{max_depth}

Format your response as:
1. [Approach 1 description]
2. [Approach 2 description]  
3. [Approach 3 description]

Make each approach specific and actionable.
"""
        )
        
        self.evaluation_template = PromptTemplate(
            input_variables=["problem", "thought_path"],
            template="""
Problem: {problem}

Thought path: {thought_path}

Evaluate this thought path on a scale of 0-10 based on:
- Relevance to the problem
- Logical consistency
- Likelihood of leading to a solution
- Creativity and insight

Provide only a numerical score (0-10) and a brief explanation.
Format: Score: X/10 - [explanation]
"""
        )
    
    def reason(self, problem: str) -> Dict[str, Any]:
        """Perform tree-of-thought reasoning on a problem."""
        try:
            # Initialize root node
            root_id = "root"
            self.nodes[root_id] = ThoughtNode(
                id=root_id,
                content=f"Starting problem: {problem}",
                parent_id=None,
                children=[],
                score=5.0,
                depth=0
            )
            
            # Build the tree
            self._build_tree(root_id, problem)
            
            # Find the best solution path
            best_path = self._find_best_path()
            
            # Generate final solution
            final_solution = self._generate_final_solution(problem, best_path)
            
            return {
                "success": True,
                "problem": problem,
                "tree_nodes": len(self.nodes),
                "best_path": best_path,
                "final_solution": final_solution,
                "tree_structure": self._get_tree_structure()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem": problem
            }
    
    def _build_tree(self, node_id: str, problem: str):
        """Recursively build the tree of thoughts."""
        current_node = self.nodes[node_id]
        
        if current_node.depth >= self.max_depth:
            return
        
        # Generate thought path up to current node
        thought_path = self._get_thought_path(node_id)
        
        # Generate new thoughts
        new_thoughts = self._generate_thoughts(problem, thought_path, current_node.depth)
        
        # Create child nodes
        for i, thought in enumerate(new_thoughts):
            child_id = f"{node_id}_child_{i}"
            
            # Evaluate the thought
            score = self._evaluate_thought(problem, thought_path + " -> " + thought)
            
            # Create child node
            child_node = ThoughtNode(
                id=child_id,
                content=thought,
                parent_id=node_id,
                children=[],
                score=score,
                depth=current_node.depth + 1
            )
            
            self.nodes[child_id] = child_node
            current_node.children.append(child_id)
            
            # Continue building if not at max depth and score is promising
            if child_node.depth < self.max_depth and score >= 6.0:
                self._build_tree(child_id, problem)
    
    def _generate_thoughts(self, problem: str, current_thought: str, depth: int) -> List[str]:
        """Generate new thoughts from current state."""
        prompt = self.generation_template.format(
            problem=problem,
            current_thought=current_thought,
            depth=depth,
            max_depth=self.max_depth,
            max_branches=self.max_branches
        )
        
        try:
            response = self.llm(prompt)
            
            # Parse the response
            thoughts = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    thought = line[2:].strip()  # Remove number prefix
                    if thought:
                        thoughts.append(thought)
            
            return thoughts[:self.max_branches]
            
        except Exception:
            # Fallback thoughts if generation fails
            return [
                f"Continue with approach A at depth {depth}",
                f"Try alternative method B at depth {depth}",
                f"Explore different angle C at depth {depth}"
            ][:self.max_branches]
    
    def _evaluate_thought(self, problem: str, thought_path: str) -> float:
        """Evaluate a thought path and return a score."""
        prompt = self.evaluation_template.format(
            problem=problem,
            thought_path=thought_path
        )
        
        try:
            response = self.llm(prompt)
            
            # Extract score from response
            if "Score:" in response:
                score_part = response.split("Score:")[1].split("-")[0].strip()
                score_str = score_part.split("/")[0].strip()
                return float(score_str)
            
            # Fallback scoring
            return 5.0
            
        except Exception:
            return 5.0  # Default score if evaluation fails
    
    def _get_thought_path(self, node_id: str) -> str:
        """Get the complete thought path from root to node."""
        path = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            if current_id != "root":  # Skip root node content
                path.append(node.content)
            current_id = node.parent_id
        
        path.reverse()
        return " -> ".join(path)
    
    def _find_best_path(self) -> List[Dict[str, Any]]:
        """Find the path with the highest cumulative score."""
        best_path = []
        best_score = -1
        
        # Find all leaf nodes
        leaf_nodes = [node for node in self.nodes.values() if not node.children]
        
        for leaf in leaf_nodes:
            path = []
            current_score = 0
            current_id = leaf.id
            
            # Build path from leaf to root
            while current_id and current_id in self.nodes:
                node = self.nodes[current_id]
                if current_id != "root":
                    path.append({
                        "id": node.id,
                        "content": node.content,
                        "score": node.score,
                        "depth": node.depth
                    })
                    current_score += node.score
                current_id = node.parent_id
            
            path.reverse()
            
            # Check if this is the best path
            avg_score = current_score / len(path) if path else 0
            if avg_score > best_score:
                best_score = avg_score
                best_path = path
        
        return best_path
    
    def _generate_final_solution(self, problem: str, best_path: List[Dict[str, Any]]) -> str:
        """Generate a final solution based on the best path."""
        if not best_path:
            return "No solution path found."
        
        path_description = " -> ".join([step["content"] for step in best_path])
        
        final_template = PromptTemplate(
            input_variables=["problem", "path"],
            template="""
Problem: {problem}

Best reasoning path found: {path}

Based on this reasoning path, provide a comprehensive final solution to the problem.
Make it clear, actionable, and complete.

Final Solution:
"""
        )
        
        prompt = final_template.format(problem=problem, path=path_description)
        
        try:
            return self.llm(prompt)
        except Exception:
            return f"Solution based on path: {path_description}"
    
    def _get_tree_structure(self) -> Dict[str, Any]:
        """Get a representation of the tree structure."""
        def build_tree_dict(node_id: str) -> Dict[str, Any]:
            if node_id not in self.nodes:
                return {}
            
            node = self.nodes[node_id]
            return {
                "id": node.id,
                "content": node.content,
                "score": node.score,
                "depth": node.depth,
                "children": [build_tree_dict(child_id) for child_id in node.children]
            }
        
        return build_tree_dict("root")
    
    def reset(self):
        """Reset the tree for a new problem."""
        self.nodes.clear()

