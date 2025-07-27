"""
Reasoning Manager for BabyAGI
Coordinates different reasoning strategies and selects the best approach.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from langchain.llms.base import BaseLLM
from .chain_of_thought import ChainOfThoughtReasoner
from .tree_of_thought import TreeOfThoughtReasoner
from .reflection import ReflectionReasoner


class ReasoningStrategy(Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REFLECTION = "reflection"
    HYBRID = "hybrid"
    AUTO_SELECT = "auto_select"


class ReasoningManager:
    """Manages and coordinates different reasoning strategies."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.cot_reasoner = ChainOfThoughtReasoner(llm)
        self.tot_reasoner = TreeOfThoughtReasoner(llm)
        self.reflection_reasoner = ReflectionReasoner(llm)
        
        # Strategy selection criteria
        self.strategy_keywords = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: [
                "step by step", "sequential", "process", "procedure", "method"
            ],
            ReasoningStrategy.TREE_OF_THOUGHT: [
                "explore", "alternatives", "options", "possibilities", "creative", "brainstorm"
            ],
            ReasoningStrategy.REFLECTION: [
                "improve", "refine", "review", "analyze", "critique", "evaluate"
            ]
        }
    
    def reason(self, problem: str, strategy: ReasoningStrategy = ReasoningStrategy.AUTO_SELECT,
               **kwargs) -> Dict[str, Any]:
        """Apply reasoning strategy to solve a problem."""
        
        # Auto-select strategy if requested
        if strategy == ReasoningStrategy.AUTO_SELECT:
            strategy = self._select_strategy(problem)
        
        try:
            if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
                return self._apply_chain_of_thought(problem, **kwargs)
            
            elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
                return self._apply_tree_of_thought(problem, **kwargs)
            
            elif strategy == ReasoningStrategy.REFLECTION:
                return self._apply_reflection(problem, **kwargs)
            
            elif strategy == ReasoningStrategy.HYBRID:
                return self._apply_hybrid_reasoning(problem, **kwargs)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown reasoning strategy: {strategy}",
                    "problem": problem
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem": problem,
                "strategy_used": strategy.value
            }
    
    def _select_strategy(self, problem: str) -> ReasoningStrategy:
        """Automatically select the best reasoning strategy for a problem."""
        problem_lower = problem.lower()
        
        # Score each strategy based on keyword matches
        strategy_scores = {}
        
        for strategy, keywords in self.strategy_keywords.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            strategy_scores[strategy] = score
        
        # Additional heuristics
        if len(problem.split()) > 50:  # Long, complex problems
            strategy_scores[ReasoningStrategy.TREE_OF_THOUGHT] += 2
        
        if any(word in problem_lower for word in ["calculate", "compute", "solve"]):
            strategy_scores[ReasoningStrategy.CHAIN_OF_THOUGHT] += 2
        
        if any(word in problem_lower for word in ["best", "optimal", "compare"]):
            strategy_scores[ReasoningStrategy.TREE_OF_THOUGHT] += 1
        
        # Select strategy with highest score
        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            if strategy_scores[best_strategy] > 0:
                return best_strategy
        
        # Default to chain of thought
        return ReasoningStrategy.CHAIN_OF_THOUGHT
    
    def _apply_chain_of_thought(self, problem: str, **kwargs) -> Dict[str, Any]:
        """Apply chain-of-thought reasoning."""
        context = kwargs.get("context", "")
        examples = kwargs.get("examples", "")
        
        result = self.cot_reasoner.reason(problem, context, examples)
        result["strategy_used"] = ReasoningStrategy.CHAIN_OF_THOUGHT.value
        return result
    
    def _apply_tree_of_thought(self, problem: str, **kwargs) -> Dict[str, Any]:
        """Apply tree-of-thought reasoning."""
        # Reset the tree for new problem
        self.tot_reasoner.reset()
        
        result = self.tot_reasoner.reason(problem)
        result["strategy_used"] = ReasoningStrategy.TREE_OF_THOUGHT.value
        return result
    
    def _apply_reflection(self, problem: str, **kwargs) -> Dict[str, Any]:
        """Apply reflection-based reasoning."""
        initial_response = kwargs.get("initial_response")
        
        if not initial_response:
            # Generate initial response first using chain of thought
            cot_result = self.cot_reasoner.reason(problem)
            if cot_result.get("success"):
                initial_response = cot_result.get("final_answer", "")
            else:
                return {
                    "success": False,
                    "error": "Could not generate initial response for reflection",
                    "problem": problem,
                    "strategy_used": ReasoningStrategy.REFLECTION.value
                }
        
        max_iterations = kwargs.get("max_iterations", 2)
        
        if max_iterations > 1:
            result = self.reflection_reasoner.iterative_reflection(
                problem, initial_response, max_iterations
            )
        else:
            result = self.reflection_reasoner.reflect_and_improve(problem, initial_response)
        
        result["strategy_used"] = ReasoningStrategy.REFLECTION.value
        return result
    
    def _apply_hybrid_reasoning(self, problem: str, **kwargs) -> Dict[str, Any]:
        """Apply hybrid reasoning combining multiple strategies."""
        results = {}
        
        # Apply chain of thought
        cot_result = self._apply_chain_of_thought(problem, **kwargs)
        results["chain_of_thought"] = cot_result
        
        # Apply tree of thought
        tot_result = self._apply_tree_of_thought(problem, **kwargs)
        results["tree_of_thought"] = tot_result
        
        # Apply reflection on the best initial result
        best_initial = None
        if cot_result.get("success") and tot_result.get("success"):
            # Choose based on confidence or other metrics
            cot_confidence = cot_result.get("overall_confidence", 0)
            tot_path = tot_result.get("best_path", [])
            tot_confidence = sum(step.get("score", 0) for step in tot_path) / len(tot_path) if tot_path else 0
            
            if cot_confidence >= tot_confidence:
                best_initial = cot_result.get("final_answer", "")
            else:
                best_initial = tot_result.get("final_solution", "")
        elif cot_result.get("success"):
            best_initial = cot_result.get("final_answer", "")
        elif tot_result.get("success"):
            best_initial = tot_result.get("final_solution", "")
        
        if best_initial:
            reflection_result = self.reflection_reasoner.reflect_and_improve(problem, best_initial)
            results["reflection"] = reflection_result
        
        # Synthesize final answer
        final_answer = self._synthesize_hybrid_results(problem, results)
        
        return {
            "success": True,
            "problem": problem,
            "strategy_used": ReasoningStrategy.HYBRID.value,
            "individual_results": results,
            "final_answer": final_answer,
            "synthesis_method": "hybrid_combination"
        }
    
    def _synthesize_hybrid_results(self, problem: str, results: Dict[str, Any]) -> str:
        """Synthesize results from multiple reasoning strategies."""
        synthesis_parts = []
        
        # Extract key insights from each strategy
        if "chain_of_thought" in results and results["chain_of_thought"].get("success"):
            cot_answer = results["chain_of_thought"].get("final_answer", "")
            if cot_answer:
                synthesis_parts.append(f"Chain-of-thought analysis: {cot_answer}")
        
        if "tree_of_thought" in results and results["tree_of_thought"].get("success"):
            tot_answer = results["tree_of_thought"].get("final_solution", "")
            if tot_answer:
                synthesis_parts.append(f"Tree-of-thought exploration: {tot_answer}")
        
        if "reflection" in results and results["reflection"].get("success"):
            reflection_answer = results["reflection"].get("revised_response", "")
            if reflection_answer:
                synthesis_parts.append(f"Refined through reflection: {reflection_answer}")
        
        if not synthesis_parts:
            return "Unable to synthesize results from hybrid reasoning."
        
        # Create synthesis prompt
        synthesis_prompt = f"""
Problem: {problem}

I have analyzed this problem using multiple reasoning strategies:

{chr(10).join(synthesis_parts)}

Based on these different approaches, provide a comprehensive final answer that incorporates the best insights from each method:
"""
        
        try:
            return self.llm(synthesis_prompt)
        except Exception:
            # Fallback: return the best individual result
            return synthesis_parts[0] if synthesis_parts else "Synthesis failed."
    
    def compare_strategies(self, problem: str, strategies: List[ReasoningStrategy] = None) -> Dict[str, Any]:
        """Compare different reasoning strategies on the same problem."""
        if strategies is None:
            strategies = [
                ReasoningStrategy.CHAIN_OF_THOUGHT,
                ReasoningStrategy.TREE_OF_THOUGHT,
                ReasoningStrategy.REFLECTION
            ]
        
        comparison_results = {}
        
        for strategy in strategies:
            if strategy == ReasoningStrategy.REFLECTION:
                # Need initial response for reflection
                cot_result = self._apply_chain_of_thought(problem)
                if cot_result.get("success"):
                    initial_response = cot_result.get("final_answer", "")
                    result = self._apply_reflection(problem, initial_response=initial_response)
                else:
                    result = {"success": False, "error": "Could not generate initial response"}
            else:
                result = self.reason(problem, strategy)
            
            comparison_results[strategy.value] = result
        
        return {
            "problem": problem,
            "strategies_compared": [s.value for s in strategies],
            "results": comparison_results,
            "comparison_summary": self._generate_comparison_summary(comparison_results)
        }
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary comparing different reasoning strategies."""
        summary = {
            "successful_strategies": [],
            "failed_strategies": [],
            "best_strategy": None,
            "insights": []
        }
        
        for strategy, result in results.items():
            if result.get("success"):
                summary["successful_strategies"].append(strategy)
            else:
                summary["failed_strategies"].append(strategy)
        
        # Simple heuristic for best strategy (could be more sophisticated)
        if summary["successful_strategies"]:
            summary["best_strategy"] = summary["successful_strategies"][0]
        
        return summary

