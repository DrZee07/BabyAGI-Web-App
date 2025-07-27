"""
Reflection Reasoning for BabyAGI
Implements self-reflection and meta-cognitive capabilities.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate


@dataclass
class ReflectionResult:
    """Result of a reflection process."""
    original_response: str
    reflection: str
    improvements: List[str]
    confidence_before: float
    confidence_after: float
    should_revise: bool


class ReflectionReasoner:
    """Implements self-reflection for improving reasoning quality."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        
        self.reflection_template = PromptTemplate(
            input_variables=["problem", "response"],
            template="""
Problem: {problem}

My initial response: {response}

Now I need to reflect on this response. Let me think critically:

1. Accuracy: Is my response factually correct and logically sound?
2. Completeness: Did I address all aspects of the problem?
3. Clarity: Is my response clear and well-structured?
4. Assumptions: What assumptions did I make? Are they valid?
5. Alternative approaches: Are there better ways to solve this?
6. Potential errors: What mistakes might I have made?

Reflection:
[Provide detailed self-reflection here]

Confidence in original response (0-10): 
Suggested improvements:
1. [Improvement 1]
2. [Improvement 2]
3. [Improvement 3]

Should I revise my response? (Yes/No):
"""
        )
        
        self.revision_template = PromptTemplate(
            input_variables=["problem", "original_response", "reflection", "improvements"],
            template="""
Problem: {problem}

Original response: {original_response}

My reflection: {reflection}

Suggested improvements: {improvements}

Based on my reflection, here is my revised and improved response:

Revised Response:
"""
        )
    
    def reflect_and_improve(self, problem: str, initial_response: str) -> Dict[str, Any]:
        """Perform reflection on an initial response and potentially improve it."""
        try:
            # Generate reflection
            reflection_result = self._generate_reflection(problem, initial_response)
            
            # Decide whether to revise
            if reflection_result.should_revise:
                revised_response = self._generate_revision(
                    problem, 
                    initial_response, 
                    reflection_result
                )
                
                return {
                    "success": True,
                    "original_response": initial_response,
                    "reflection": reflection_result.reflection,
                    "improvements": reflection_result.improvements,
                    "revised_response": revised_response,
                    "confidence_before": reflection_result.confidence_before,
                    "confidence_after": reflection_result.confidence_after,
                    "was_revised": True
                }
            else:
                return {
                    "success": True,
                    "original_response": initial_response,
                    "reflection": reflection_result.reflection,
                    "improvements": reflection_result.improvements,
                    "revised_response": initial_response,  # No revision needed
                    "confidence_before": reflection_result.confidence_before,
                    "confidence_after": reflection_result.confidence_before,
                    "was_revised": False
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_response": initial_response
            }
    
    def _generate_reflection(self, problem: str, response: str) -> ReflectionResult:
        """Generate a reflection on the given response."""
        prompt = self.reflection_template.format(
            problem=problem,
            response=response
        )
        
        reflection_response = self.llm(prompt)
        
        # Parse the reflection response
        reflection_text = ""
        confidence_before = 7.0  # Default
        improvements = []
        should_revise = False
        
        lines = reflection_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Reflection:"):
                current_section = "reflection"
                continue
            elif line.startswith("Confidence in original response"):
                # Extract confidence score
                try:
                    confidence_part = line.split(":")[-1].strip()
                    confidence_before = float(confidence_part.split()[0])
                except:
                    confidence_before = 7.0
                current_section = None
                continue
            elif line.startswith("Suggested improvements:"):
                current_section = "improvements"
                continue
            elif line.startswith("Should I revise"):
                should_revise = "yes" in line.lower()
                current_section = None
                continue
            
            # Process content based on current section
            if current_section == "reflection" and line:
                reflection_text += line + " "
            elif current_section == "improvements" and line:
                if line.startswith(("1.", "2.", "3.", "-", "•")):
                    improvement = line[2:].strip() if line[1] == '.' else line[1:].strip()
                    if improvement:
                        improvements.append(improvement)
        
        # Calculate confidence after reflection (usually slightly higher due to reflection)
        confidence_after = min(10.0, confidence_before + 0.5)
        
        return ReflectionResult(
            original_response=response,
            reflection=reflection_text.strip(),
            improvements=improvements,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            should_revise=should_revise
        )
    
    def _generate_revision(self, problem: str, original_response: str, 
                          reflection_result: ReflectionResult) -> str:
        """Generate a revised response based on reflection."""
        improvements_text = "\n".join([f"- {imp}" for imp in reflection_result.improvements])
        
        prompt = self.revision_template.format(
            problem=problem,
            original_response=original_response,
            reflection=reflection_result.reflection,
            improvements=improvements_text
        )
        
        return self.llm(prompt)
    
    def iterative_reflection(self, problem: str, initial_response: str, 
                           max_iterations: int = 3) -> Dict[str, Any]:
        """Perform multiple rounds of reflection and improvement."""
        current_response = initial_response
        reflection_history = []
        
        for iteration in range(max_iterations):
            result = self.reflect_and_improve(problem, current_response)
            
            if not result["success"]:
                break
            
            reflection_history.append({
                "iteration": iteration + 1,
                "reflection": result["reflection"],
                "improvements": result["improvements"],
                "confidence_before": result["confidence_before"],
                "confidence_after": result["confidence_after"],
                "was_revised": result["was_revised"]
            })
            
            # If no revision was made, we're done
            if not result["was_revised"]:
                break
            
            current_response = result["revised_response"]
        
        return {
            "success": True,
            "original_response": initial_response,
            "final_response": current_response,
            "reflection_history": reflection_history,
            "total_iterations": len(reflection_history),
            "final_confidence": reflection_history[-1]["confidence_after"] if reflection_history else 7.0
        }
    
    def meta_reflection(self, problem: str, solution_attempts: List[str]) -> Dict[str, Any]:
        """Perform meta-reflection on multiple solution attempts."""
        meta_template = PromptTemplate(
            input_variables=["problem", "attempts"],
            template="""
Problem: {problem}

I have made multiple attempts to solve this problem:

{attempts}

Now I need to perform meta-reflection:

1. What patterns do I see across my attempts?
2. What are the strengths and weaknesses of each approach?
3. What have I learned about this type of problem?
4. How can I improve my problem-solving process?
5. What would be the best synthesis of these approaches?

Meta-Reflection:
[Provide comprehensive meta-analysis]

Best synthesis approach:
[Describe the optimal combination of approaches]

Key learnings:
1. [Learning 1]
2. [Learning 2]
3. [Learning 3]
"""
        )
        
        attempts_text = ""
        for i, attempt in enumerate(solution_attempts, 1):
            attempts_text += f"Attempt {i}: {attempt}\n\n"
        
        prompt = meta_template.format(
            problem=problem,
            attempts=attempts_text
        )
        
        try:
            meta_response = self.llm(prompt)
            
            # Parse the meta-reflection
            sections = meta_response.split("Meta-Reflection:")
            if len(sections) > 1:
                meta_reflection = sections[1].split("Best synthesis approach:")[0].strip()
                
                synthesis_part = meta_response.split("Best synthesis approach:")
                if len(synthesis_part) > 1:
                    synthesis = synthesis_part[1].split("Key learnings:")[0].strip()
                else:
                    synthesis = "No synthesis provided"
                
                # Extract key learnings
                learnings = []
                if "Key learnings:" in meta_response:
                    learnings_section = meta_response.split("Key learnings:")[1]
                    for line in learnings_section.split('\n'):
                        line = line.strip()
                        if line.startswith(("1.", "2.", "3.", "-", "•")):
                            learning = line[2:].strip() if line[1] == '.' else line[1:].strip()
                            if learning:
                                learnings.append(learning)
            else:
                meta_reflection = meta_response
                synthesis = "No synthesis provided"
                learnings = []
            
            return {
                "success": True,
                "problem": problem,
                "meta_reflection": meta_reflection,
                "synthesis_approach": synthesis,
                "key_learnings": learnings,
                "attempts_analyzed": len(solution_attempts)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem": problem
            }

