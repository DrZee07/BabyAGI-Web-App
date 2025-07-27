"""
Chain of Thought Reasoning for BabyAGI
Implements step-by-step reasoning with explicit thought processes.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate


@dataclass
class ThoughtStep:
    """Represents a single step in the chain of thought."""
    step_number: int
    thought: str
    reasoning: str
    confidence: float
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class ChainOfThoughtReasoner:
    """Implements chain-of-thought reasoning for complex problem solving."""
    
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.cot_template = PromptTemplate(
            input_variables=["problem", "context", "examples"],
            template="""
You are an expert problem solver. Break down the following problem step by step using chain-of-thought reasoning.

Problem: {problem}

Context: {context}

Examples of good reasoning:
{examples}

Please solve this step by step:

Step 1: [Understand the problem]
Think: What exactly is being asked? What are the key components?
Reasoning: [Your reasoning here]

Step 2: [Analyze the situation]
Think: What information do I have? What do I need to find out?
Reasoning: [Your reasoning here]

Step 3: [Develop approach]
Think: What's the best way to tackle this problem?
Reasoning: [Your reasoning here]

Step 4: [Execute solution]
Think: Let me work through the solution systematically.
Reasoning: [Your reasoning here]

Step 5: [Verify and conclude]
Think: Does this solution make sense? Is it complete?
Reasoning: [Your reasoning here]

Final Answer: [Your final answer here]
"""
        )
    
    def reason(self, problem: str, context: str = "", examples: str = "") -> Dict[str, Any]:
        """Perform chain-of-thought reasoning on a problem."""
        # Generate the reasoning chain
        prompt = self.cot_template.format(
            problem=problem,
            context=context,
            examples=examples or self._get_default_examples()
        )
        
        try:
            response = self.llm(prompt)
            
            # Parse the response into thought steps
            thought_steps = self._parse_reasoning_steps(response)
            
            # Extract final answer
            final_answer = self._extract_final_answer(response)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(thought_steps)
            
            return {
                "success": True,
                "problem": problem,
                "thought_steps": thought_steps,
                "final_answer": final_answer,
                "overall_confidence": overall_confidence,
                "raw_response": response
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem": problem
            }
    
    def _parse_reasoning_steps(self, response: str) -> List[ThoughtStep]:
        """Parse the LLM response into structured thought steps."""
        steps = []
        lines = response.split('\n')
        current_step = None
        current_thought = ""
        current_reasoning = ""
        step_number = 0
        
        for line in lines:
            line = line.strip()
            
            # Detect step headers
            if line.startswith('Step ') and ':' in line:
                # Save previous step if exists
                if current_step is not None:
                    steps.append(ThoughtStep(
                        step_number=step_number,
                        thought=current_thought.strip(),
                        reasoning=current_reasoning.strip(),
                        confidence=0.8  # Default confidence
                    ))
                
                # Start new step
                step_number += 1
                current_step = line
                current_thought = ""
                current_reasoning = ""
                
            elif line.startswith('Think:'):
                current_thought = line.replace('Think:', '').strip()
                
            elif line.startswith('Reasoning:'):
                current_reasoning = line.replace('Reasoning:', '').strip()
                
            elif current_step and not line.startswith('Final Answer:'):
                # Continue building current reasoning
                if current_reasoning:
                    current_reasoning += " " + line
                elif current_thought:
                    current_thought += " " + line
        
        # Add the last step
        if current_step is not None:
            steps.append(ThoughtStep(
                step_number=step_number,
                thought=current_thought.strip(),
                reasoning=current_reasoning.strip(),
                confidence=0.8
            ))
        
        return steps
    
    def _extract_final_answer(self, response: str) -> str:
        """Extract the final answer from the response."""
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('Final Answer:'):
                return line.replace('Final Answer:', '').strip()
        
        # If no explicit final answer, try to extract from last step
        return "No explicit final answer found"
    
    def _calculate_overall_confidence(self, thought_steps: List[ThoughtStep]) -> float:
        """Calculate overall confidence based on individual step confidences."""
        if not thought_steps:
            return 0.0
        
        # Simple average for now, could be more sophisticated
        return sum(step.confidence for step in thought_steps) / len(thought_steps)
    
    def _get_default_examples(self) -> str:
        """Get default examples for chain-of-thought reasoning."""
        return """
Example 1:
Problem: If a store has 15 apples and sells 3/5 of them, how many apples are left?
Step 1: Understand - I need to find how many apples remain after selling 3/5 of 15 apples.
Step 2: Calculate sold - 3/5 × 15 = 9 apples sold
Step 3: Calculate remaining - 15 - 9 = 6 apples left
Final Answer: 6 apples

Example 2:
Problem: Plan a birthday party for 20 people with a $200 budget.
Step 1: Understand - Need to plan party elements within budget constraints.
Step 2: Break down costs - Food (~$100), Decorations (~$50), Activities (~$50)
Step 3: Plan specifics - Pizza/cake for food, balloons/streamers for decor, games for activities
Final Answer: Pizza party with decorations and games within $200 budget.
"""
    
    def guided_reasoning(self, problem: str, guidance_questions: List[str]) -> Dict[str, Any]:
        """Perform guided chain-of-thought reasoning with specific questions."""
        guided_template = PromptTemplate(
            input_variables=["problem", "questions"],
            template="""
Problem: {problem}

Please work through this problem by answering these guiding questions step by step:

{questions}

For each question, provide:
- Your thought process
- Your reasoning
- Your answer

Then provide a final comprehensive solution.
"""
        )
        
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(guidance_questions)])
        
        prompt = guided_template.format(
            problem=problem,
            questions=questions_text
        )
        
        try:
            response = self.llm(prompt)
            
            return {
                "success": True,
                "problem": problem,
                "guidance_questions": guidance_questions,
                "guided_response": response,
                "final_answer": self._extract_final_answer(response)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem": problem
            }
    
    def verify_reasoning(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the quality of chain-of-thought reasoning."""
        if not reasoning_result.get("success"):
            return {"verification_passed": False, "issues": ["Original reasoning failed"]}
        
        verification_template = PromptTemplate(
            input_variables=["problem", "reasoning", "answer"],
            template="""
Please verify this chain-of-thought reasoning:

Original Problem: {problem}

Reasoning Steps: {reasoning}

Final Answer: {answer}

Verification Questions:
1. Is each step logically connected to the previous one?
2. Are there any gaps in the reasoning?
3. Is the final answer supported by the reasoning steps?
4. Are there any obvious errors or inconsistencies?
5. Could the reasoning be improved?

Please provide:
- Overall assessment (PASS/FAIL)
- Specific issues found (if any)
- Suggestions for improvement
"""
        )
        
        # Format reasoning steps for verification
        steps_text = ""
        for step in reasoning_result.get("thought_steps", []):
            steps_text += f"Step {step.step_number}: {step.thought}\nReasoning: {step.reasoning}\n\n"
        
        prompt = verification_template.format(
            problem=reasoning_result["problem"],
            reasoning=steps_text,
            answer=reasoning_result.get("final_answer", "")
        )
        
        try:
            verification_response = self.llm(prompt)
            
            # Simple parsing of verification result
            verification_passed = "PASS" in verification_response.upper()
            
            return {
                "verification_passed": verification_passed,
                "verification_response": verification_response,
                "original_confidence": reasoning_result.get("overall_confidence", 0.0)
            }
            
        except Exception as e:
            return {
                "verification_passed": False,
                "error": f"Verification failed: {str(e)}"
            }

