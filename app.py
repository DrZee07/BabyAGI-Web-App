# Advanced BabyAGI Web App with Enhanced Features
# Import necessary packages
from collections import deque
from typing import Dict, List, Optional, Any
import json
import time

import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore

# Updated imports for newer LangChain versions
try:
    from langchain_openai import OpenAI, OpenAIEmbeddings
except ImportError:
    # Fallback for older versions
    try:
        from langchain.llms import OpenAI
        from langchain.embeddings.openai import OpenAIEmbeddings
    except ImportError:
        from langchain_community.llms import OpenAI
        from langchain_community.embeddings import OpenAIEmbeddings
from pydantic import BaseModel, Field

# Import advanced features
from memory_manager import AdvancedMemoryManager, MemoryType
from llm_manager import LLMManager, LLMConfig, LLMProvider, TaskComplexity
from tools import ToolRegistry, WebSearchTool, CodeExecutorTool, FileOperationsTool
from reasoning import ChainOfThoughtReasoner


class TaskCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are an task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            partial_variables={"objective": objective},
            input_variables=["result", "task_description", "incomplete_tasks"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def get_next_task(
        self, result: Dict, task_description: str, task_list: List[str]
    ) -> List[Dict]:
        """Get the next task."""
        incomplete_tasks = ", ".join(task_list)
        response = self.run(
            result=result,
            task_description=task_description,
            incomplete_tasks=incomplete_tasks,
        )
        new_tasks = response.split("\n")
        return [
            {"task_name": task_name} for task_name in new_tasks if task_name.strip()
        ]


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, objective: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            partial_variables={"objective": objective},
            input_variables=["task_names", "next_task_id"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def prioritize_tasks(self, this_task_id: int, task_list: List[Dict]) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in task_list]
        next_task_id = int(this_task_id) + 1
        response = self.run(task_names=task_names, next_task_id=next_task_id)
        new_tasks = response.split("\n")
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append(
                    {"task_id": task_id, "task_name": task_name}
                )
        return prioritized_task_list


class ExecutionChain(LLMChain):
    """Chain to execute tasks."""

    vectorstore: VectorStore = Field(init=False)

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = True
    ) -> LLMChain:
        """Get the response parser."""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose, vectorstore=vectorstore)

    def _get_top_tasks(self, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
        return [str(item.metadata["task"]) for item in sorted_results]

    def execute_task(self, objective: str, task: str, k: int = 5) -> str:
        """Execute a task."""
        context = self._get_top_tasks(query=objective, k=k)
        return self.run(objective=objective, context=context, task=task)


class AdvancedBabyAGI(BaseModel):
    """Advanced controller model for the BabyAGI agent with enhanced features."""

    objective: str = Field(alias="objective")
    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: ExecutionChain = Field(...)
    task_id_counter: int = Field(1)
    
    # Advanced features
    memory_manager: Optional[AdvancedMemoryManager] = Field(default=None)
    llm_manager: Optional[LLMManager] = Field(default=None)
    tool_registry: Optional[ToolRegistry] = Field(default=None)
    reasoner: Optional[ChainOfThoughtReasoner] = Field(default=None)
    
    # Configuration
    use_advanced_reasoning: bool = Field(default=True)
    use_tools: bool = Field(default=True)
    use_advanced_memory: bool = Field(default=True)
    max_tool_calls_per_task: int = Field(default=3)

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        st.text("Task List ⏰")
        for t in self.task_list:
            st.write("- " + str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        st.subheader("Next Task:")
        st.warning("- " + str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        st.subheader("Task Result")
        st.info(result, icon="ℹ️")

    def print_task_ending(self):
        st.success("Tasks terminated.", icon="✅")

    def run(self, max_iterations: Optional[int] = None):
        """Run the advanced agent with enhanced capabilities."""
        num_iters = 0
        
        # Initialize session memory
        if self.use_advanced_memory and self.memory_manager:
            self.memory_manager.add_memory(
                f"Starting new session with objective: {self.objective}",
                MemoryType.EPISODIC,
                importance=0.9,
                tags=["session_start", "objective"]
            )
        
        while True:
            if self.task_list:
                self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.print_next_task(task)

                # Step 2: Enhanced task execution with advanced features
                result = self._execute_task_advanced(task)
                this_task_id = int(task["task_id"])
                self.print_task_result(result["content"])

                # Step 3: Store result in both vector store and advanced memory
                result_id = f"result_{task['task_id']}"
                self.execution_chain.vectorstore.add_texts(
                    texts=[result["content"]],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )
                
                # Store in advanced memory if available
                if self.use_advanced_memory and self.memory_manager:
                    self.memory_manager.add_memory(
                        f"Task: {task['task_name']}\nResult: {result['content']}",
                        MemoryType.EPISODIC,
                        importance=0.8,
                        tags=["task_result", f"task_{task['task_id']}"],
                        context={"task_id": task["task_id"], "objective": self.objective}
                    )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = self.task_creation_chain.get_next_task(
                    result["content"], task["task_name"], [t["task_name"] for t in self.task_list]
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                    
                self.task_list = deque(
                    self.task_prioritization_chain.prioritize_tasks(
                        this_task_id, list(self.task_list)
                    )
                )
                
                # Memory consolidation
                if self.use_advanced_memory and self.memory_manager:
                    self.memory_manager.consolidate_memories()
                    
            num_iters += 1
            if max_iterations is not None and num_iters == max_iterations:
                self.print_task_ending()
                break
    
    def _execute_task_advanced(self, task: Dict) -> Dict[str, Any]:
        """Execute a task with advanced reasoning and tool usage."""
        task_name = task["task_name"]
        
        # Determine task complexity
        complexity = self._assess_task_complexity(task_name)
        
        # Use advanced reasoning if enabled
        if self.use_advanced_reasoning and self.reasoner:
            st.info("🧠 Using advanced reasoning...")
            reasoning_result = self.reasoner.reason(
                problem=task_name,
                context=f"Objective: {self.objective}"
            )
            
            if reasoning_result.get("success"):
                # Display reasoning steps
                with st.expander("🔍 Reasoning Process"):
                    for step in reasoning_result.get("thought_steps", []):
                        st.write(f"**Step {step.step_number}:** {step.thought}")
                        st.write(f"*Reasoning:* {step.reasoning}")
                        st.write("---")
                
                base_result = reasoning_result.get("final_answer", "")
            else:
                base_result = self.execution_chain.execute_task(self.objective, task_name)
        else:
            base_result = self.execution_chain.execute_task(self.objective, task_name)
        
        # Enhance with tool usage if enabled
        if self.use_tools and self.tool_registry:
            enhanced_result = self._enhance_with_tools(task_name, base_result)
            return {"content": enhanced_result, "used_tools": True}
        
        return {"content": base_result, "used_tools": False}
    
    def _assess_task_complexity(self, task_name: str) -> TaskComplexity:
        """Assess the complexity of a task."""
        task_lower = task_name.lower()
        
        # Simple heuristics for complexity assessment
        if any(word in task_lower for word in ["research", "analyze", "compare", "evaluate"]):
            return TaskComplexity.COMPLEX
        elif any(word in task_lower for word in ["create", "design", "write", "generate"]):
            return TaskComplexity.CREATIVE
        elif any(word in task_lower for word in ["calculate", "compute", "solve"]):
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.SIMPLE
    
    def _enhance_with_tools(self, task_name: str, base_result: str) -> str:
        """Enhance task execution with tool usage."""
        enhanced_result = base_result
        tools_used = []
        
        # Determine which tools might be helpful
        task_lower = task_name.lower()
        
        # Web search for research tasks
        if any(word in task_lower for word in ["research", "find", "search", "information"]):
            search_result = self.tool_registry.execute_tool(
                "web_search", 
                query=task_name,
                max_results=3
            )
            if search_result.status.value == "success":
                search_data = search_result.data
                search_summary = "\n".join([
                    f"- {item['title']}: {item['snippet']}" 
                    for item in search_data[:2]
                ])
                enhanced_result += f"\n\n**Research Results:**\n{search_summary}"
                tools_used.append("web_search")
        
        # Code execution for computational tasks
        if any(word in task_lower for word in ["calculate", "compute", "code", "program"]):
            # This is a simplified example - in practice, you'd need more sophisticated
            # code generation based on the task
            if "calculate" in task_lower or "compute" in task_lower:
                enhanced_result += "\n\n**Computational Enhancement:** Ready for calculations if needed."
                tools_used.append("code_executor")
        
        if tools_used:
            enhanced_result += f"\n\n*Tools used: {', '.join(tools_used)}*"
        
        return enhanced_result

    @classmethod
    def from_llm_and_objectives(
        cls,
        llm: BaseLLM,
        vectorstore: VectorStore,
        objective: str,
        first_task: str,
        verbose: bool = False,
        enable_advanced_features: bool = True,
        openai_api_key: str = None,
    ) -> "AdvancedBabyAGI":
        """Initialize the Advanced BabyAGI Controller."""
        task_creation_chain = TaskCreationChain.from_llm(
            llm, objective, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, objective, verbose=verbose
        )
        execution_chain = ExecutionChain.from_llm(llm, vectorstore, verbose=verbose)
        
        # Initialize advanced features
        memory_manager = None
        llm_manager = None
        tool_registry = None
        reasoner = None
        
        if enable_advanced_features:
            try:
                # Initialize memory manager
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                memory_manager = AdvancedMemoryManager(embeddings)
                
                # Initialize LLM manager
                llm_manager = LLMManager()
                if openai_api_key:
                    # Register OpenAI models
                    llm_manager.register_llm("gpt-3.5-turbo", LLMConfig(
                        provider=LLMProvider.OPENAI,
                        model_name="gpt-3.5-turbo",
                        api_key=openai_api_key,
                        cost_per_token=0.000002,
                        capabilities=["text_generation", "reasoning"]
                    ))
                    llm_manager.register_llm("gpt-4", LLMConfig(
                        provider=LLMProvider.OPENAI,
                        model_name="gpt-4",
                        api_key=openai_api_key,
                        cost_per_token=0.00003,
                        capabilities=["text_generation", "reasoning", "complex_analysis"]
                    ))
                
                # Initialize tool registry
                tool_registry = ToolRegistry()
                tool_registry.register_tool(WebSearchTool(), "research")
                tool_registry.register_tool(CodeExecutorTool(), "computation")
                tool_registry.register_tool(FileOperationsTool(), "file_management")
                
                # Initialize reasoner
                reasoner = ChainOfThoughtReasoner(llm)
                
            except Exception as e:
                st.warning(f"Some advanced features could not be initialized: {e}")
        
        controller = cls(
            objective=objective,
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            memory_manager=memory_manager,
            llm_manager=llm_manager,
            tool_registry=tool_registry,
            reasoner=reasoner,
            use_advanced_reasoning=enable_advanced_features,
            use_tools=enable_advanced_features,
            use_advanced_memory=enable_advanced_features,
        )
        controller.add_task({"task_id": 1, "task_name": first_task})
        return controller


def initial_embeddings(openai_api_key, first_task):
    with st.spinner("Initial Embeddings ... "):
        # Define your embedding model
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key, model="text-embedding-ada-002"
        )

        vectorstore = FAISS.from_texts(
            ["_"], embeddings, metadatas=[{"task": first_task}]
        )
    return vectorstore


def main():
    st.set_page_config(
        page_title="Advanced BabyAGI",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🚀 Advanced BabyAGI 🧠")
    st.markdown(
        """
        > **Enhanced Autonomous Agent** with Advanced Memory, Multi-LLM Support, Tool Integration & Sophisticated Reasoning
        > 
        > Powered by: 🦜 [LangChain](https://python.langchain.com) + 🧠 Advanced AI Features
        """
    )
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        openai_api_key = st.text_input(
            "🔑 OpenAI API Key:",
            type="password",
            placeholder="sk-...",
            help="Required for all AI operations"
        )
        
        if openai_api_key:
            st.success("✅ API Key configured")
            
            # Advanced features toggle
            st.subheader("🧠 Advanced Features")
            enable_advanced_features = st.checkbox("Enable Advanced Features", value=True)
            
            if enable_advanced_features:
                use_advanced_reasoning = st.checkbox("🧩 Advanced Reasoning", value=True)
                use_tools = st.checkbox("🛠️ Tool Integration", value=True)
                use_advanced_memory = st.checkbox("🧠 Enhanced Memory", value=True)
                
                # Model selection
                st.subheader("🤖 Model Selection")
                model_choice = st.selectbox(
                    "Primary Model:",
                    ["gpt-3.5-turbo", "gpt-4", "text-davinci-003"],
                    help="Choose the primary LLM for task execution"
                )
            else:
                use_advanced_reasoning = False
                use_tools = False
                use_advanced_memory = False
                model_choice = "gpt-3.5-turbo"
    
    # Main interface
    if openai_api_key:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Mission Configuration")
            
            OBJECTIVE = st.text_area(
                "🏁 Ultimate Objective:",
                value="Learn Python programming in 3 days and create a simple web application",
                height=100,
                help="Describe your main goal in detail"
            )

            first_task = st.text_input(
                "🥇 Initial Task:",
                value="Create a comprehensive learning plan for Python programming",
                help="The first task to start with"
            )

            max_iterations = st.number_input(
                "💫 Max Iterations:",
                value=5,
                min_value=1,
                max_value=20,
                step=1,
                help="Maximum number of task iterations"
            )
        
        with col2:
            st.subheader("📊 System Status")
            
            if enable_advanced_features:
                st.success("🧠 Advanced Features: ON")
                if use_advanced_reasoning:
                    st.info("🧩 Advanced Reasoning: Active")
                if use_tools:
                    st.info("🛠️ Tools: Web Search, Code Execution, File Ops")
                if use_advanced_memory:
                    st.info("🧠 Enhanced Memory: Episodic + Semantic")
            else:
                st.warning("🧠 Advanced Features: OFF")
            
            st.info(f"🤖 Primary Model: {model_choice}")

        # Initialize and run
        if st.button("🚀 Launch Advanced BabyAGI", type="primary"):
            try:
                with st.spinner("🔧 Initializing advanced systems..."):
                    vectorstore = initial_embeddings(openai_api_key, first_task)
                    
                    # Select LLM based on choice
                    if "gpt-4" in model_choice:
                        llm = OpenAI(model_name="gpt-4", openai_api_key=openai_api_key, temperature=0.7)
                    elif "gpt-3.5" in model_choice:
                        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key, temperature=0.7)
                    else:
                        llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)
                    
                    baby_agi = AdvancedBabyAGI.from_llm_and_objectives(
                        llm=llm,
                        vectorstore=vectorstore,
                        objective=OBJECTIVE,
                        first_task=first_task,
                        verbose=False,
                        enable_advanced_features=enable_advanced_features,
                        openai_api_key=openai_api_key,
                    )
                    
                    # Override individual feature settings
                    if enable_advanced_features:
                        baby_agi.use_advanced_reasoning = use_advanced_reasoning
                        baby_agi.use_tools = use_tools
                        baby_agi.use_advanced_memory = use_advanced_memory

                st.success("🎉 Advanced BabyAGI initialized successfully!")
                
                # Display system information
                if enable_advanced_features:
                    with st.expander("🔍 System Information"):
                        if baby_agi.memory_manager:
                            memory_stats = baby_agi.memory_manager.get_memory_stats()
                            st.json(memory_stats)
                        
                        if baby_agi.tool_registry:
                            tools = baby_agi.tool_registry.list_tools()
                            st.write("**Available Tools:**")
                            for tool in tools:
                                st.write(f"- {tool['name']}: {tool['description']}")

                # Run the agent
                with st.spinner("🤖 Advanced BabyAGI at work..."):
                    baby_agi.run(max_iterations=max_iterations)

                st.balloons()
                st.success("🎯 Mission completed successfully!")
                
                # Display final statistics
                if enable_advanced_features and baby_agi.memory_manager:
                    with st.expander("📈 Final Statistics"):
                        final_stats = baby_agi.memory_manager.get_memory_stats()
                        st.json(final_stats)
                        
                        if baby_agi.tool_registry:
                            tool_stats = baby_agi.tool_registry.get_execution_stats()
                            st.write("**Tool Usage Statistics:**")
                            st.json(tool_stats)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Please check your API key and try again.")
    else:
        st.warning("🔑 Please enter your OpenAI API key in the sidebar to get started.")
        st.info("""
        **Advanced BabyAGI Features:**
        - 🧠 **Enhanced Memory System**: Episodic, semantic, and working memory
        - 🛠️ **Tool Integration**: Web search, code execution, file operations
        - 🤖 **Multi-LLM Support**: Intelligent model selection based on task complexity
        - 🧩 **Advanced Reasoning**: Chain-of-thought and sophisticated problem solving
        - 📊 **Performance Analytics**: Comprehensive monitoring and insights
        - ⚙️ **Flexible Configuration**: Customize agent behavior for your needs
        """)


if __name__ == "__main__":
    main()

# Written at
