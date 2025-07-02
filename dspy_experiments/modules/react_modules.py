"""
DSPy ReAct Agent Implementation

A DSPy-based implementation of ReAct (Reasoning + Acting) agents
that can use tools to answer questions.
"""

import dspy
from typing import List, Dict, Any, Optional
from .signatures import ToolSelection, TaskPlanning, ContextualQA


class DSPyTool:
    """Base class for DSPy tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, input_text: str, **kwargs) -> str:
        """Execute the tool with given input"""
        raise NotImplementedError("Each tool must implement execute method")
    
    def get_description(self) -> str:
        """Get tool description for the agent"""
        return f"{self.name}: {self.description}"


class DSPyDocumentSearchTool(DSPyTool):
    """Document search tool using DSPy retrievers"""
    
    def __init__(self, retriever):
        super().__init__(
            name="document_search",
            description="Search for relevant documents and passages to answer questions"
        )
        self.retriever = retriever
    
    def execute(self, input_text: str, **kwargs) -> str:
        """Search for documents related to the input"""
        try:
            result = self.retriever(input_text)
            if result.passages:
                formatted_docs = [
                    f"Source {i+1}: {passage.text[:300]}..." 
                    for i, passage in enumerate(result.passages[:3])
                ]
                return "\n---\n".join(formatted_docs)
            else:
                return "No relevant documents found."
        except Exception as e:
            return f"Search failed: {str(e)}"


class DSPyReActAgent(dspy.Module):
    """
    DSPy-based ReAct agent that uses tools to answer questions
    """
    
    def __init__(self, tools: List[DSPyTool], max_iterations: int = 5):
        super().__init__()
        
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        
        # DSPy modules for different aspects
        self.plan_task = dspy.ChainOfThought(TaskPlanning)
        self.select_tool = dspy.ChainOfThought(ToolSelection)
        self.answer_question = dspy.ChainOfThought(ContextualQA)
        
    def forward(self, question: str) -> dspy.Prediction:
        """
        Run the ReAct loop to answer the question
        """
        reasoning_steps = []
        gathered_info = []
        
        # Step 1: Create initial plan
        try:
            plan_pred = self.plan_task(question=question)
            plan = plan_pred.plan
            reasoning_steps.append(f"Plan: {plan}")
        except Exception as e:
            reasoning_steps.append(f"Planning failed: {e}")
            plan = "Search for relevant information and provide an answer."
        
        # Step 2: Iterative tool usage
        for iteration in range(self.max_iterations):
            reasoning_steps.append(f"Iteration {iteration + 1}")
            
            # Check if we have enough information
            if iteration > 0 and len(gathered_info) >= 2:
                reasoning_steps.append("Sufficient information gathered, proceeding to answer")
                break
            
            # Select appropriate tool
            try:
                available_tools_desc = "\n".join([tool.get_description() for tool in self.tools.values()])
                context = "\n".join(gathered_info) if gathered_info else "No information gathered yet"
                
                tool_pred = self.select_tool(
                    question=f"{question}\nCurrent context: {context}",
                    available_tools=available_tools_desc
                )
                
                selected_tool_name = tool_pred.selected_tool.strip()
                reasoning_steps.append(f"Selected tool: {selected_tool_name}")
                reasoning_steps.append(f"Reasoning: {tool_pred.reasoning}")
                
                # Execute selected tool
                if selected_tool_name in self.tools:
                    tool_result = self.tools[selected_tool_name].execute(question)
                    gathered_info.append(tool_result)
                    reasoning_steps.append(f"Tool result: {tool_result[:200]}...")
                else:
                    reasoning_steps.append(f"Tool '{selected_tool_name}' not found, using document_search")
                    if "document_search" in self.tools:
                        tool_result = self.tools["document_search"].execute(question)
                        gathered_info.append(tool_result)
                        reasoning_steps.append(f"Tool result: {tool_result[:200]}...")
                    
            except Exception as e:
                reasoning_steps.append(f"Tool execution failed: {e}")
                break
        
        # Step 3: Generate final answer
        context = "\n---\n".join(gathered_info) if gathered_info else "No additional context available"
        
        try:
            answer_pred = self.answer_question(context=context, question=question)
            final_answer = answer_pred.answer
        except Exception as e:
            reasoning_steps.append(f"Answer generation failed: {e}")
            final_answer = "I apologize, but I encountered an error while generating the answer."
        
        return dspy.Prediction(
            answer=final_answer,
            reasoning_steps=reasoning_steps,
            context=gathered_info,
            iterations_used=iteration + 1 if 'iteration' in locals() else 0
        )


# Test function
def test_react_agent():
    """Test the DSPy ReAct agent"""
    print("🧪 Testing DSPy ReAct Agent...")
    
    try:
        # This would need a real retriever in practice
        # For testing, we'll create a mock tool
        class MockSearchTool(DSPyTool):
            def __init__(self):
                super().__init__("mock_search", "Mock document search for testing")
            
            def execute(self, input_text: str, **kwargs) -> str:
                return f"Mock search result for: {input_text}"
        
        # Create agent with mock tool
        tools = [MockSearchTool()]
        agent = DSPyReActAgent(tools=tools, max_iterations=3)
        
        # Test with simple question
        result = agent("What is artificial intelligence?")
        
        print(f"✅ ReAct agent test completed")
        print(f"📝 Answer: {result.answer[:100]}...")
        print(f"🔄 Iterations used: {result.iterations_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ ReAct agent test failed: {e}")
        return False


if __name__ == "__main__":
    test_react_agent() 