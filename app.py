import os
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.tools import Tool                                          # langchain_core >= 0.1
from langchain_community.agent_toolkits.load_tools import load_tools          # langchain_community >= 0.0.38
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# TODO: Import and configure your LLM, tools, and agent here.
# Use st.secrets["OPENAI_API_KEY"] and st.secrets["SERPAPI_API_KEY"]
# to read keys when running outside Colab.

# Set API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]

# Initialize LLM
llm = OpenAI(temperature=0)

# Load search tool
search_tool = load_tools(["serpapi"], llm=llm)[0]

# Compare tool
def compare_items(query: str) -> str:
    try:
        parts = [part.strip() for part in query.split(",")]
        if len(parts) < 3:
            return "Error: please provide at least two items and a category."

        items = parts[:-1]
        category = parts[-1]

        comparison_template = """
You are a helpful comparison assistant.

Compare the following items in the category: {category}

Items:
{items}

Provide a concise comparison including:
1. Key similarities
2. Key differences
3. Strengths and weaknesses of each item
4. A brief overall conclusion

Keep the response clear and relevant.
"""
        comparison_prompt = PromptTemplate(
            input_variables=["items", "category"],
            template=comparison_template
        )

        comparison_chain = LLMChain(llm=llm, prompt=comparison_prompt)
        result = comparison_chain.invoke({
            "items": "\n".join(items),
            "category": category
        })["text"]
        return result.strip()

    except Exception as e:
        return f"Error in compare_items: {str(e)}"

compare_tool = Tool(
    name="compare",
    func=compare_items,
    description=(
        "Use this tool to compare two or more items within a category. "
        "Input format must be: 'item1, item2, ..., category'. "
        "Example: 'iPhone 15 Pro, Samsung Galaxy S24 Ultra, smartphones'."
    )
)

# Analyze tool
def analyze_results(query: str) -> str:
    try:
        analysis_template = """
You are a helpful analysis assistant.

Analyze the following text and provide:
1. A concise summary
2. The most important key takeaways
3. Any notable insights or conclusions

Text:
{text}

Keep the response clear, concise, and relevant.
"""
        analysis_prompt = PromptTemplate(
            input_variables=["text"],
            template=analysis_template
        )

        analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
        result = analysis_chain.invoke({"text": query})["text"]
        return result.strip()

    except Exception as e:
        return f"Error in analyze_results: {str(e)}"

analyze_tool = Tool(
    name="analyze_results",
    func=analyze_results,
    description=(
        "Summarize and analyze text, search results, or comparison results. "
        "Use this tool when you need to extract key points, insights, or conclusions."
    )
)

# Tools list
tools = [search_tool, compare_tool, analyze_tool]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    max_iterations=8,
    handle_parsing_errors=True
)

def process_query(query: str):
    try:
        return agent.invoke({"input": query})
    except Exception as e:
        return {"output": f"Error: {str(e)}", "intermediate_steps": []}
      
st.title("ReAct Agent")
st.write("Ask a complex question and let the agent reason through it step by step.")

# TODO: Add a text input widget for the user's query.
query = st.text_input("Enter your question:")

# TODO: Add a button to submit the query.
if st.button("Submit"):
    if query:
        with st.spinner("Thinking..."):
            # TODO: Call process_query() (or agent.invoke()) and store the result.
            result = process_query(query)

        # TODO: Display the final answer.
        st.subheader("Answer")
        st.write(result.get("output", "No output returned."))

        # (Optional) Display the step-by-step reasoning trace.
        # Hint: use return_intermediate_steps=True on initialize_agent, then
        # iterate over output["intermediate_steps"] to show each action + observation.
        with st.expander("Reasoning Trace"):
            steps = result.get("intermediate_steps", [])
            if steps:
                for i, (action, observation) in enumerate(steps, 1):
                    st.markdown(f"**Step {i}**")
                    st.write(f"**Action:** {action.tool}")
                    st.write(f"**Action Input:** {action.tool_input}")
                    st.write(f"**Observation:** {observation}")
                    st.write(f"**Thought Log:** {action.log}")
                    st.markdown("---")
            else:
                st.write("No intermediate reasoning steps available.")

    else:
        st.warning("Please enter a query before submitting.")
