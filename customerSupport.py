from PyPackage.database import Database
import os
import config
from PyPackage.policy import Policy
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages     
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
#from PyPackage.agent import Assistant
#from PyPackage.utilities import create_tool_node_with_fallback

os.environ["ANTHROPIC_API_KEY"] = config.ANTHROPIC_API_KEY
os.environ["TAVILY_API_KEY"] =config.TAVILY_API_KEY
os.environ["LANGCHAIN_API_KEY"] =config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Customer Support Bot Tutorial"

# from openai import OpenAI
# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )
# print(completion.choices[0].message)



if not os.path.isfile("DataBase\\travel2.sqlite"):
    Database()


# #Agent
# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
    

# #Define Graph¶
# builder = StateGraph(State)
# # Define nodes: these do the work
# builder.add_node("assistant", Assistant(part_1_assistant_runnable))
# builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# # Define edges: these determine how the control flow moves
# builder.add_edge(START, "assistant")
# builder.add_conditional_edges(
#     "assistant",
#     tools_condition,
# )
# builder.add_edge("tools", "assistant")

# # The checkpointer lets the graph persist its state
# # this is a complete memory for the entire graph.
# memory = SqliteSaver.from_conn_string(":memory:")
# part_1_graph = builder.compile(checkpointer=memory)


print("Run succefully")