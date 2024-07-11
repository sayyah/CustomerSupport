from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing import List, Dict, Annotated, Any, Optional, Sequence
from langgraph.checkpoint.base import empty_checkpoint
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, AnyMessage, ToolCall
from langchain_core.tools import BaseTool
from datetime import datetime
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

from PyPackage.database import Database
from PyPackage.fligts import Fligts
from PyPackage.policy import Policy
from PyPackage.carRental import CarRental
from PyPackage.hotels import Hotels
from PyPackage.excursions import Excursions
from PyPackage.utilities import create_tool_node_with_fallback, handle_tool_error, print_event


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + \
                    [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


# Haiku is faster and cheaper, but less accurate
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)
# You could swap LLMs, though you will likely want to update the prompts when
# doing so!
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4-turbo-preview")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n\n{user_info}\n"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

fligts = Fligts()
policy = Policy()
carRental = CarRental()
hotels = Hotels()
excursions = Excursions()
part_1_tools = [
    TavilySearchResults(max_results=1),
    fligts.fetch_user_flight_information,
    fligts.search_flights,
    policy.lookup_policy,
    fligts.update_ticket_to_new_flight,
    fligts.cancel_ticket,
    carRental.search_car_rentals,
    carRental.book_car_rental,
    carRental.update_car_rental,
    carRental.cancel_car_rental,
    hotels.search_hotels,
    hotels.book_hotel,
    hotels.update_hotel,
    hotels.cancel_hotel,
    excursions.search_trip_recommendations,
    excursions.book_excursion,
    excursions.update_excursion,
    excursions.cancel_excursion,
]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    part_1_tools)


SYSTEM_PROMPT_TEMPLATE = \
    """
You are a helpful Persian customer support assistant for Iran Airlines.
Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
When searching, be persistent. Expand your query bounds if the first search returns no results. 
If a search comes up empty, expand your search before giving up.

You are going to have a conversation with two users. The first user is the MAIN USER, 
who asks questions and needs to be assisted. The second user is our TOOL MANAGER, which 
runs the requested tools and delivers the tool results.

You have access to the following tools to get more information if needed:

{tool_descs}

You also have access to the history of privious messages.

Generate the response in the following json format:
{{
    "THOUGHT": "<you should always think about what to do>",
    "ACTION": "<the action to take, must be one tool_name from above tools>",
    "ACTION_PARAMS": "<the input parameters to the ACTION, it must be in json format complying with the tool_params>"
    "FINAL_ANSWER": "<a text containing the final answer to the original input question>",
}}
If you don't know the answer, you can take an action using one of the provided tools.
But if you do, don't take and action and leave the action-related attributes empty.
The values `ACTION` and `FINAL_ANSWER` can never ever be filled at the same time.
If you have any questions from the user, put that in `FINAL_ANSWER` as well.

Always make sure that your output is a json complying with above format.
Do NOT add anything before or after the json response.

Current user:\n<User>\n{user_info}\n</User>
Current time: {time}.
"""


def get_tool_description(tool: BaseTool) -> str:
    tool_params = [
        f"{name}: {info['type']} ({info['description']})"
        for name, info in tool.args.items()
    ]
    tool_params_string = ', '.join(tool_params)
    return (
        f"tool_name -> {tool.name}\n"
        f"tool_params -> {tool_params_string}\n"
        f"tool_description ->\n{tool.description}"
    )


def get_tools_description(tools: List[BaseTool]) -> str:
    return '\n\n'.join([get_tool_description(tool) for tool in tools])


class Agent:
    def run(
        self, question: str, config: Dict,
        reset_db: bool = False, clear_message_history: bool = False,
    ) -> None:
        if reset_db:
            database = Database()
            database.reset_and_prepare()

        # if clear_message_history:
        #     self._graph.checkpointer.put(config, checkpoint=empty_checkpoint())

        new_messages = []

        if clear_message_history:
            system_message = SystemMessage(SYSTEM_PROMPT_TEMPLATE.format(
                tool_descs=get_tools_description(self.tools),
                time=datetime.now(),
                user_info=f"passenger_id: {config.get('configurable', {}).get('passenger_id', None)}"
            ))
            new_messages.append(system_message)

        user_message = HumanMessage(question)
        new_messages.append(user_message)

        events = self._graph.stream(
            {'messages': new_messages}, config, stream_mode='values'
        )

        for event in events:
            print_event(event, self._printed_messages)

    def _build_graph(self) -> CompiledGraph:
        builder = StateGraph(State)
        # Define nodes: these do the work
        builder.add_node("assistant", Assistant(part_1_assistant_runnable))
        builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
        # Define edges: these determine how the control flow moves
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        # The checkpointer lets the graph persist its state
        # this is a complete memory for the entire graph.
        memory = SqliteSaver.from_conn_string(":memory:")
        return builder.compile(checkpointer=memory)
