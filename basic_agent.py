from langchain.agents import create_agent
from langchain.tools import tool


@tool('get_weather', description='Return weather info about city', return_direct=False)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"""
    Weather report for {city}:
    Condition: Sunny
    Source: get_weather tool
    This is the available weather information.
    """



system_prompt="""
You are a helpful assistant.

IMPORTANT:
When get_weather is called, its returned result is the authoritative
weather information for this task.

Never contradict the tool output.
Do not claim that weather information is unavailable
when the tool has returned a result.
"""

agent = create_agent(
    model="ollama:llama3.2",
    tools=[get_weather],
    system_prompt=system_prompt
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)

for message in result["messages"]:
    print("\nType:", type(message).__name__)
    print("Content:", message.content)
    print("Tool calls:", getattr(message, "tool_calls", None))