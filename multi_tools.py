from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver
import requests


system_prompt = """
You are a weather assistant with access to two tools.

Tool workflow:

1. Check whether the user has provided a city.
2. If no city is provided, call locate_user.
3. Wait for the locate_user tool result.
4. The tool result contains the actual city name.
5. Use that exact returned city as the city argument when calling get_weather.
6. Wait for get_weather to return its result.
7. Only then generate the final structured response.

IMPORTANT:
- Never use the tool name as a city.
- Never guess or modify the city returned by locate_user.
- Never generate a final response before get_weather executes.
- Always use actual tool calls, never write tool calls as text.
"""

@dataclass
class Context:
    userId: str


@dataclass
class ResponseFormat:
    summary:str
    fahrenheit : str
    celsius: str
    wind: str



@tool('get_weather', description="Tell the weather details based on the city")

def getWeather(city: str):
    print(f'city: {city}')

    if city.lower() in ["locate_user", "get_weather"]:
        return "Invalid city. Provide an actual city name."

    print(f"Received city: {city}")

    response = requests.get(
        f"https://wttr.in/{city}",
        params={"format": "j1"},
        timeout=15
    )

    response.raise_for_status()
    return response.json()

@tool('locate_user', description="Based on Userid found the city", return_direct=False)
def locateUser(runtime:ToolRuntime[Context]):
    print(f"user id: {runtime.context.userId}")
    match runtime.context.userId:
        case 'ABC':
            print('Vienna')
            return "Vienna"
        case 'QWE':
            return 'Canberra'
        case _:
            return "Cairo"
        



model = init_chat_model(
    model='ollama:llama3.2',
    temperature=0.1
)

checkpoint = MemorySaver()


agent = create_agent(
    model=model,
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpoint,
    tools=[locateUser, getWeather],
    system_prompt=system_prompt
    
)


config = {'configurable': {'thread_id':'1'}}

response = agent.invoke({
    'messages': [{ 'role': 'user', 'content':'What is weather like '}],
    
    },
    context=Context(userId='ABC'),
    config=config,

)


for msg in response["messages"]:
    print(type(msg).__name__, msg.content)


print(response["messages"][-1].content)
print(response.get("structured_response"))

# print(response.json)
