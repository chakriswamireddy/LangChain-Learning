from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from langchain.agents.middleware import  dynamic_prompt, ModelRequest, ModelResponse,  wrap_model_call

from dataclasses import dataclass

from langchain.messages import AIMessage, HumanMessage, SystemMessage


@dataclass
class Context:
    user_role: str


@dynamic_prompt
def dynamicRolePrompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are witty assistant with humour tone, no techincal details"
    match user_role:
        case 'crisp':
            return 'f{base_prompt} Answer in one line'
        case _:
            return 'f{base_prompt}'


agent = create_agent(
    model='ollama:llama3.2',
    middleware=[dynamicRolePrompt],
    context_schema=Context,
    # system_prompt=system
)

response = agent.invoke(
    { 'messages': [ 
        SystemMessage("You are Witty Assistant"),
        HumanMessage("what is (1+x)/10 = 12 "),
        SystemMessage("are you asking x value ?"),
        HumanMessage("yes, solve step by step  "),

        
        
     ] },
    context=Context(user_role='crisp2')
)  

print(response["messages"][-1].content)
