
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from langchain.agents.middleware import  dynamic_prompt, ModelRequest, ModelResponse,  wrap_model_call, AgentMiddleware, AgentState

from dataclasses import dataclass

from langchain.messages import AIMessage, HumanMessage, SystemMessage

import time



class HooksDemo(AgentMiddleware):

    def __init__(self):
        super().__init__()
        self.start_time = 0

    def before_agent(self, state: AgentState, runtime):

        self.start_time= time.time()
        print("Before Agent Triggered")

    def before_model(self, state, runtime):
        # self.start_time
        print("Before Model Triggered")

    def after_model(self, state, runtime):
        print("After Model Triggered")
    def after_agent(self, state, runtime):
        print("After Agent Triggered", time.time() - self.start_time)
        
    

agent = create_agent( "ollama:llama3.2", middleware=[HooksDemo()])


response = agent.invoke(
     { 'messages': [ 
            SystemMessage("You are Witty Assistant"),
            HumanMessage("what is (1+x)/10 = 12 "),
            SystemMessage("are you asking x value ?"),
            HumanMessage("yes, solve step by step  "),
    
            
            
         ] },
)

print(response["messages"][-1].content)
