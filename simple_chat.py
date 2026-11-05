from langchain.chat_models import init_chat_model

from langchain.messages import HumanMessage, SystemMessage, AIMessage


model = init_chat_model(
    model ='ollama:llama3.2',
    temperature=0
)

conversation = [
    SystemMessage('You are helpful Assistant'),
    HumanMessage('why is python?'),
    AIMessage('python available to install'),
    HumanMessage('give me website')
]



for chunk in model.stream(conversation):
    print(chunk.text, end='', flush=True)

# response = model.invoke(conversation)

# response = model.invoke('Hello, who is python?')

# print(response.content)