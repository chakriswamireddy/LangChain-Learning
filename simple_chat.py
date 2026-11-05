from langchain.chat_models import init_chat_model


model = init_chat_model(
    model ='ollama:llama3.2',
    temperature=0
)

response = model.invoke('Hello, who is python?')

print(response.content)
