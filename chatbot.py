from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv 

load_dotenv() 
# 🔑 Add your API key here
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key="OPERNIAI_API_KEY"
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are {assistant_name}, a helpful and friendly AI assistant.
Your personality: {personality}
Always be concise and clear in your responses."""),

    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

chain = prompt | llm

# Chat loop (modified for Colab)
chat_history = []

config = {
    "assistant_name": "Aria",
    "personality": "witty, slightly humorous, helpful and friendly"
}

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Aria: Goodbye 👋")
        break

    response = chain.invoke({
        **config,
        "chat_history": chat_history,
        "user_input": user_input
    })

    print("Aria:", response.content)

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))