from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ── 1. Model (LOCAL - No API Key) ───────────────────────────
llm = ChatOllama(
    model="llama3",   # make sure ollama model is installed
    temperature=0.7
)

# ── 2. Prompt Template ──────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are {assistant_name}, a helpful and friendly AI assistant.
Your personality: {personality}
Always be concise and clear in your responses."""),

    MessagesPlaceholder(variable_name="chat_history"),

    ("human", "{user_input}")
])

# ── 3. Chain ────────────────────────────────────────────────
chain = prompt | llm

# ── 4. Chat Loop ────────────────────────────────────────────
def run_chatbot():
    print("🤖 Local LangChain Chatbot (No API Key)")
    print("=" * 40)
    print("Type 'quit' to exit | 'clear' to reset chat\n")

    chat_history = []

    config = {
        "assistant_name": "Aria",
        "personality": "witty, slightly humorous, helpful and friendly"
    }

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Aria: Goodbye! 👋")
            break

        if user_input.lower() == "clear":
            chat_history = []
            print("🔄 Chat history cleared!\n")
            continue

        # Call the chain
        response = chain.invoke({
            **config,
            "chat_history": chat_history,
            "user_input": user_input
        })

        assistant_reply = response.content
        print(f"\nAria: {assistant_reply}\n")

        # Update memory
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=assistant_reply))


if __name__ == "__main__":
    run_chatbot()