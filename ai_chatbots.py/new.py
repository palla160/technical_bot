from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=400
)


class SupportState(TypedDict):
    user_message: str
    response: Optional[str]
    intent: Optional[str]
    escalate: Optional[str]

loader = PyPDFLoader("C:\\Users\\user\\Downloads\\ecommerce_support_data.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

def intent_agent(state: SupportState):
    prompt = f"""
Classify the message into:
billing, technical, banking, unknown

Message:
{state['user_message']}

Return only the category.
"""
    result = llm.invoke(prompt).content.strip().lower()
    return {"intent": result}

def faq_agent(state: SupportState):
    docs = retriever.invoke(state["user_message"])

    texts = []
    for item in docs:
        while isinstance(item, tuple):
            item = item[0]
        texts.append(item.page_content)

    context = "\n\n".join(texts)

    prompt = f"""
You are a support assistant.
Answer ONLY from the context.

Context:
{context}

Question:
{state['user_message']}
"""
    result = llm.invoke(prompt).content
    return {"response": result}

def troubleshoot_agent(state: SupportState):
    prompt = f"""
You are a technical support agent.
Guide step-by-step.

Issue:
{state['user_message']}
"""
    result = llm.invoke(prompt).content
    return {"response": result}

def escalation_agent(state: SupportState):
    prompt = f"""
Does this require a human agent?
Answer true or false.

Message:
{state['user_message']}
"""
    result = llm.invoke(prompt).content.strip().lower()
    return {"escalate": result}

def route_agent(state: SupportState):
    if state["intent"] == "technical":
        return "troubleshoot"
    elif state["intent"] in ["billing", "banking"]:
        return "faq"
    else:
        return "escalate"

graph = StateGraph(SupportState)

graph.add_node("intent", intent_agent)
graph.add_node("faq", faq_agent)
graph.add_node("troubleshoot", troubleshoot_agent)
graph.add_node("escalate", escalation_agent)

graph.add_edge(START, "intent")

graph.add_conditional_edges(
    "intent",
    route_agent,
    {
        "faq": "faq",
        "troubleshoot": "troubleshoot",
        "escalate": "escalate"
    }
)

graph.add_edge("faq", END)
graph.add_edge("troubleshoot", END)
graph.add_edge("escalate", END)

support_graph = graph.compile()

while True:
    user_input = input("Ask question: ")
    if user_input.lower() == "exit":
        break

    result = support_graph.invoke({
        "user_message": user_input
    })

    print("AI:", result.get("response", "No response"))