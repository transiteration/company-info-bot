from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langgraph.graph import StateGraph, END

load_dotenv()

# This is the state for the workflow
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    query_type: str
    transform_attempts: int

# Used for grading documents
class GraderOutput(BaseModel):
    score: str

# Used for query classification
class QueryTypeOutput(BaseModel):
    query_type: str

# Set up LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Connect to Qdrant vector DB
qdrant_client = Qdrant.from_existing_collection(
    path=None,
    url="http://localhost:6333",
    collection_name="company_docs_collection",
    embedding=embeddings,
)
retriever = qdrant_client.as_retriever(
        search_kwargs={"k": 5}
)

MAX_TRANSFORM_ATTEMPTS = 1

def classify_query(state: GraphState) -> dict:
    # Classifies the user question type
    print("---NODE: CLASSIFY QUERY---")
    question = state["question"]
    
    prompt_template = ChatPromptTemplate.from_template(
        """You are a query classifier. Classify the user's input into one of the following categories:
        - "informational": The user is asking a question that requires information retrieval (e.g., about company policies, projects, history).
        - "greeting": The user is saying hello or a similar greeting.
        - "chit_chat": The user is making a general conversational statement or asking a question not related to company information (e.g., "how are you?", "what's the weather?").

        Provide the classification as a JSON with a single key 'query_type' and no preamble or explanation.
        User input: {question}"""
    )
    
    structured_llm_classifier = llm.with_structured_output(QueryTypeOutput, method="json_mode", include_raw=False)
    classifier_chain = prompt_template | structured_llm_classifier
    
    classification_result = classifier_chain.invoke({"question": question})
    query_type = classification_result.query_type
    print(f"---CLASSIFICATION: {query_type}---")
    
    # Initialize documents as empty list for the new flow
    return {
        "query_type": query_type, 
        "question": question, 
        "transform_attempts": state.get("transform_attempts", 0),
        "documents": []
    }

def handle_greeting_or_chit_chat(state: GraphState) -> dict:
    # Handles greetings and chit-chat
    print("---NODE: HANDLE GREETING/CHIT-CHAT---")
    query_type = state["query_type"]

    if query_type == "greeting":
        response_text = "Hello there! I'm doing well, thank you. My main role is to provide information about the company. How can I assist you with company-related questions today?"
    elif query_type == "chit_chat":
        response_text = "I'm primarily designed to help with questions about our company's policies, projects, or history. Do you have any questions related to those topics?"
    else:
        response_text = "I can help with company information. What would you like to know?"
        
    return {"generation": response_text, "question": state["question"]}

def retrieve_documents(state):
    # Retrieves documents from vector DB
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} documents.")
    return {"documents": documents, "question": question}

def grade_documents(state):
    # Grades if documents are relevant to the question
    print("---NODE: GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keywords related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

        Retrieved document:
        \n\n {document} \n\n
        User question: {question}"""
    )

    structured_llm_grader = llm.with_structured_output(
        GraderOutput,
        method="json_mode",
        include_raw=False
    )

    grader_chain = prompt | structured_llm_grader

    filtered_docs = []
    for d in documents:
        response = grader_chain.invoke({"question": question, "document": d.page_content})
        grade = response.score 
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print(f"---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}

def generate(state):
    # Generates the final answer using LLM
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]

    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks for a company.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        Question: {question}
        Context: {context}
        Answer:"""
    )

    context = "\n".join([d.page_content for d in documents])
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def transform_query(state: GraphState) -> dict:
    # Rewrites the question to improve retrieval
    print("---NODE: TRANSFORM QUERY---")
    question = state["question"]

    prompt = ChatPromptTemplate.from_template(
        """You are a query transformation expert. Your task is to rewrite a user's question to be more effective for a vector database search.
        The original query might be conversational or vague. Reformulate it into a question that is more likely to retrieve relevant documents about company policies, projects, or history.

        Original question: {question}
        Rewritten question:"""
    )

    rewriter = prompt | llm | StrOutputParser()
    better_question = rewriter.invoke({"question": question})
    print(f"Rewritten question: {better_question}")
    
    new_attempts = state.get("transform_attempts", 0) + 1
    # Return documents as empty list because the next step is retrieval
    return {"documents": [], "question": better_question, "transform_attempts": new_attempts}

def decide_to_generate(state: GraphState) -> str:
    # Decides if we should generate or try to transform query again
    print("---CONDITIONAL EDGE: DECIDE TO GENERATE---")
    documents = state["documents"]
    current_attempts = state.get("transform_attempts", 0)

    if not documents:
        if current_attempts < MAX_TRANSFORM_ATTEMPTS:
            print(f"---DECISION: REWRITE QUERY (Attempt {current_attempts + 1} of {MAX_TRANSFORM_ATTEMPTS})---")
            return "transform_query"
        else:
            print(f"---DECISION: MAX REWRITE ATTEMPTS REACHED ({current_attempts}), GENERATE ANYWAY---")
            return "generate" 
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def decide_query_type(state: GraphState) -> str:
    # Decides which branch to take based on query type
    print(f"---CONDITIONAL EDGE: ROUTE QUERY TYPE ({state['query_type']})---")
    if state["query_type"] == "informational":
        return "retrieve"
    else:
        return "handle_greeting_or_chit_chat"

# Build the workflow graph
workflow = StateGraph(GraphState)

workflow.add_node("classify_query", classify_query)
workflow.add_node("handle_greeting_or_chit_chat", handle_greeting_or_chit_chat)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.set_entry_point("classify_query")

workflow.add_conditional_edges(
    "classify_query",
    decide_query_type,
    {
        "retrieve": "retrieve",
        "handle_greeting_or_chit_chat": "handle_greeting_or_chit_chat"
    }
)
workflow.add_edge("handle_greeting_or_chit_chat", END)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()
print("Graph compiled successfully")