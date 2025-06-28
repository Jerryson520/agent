from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent import build_graph
from langchain_core.messages import HumanMessage
import re

app = FastAPI()

class Query(BaseModel):
    question: str
    
graph = build_graph()

@app.post("/chat")
def chat(query: Query):
    messages = [HumanMessage(content=query.question)]
    messages = graph.invoke({"messages": messages})
    answer = messages["messages"][-1].content
    answer = re.findall(r"FINAL\sANSWER: (.+)", answer)
    if answer == "":
         return {"response": "Sorry, I couldn't find a relevant answer to your question."}
    else:
        return {"response": answer[0]}
    # return {"response": answer[14:]}

if __name__ == "__main__":
    pass