from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
root = os.getenv("PROJECT_ROOT")

# load the system prompt from the file
with open(os.path.join(root, "prompts/system_prompt.txt"), "r", encoding="utf-8") as f:
    system_prompt = f.read()
# System message
sys_msg = SystemMessage(content=system_prompt)