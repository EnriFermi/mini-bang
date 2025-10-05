import keyring
import os

from prompt import prompt
from sample_trajectory import sample_trajectory



from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

def sample_trajectory_structured(T: int, N: int) -> Dict[str, Any]:
    """
    Run generator with args T and N; returns 'N=1'..'N=N' mapping to final states.
    Replace the body with your real simulator; must return JSON-serializable data.
    """
    out = sample_trajectory({'T': T, 'N': N})
    return out

# ---- Tool schema ----
class SampleArgs(BaseModel):
    T: int = Field(ge=0, description="Horizon length")
    N: int = Field(ge=1, description="Number of simulations to run")

sample_tool = StructuredTool.from_function(
    name="sample_n_trajectories_for_param_T",
    description="Run generator with args T and N; returns 'N=1'..'N=N' mapping to final states.",
    func=sample_trajectory_structured,
    args_schema=SampleArgs,
)

# ---- LLM ----
llm = ChatOpenAI(model="o4-mini")

# ---- Fixed prompt (your text goes here verbatim) ----
FIXED_TASK_PROMPT = prompt

# ---- System prompt: force tool-first exploration, then finalize with code ----
SYSTEM_PROMPT = (
    "You are an research agent. Explore by calling the provided tool with chosen integers T and N "
    "multiple times to probe behavior. Reason about randomness by repeating runs (you may vary N). "
    "When you have enough evidence, STOP CALLING TOOLS and output ONLY the inferred algorithm as "
    "Python code or clear pseudocode, no prose. If uncertain, present the most probable algorithm "
    "with comments stating assumptions."
)

# ---- Agent wiring ----
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", FIXED_TASK_PROMPT),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, [sample_tool], prompt)
executor = AgentExecutor(
    agent=agent,
    tools=[sample_tool],
    verbose=True,
    max_iterations=20,                 # guardrail
    return_intermediate_steps=True,   # set True if you want traces
    handle_parsing_errors=True,
)

# ---- One-shot run (no user input) ----
def run_inference():
    # The agent reads FIXED_TASK_PROMPT and decides T/N; it can call the tool repeatedly.
    result = executor.invoke({"input": ""})  # no user input; everything is in the prompt
    return result["output"]

if __name__ == "__main__":
    print(run_inference())