import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from advisor.schemas import AdvisorReportSchema, NaturalLanguageQueryResponse

class PolicyAdvisorEngine:
    def __init__(self, model_name: str = "llama3:8b-instruct-fp16", temperature: float = 0.2):
        # By default, LM Studio runs on port 1234, llama.cpp on 8080, and Ollama on 11434.
        # Setting Ollama's default port 11434 as fallback to match the rest of the project.
        local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
        
        # Connect via OpenAI-compatible endpoint (supported by LM Studio, llama.cpp, and Ollama)
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_base=local_url,
            openai_api_key="not-needed",
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def generate_executive_report(self, context_data: dict) -> AdvisorReportSchema:
        schema_json = AdvisorReportSchema.schema_json()
        system_prompt = (
            "You are an AI Policy Advisor. Your task is to analyze the provided structured JSON data "
            "from the predictive engine and generate an executive report.\n"
            "You must output valid JSON strictly matching this schema:\n"
            f"{schema_json}\n"
            "Do not output anything other than the JSON object."
        )
        human_prompt = f"Context Data: {json.dumps(context_data)}"
        
        response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        
        try:
            data = json.loads(response.content)
            return AdvisorReportSchema(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response into schema: {e}\nResponse: {response.content}")

    def query_natural_language(self, query: str, context_data: dict) -> NaturalLanguageQueryResponse:
        schema_json = NaturalLanguageQueryResponse.schema_json()
        system_prompt = (
            "You are an AI Policy Advisor. Answer the user's natural language query based ONLY on the provided context data.\n"
            "You must output valid JSON strictly matching this schema:\n"
            f"{schema_json}\n"
            "Do not output anything other than the JSON object."
        )
        human_prompt = f"Context Data: {json.dumps(context_data)}\n\nQuery: {query}"
        
        response = self.llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        
        try:
            data = json.loads(response.content)
            return NaturalLanguageQueryResponse(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response into schema: {e}\nResponse: {response.content}")

advisor_engine = PolicyAdvisorEngine()
