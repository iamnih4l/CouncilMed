import json
from typing import Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Define the local LLM instance. Make sure Ollama is running `llama3`
llm = ChatOllama(model="llama3:8b-instruct-fp16", temperature=0.1)

class CouncilState(TypedDict):
    # Inputs
    metadata: Dict[str, Any]
    densenet_outputs: Dict[str, Any]
    attention_outputs: Dict[str, Any]
    swin_unetr_outputs: Dict[str, Any]
    
    # Internal agent reports
    clinical_context: str
    radiologist_report: str
    neurologist_report: str
    
    # Final outputs
    discrepancy_found: bool
    final_diagnosis: str

def clinical_extractor_node(state: CouncilState):
    """Parses raw metadata into a readable clinical context."""
    system_prompt = "You are a Clinical Data Extractor. Summarize the patient metadata and scan parameters briefly."
    human_prompt = f"Metadata: {json.dumps(state['metadata'])}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return {"clinical_context": response.content}

def radiologist_node(state: CouncilState):
    """Analyzes 2D classification and localization."""
    system_prompt = "You are an expert Neuro-Radiologist. Analyze the 2D MRI outputs (DenseNet classification and Attention U-Net localization). Provide a brief summary of the pathological findings and potential diagnosis based solely on these numbers."
    human_prompt = f"DenseNet: {json.dumps(state['densenet_outputs'])}\nAttention: {json.dumps(state['attention_outputs'])}\nContext: {state['clinical_context']}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return {"radiologist_report": response.content}

def neurologist_node(state: CouncilState):
    """Analyzes 3D volumetric structure."""
    system_prompt = "You are an expert Neurologist. Analyze the 3D volumetric MRI output (Swin-UNETR). Focus on the structural impact, volume of the mass, and potential neurological implications."
    human_prompt = f"Swin-UNETR: {json.dumps(state['swin_unetr_outputs'])}\nContext: {state['clinical_context']}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return {"neurologist_report": response.content}

def cmo_node(state: CouncilState):
    """Synthesizes reports and acts as Chief Medical Officer."""
    system_prompt = "You are the Chief Medical Officer of a Tumor Board. Review the Radiologist and Neurologist reports. Synthesize them into a single, cohesive final diagnostic report. Resolve any minor discrepancies. Output the final report as a structured summary."
    human_prompt = f"Radiologist Report: {state['radiologist_report']}\n\nNeurologist Report: {state['neurologist_report']}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    return {"final_diagnosis": response.content}

# Build the LangGraph workflow
workflow = StateGraph(CouncilState)

workflow.add_node("Clinical_Extractor", clinical_extractor_node)
workflow.add_node("Radiologist", radiologist_node)
workflow.add_node("Neurologist", neurologist_node)
workflow.add_node("CMO", cmo_node)

# Define routing
workflow.set_entry_point("Clinical_Extractor")
workflow.add_edge("Clinical_Extractor", "Radiologist")
workflow.add_edge("Clinical_Extractor", "Neurologist")
workflow.add_edge("Radiologist", "CMO")
workflow.add_edge("Neurologist", "CMO")
workflow.add_edge("CMO", END)

# Compile graph
app = workflow.compile()

def run_council_debate(metadata, densenet, attention, swin) -> str:
    initial_state = CouncilState(
        metadata=metadata,
        densenet_outputs=densenet,
        attention_outputs=attention,
        swin_unetr_outputs=swin,
        clinical_context="",
        radiologist_report="",
        neurologist_report="",
        discrepancy_found=False,
        final_diagnosis=""
    )
    
    final_state = app.invoke(initial_state)
    return final_state["final_diagnosis"]
