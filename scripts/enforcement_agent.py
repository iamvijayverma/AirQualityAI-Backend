import os
import json
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# SCHEMA DEFINITION
# ==============================================================================
class EnforcementOutput(BaseModel):
    risk_level: str = Field(
        description="The calculated regulatory risk tier: 'Low', 'Moderate', 'High', or 'Severe'."
    )
    priority_actions: List[str] = Field(
        description="Ranked structural intervention recommendations based on sectoral contributions."
    )
    expected_aqi_reduction: str = Field(
        description="An estimate or descriptive projection of the mitigation impact (e.g., 'Expected drop of 40-60 AQI points')."
    )
    enforcement_score: int = Field(
        description="An engineered structural urgency metric graded from 0 to 100."
    )

# ==============================================================================
# ENFORCEMENT ENGINE
# ==============================================================================
def run_enforcement_agent(aqi: int, traffic_pct: float, construction_pct: float, industry_pct: float) -> str:
    """
    Evaluates environmental triggers, applies enforcement logic, and uses 
    Gemini/LangChain to output a prioritized, structured interventions JSON object.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Fallback to local rule engine if no API key is set
    if not api_key:
        logger.warning("GOOGLE_API_KEY absent. Executing deterministic fallback rules matrix.")
        return _run_fallback_enforcement(aqi, traffic_pct, construction_pct, industry_pct)

    try:
        # Initialize the LLM with strict JSON schema constraints
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.1, 
            max_retries=3
        )
        structured_llm = llm.with_structured_output(EnforcementOutput)

        # Prompt enforcing rule-boundaries and analytical processing
        template = """
        You are the Core Decision Engine for an Urban Air Quality Enforcement Intelligence Platform.
        Analyze the incoming multi-sector emission profile and generate a structured enforcement response.

        INPUT METRICS:
        - Ambient AQI: {aqi}
        - Traffic Sector Contribution: {traffic_pct}%
        - Construction Sector Contribution: {construction_pct}%
        - Industrial Sector Contribution: {industry_pct}%

        MANDATORY CORE RULES MATRIX:
        1. If AQI > 300, set risk_level to 'Severe'.
        2. If Traffic Contribution > 50%, add actions: 'Implement odd-even vehicle restrictions', 'Restrict heavy diesel truck entries during peak hours'.
        3. If Construction Contribution > 30%, add actions: 'Mandate site-wide continuous dust suppression sprays', 'Deploy immediate municipal site compliance inspections'.
        4. If Industry Contribution > 40%, add actions: 'Initiate mandatory third-party emission audits', 'Enforce temporary 30% operational capacity rollbacks'.

        ANALYTICAL METRIC CALCULATIONS:
        - Calculate enforcement_score (0-100) using a blended index: (AQI/500)*50 + (Max Sector Contribution)*0.5. Cap at 100.
        - Calculate expected_aqi_reduction by estimating cumulative impact of selected actions (e.g., traffic cuts can drop AQI by 15-20%).
        
        Ensure all mandatory rules are fully reflected in the final output arrays.
        """

        prompt = PromptTemplate(
            input_variables=["aqi", "traffic_pct", "construction_pct", "industry_pct"],
            template=template
        )

        logger.info(f"Invoking enforcement agent for AQI: {aqi}")
        chain = prompt | structured_llm
        result = chain.invoke({
            "aqi": aqi,
            "traffic_pct": traffic_pct,
            "construction_pct": construction_pct,
            "industry_pct": industry_pct
        })

        return result.model_dump_json(indent=4)

    except Exception as e:
        logger.error(f"Failed to execute LLM enforcement chain: {e}")
        return _run_fallback_enforcement(aqi, traffic_pct, construction_pct, industry_pct)

def _run_fallback_enforcement(aqi: int, traffic_pct: float, construction_pct: float, industry_pct: float) -> str:
    """Fallback rule engine matching requirements perfectly via procedural code."""
    # Rule 1: Risk Level
    risk = "Severe" if aqi > 300 else ("High" if aqi > 200 else "Moderate")
    
    actions = []
    reduction_estimate = 0
    
    # Rule 2: Traffic
    if traffic_pct > 50:
        actions.append("Implement odd-even vehicle restrictions.")
        actions.append("Restrict heavy commercial diesel trucks from entering urban zones.")
        reduction_estimate += 35
        
    # Rule 3: Construction
    if construction_pct > 30:
        actions.append("Mandate continuous dust suppression misting across all active zones.")
        actions.append("Deploy immediate task-force site compliance inspections.")
        reduction_estimate += 20
        
    # Rule 4: Industry
    if industry_pct > 40:
        actions.append("Initiate target-driven, unannounced industrial emission audits.")
        actions.append("Enforce a temporary rolling reduction in high-emission operations.")
        reduction_estimate += 45

    if not actions:
        actions.append("Maintain routine patrol sweeps and continue background baseline monitoring.")
        reduction_estimate = 5

    # Enforcement Score engineering equation (0-100)
    base_aqi_factor = min((aqi / 500) * 50, 50)
    max_sector_factor = min(max(traffic_pct, construction_pct, industry_pct) * 0.5, 50)
    score = int(base_aqi_factor + max_sector_factor)

    output = {
        "risk_level": risk,
        "priority_actions": actions,
        "expected_aqi_reduction": f"Estimated reduction of {reduction_estimate - 5}-{reduction_estimate + 10} AQI points upon deployment.",
        "enforcement_score": min(score, 100)
    }
    
    return json.dumps(output, indent=4)

# ==============================================================================
# PIPELINE MONITORING & VERIFICATION
# ==============================================================================
def main():
    print("="*70)
    print("ENFORCEMENT INTELLIGENCE AGENT - MODEL TEST RUN")
    print("="*70)

    # Test Case 1: Critical AQI, High Traffic
    print("\n[SCENARIO 1] Severe AQI with High Traffic Density Focus:")
    res_1 = run_enforcement_agent(aqi=340, traffic_pct=55.0, construction_pct=15.0, industry_pct=30.0)
    print(res_1)

    # Test Case 2: Moderate AQI, High Construction Activity
    print("\n[SCENARIO 2] Moderate AQI with Localized Construction Spikes:")
    res_2 = run_enforcement_agent(aqi=180, traffic_pct=25.0, construction_pct=42.0, industry_pct=33.0)
    print(res_2)

if __name__ == "__main__":
    main()