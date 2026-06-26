import os
import json
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# SCHEMA DEFINITION
# ==============================================================================
class HealthAdvisory(BaseModel):
    risk_level: str = Field(
        description="The calculated health risk level: 'Low', 'Moderate', 'High', 'Severe', or 'Hazardous'."
    )
    primary_advisory: str = Field(
        description="The main personalized health advisory message translated into the requested language."
    )
    precautions: List[str] = Field(
        description="A list of 3-5 specific actionable precautions translated into the requested language."
    )
    mask_recommended: bool = Field(
        description="Boolean indicating if wearing a mask (like N95) is recommended for this specific user."
    )

# ==============================================================================
# ADVISORY ENGINE
# ==============================================================================
def generate_advisory(aqi: int, age: int, condition: str, language: str = "English") -> str:
    """
    Generates a personalized, localized health advisory using Gemini and LangChain RAG principles.
    Returns a JSON string matching the HealthAdvisory schema.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Fallback mechanism for local testing without an API Key
    if not api_key:
        logger.warning("GOOGLE_API_KEY not found. Returning a deterministic fallback JSON for pipeline continuity.")
        return _generate_fallback_advisory(aqi, condition, language)

    try:
        # Initialize the Gemini model with structured output
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.2, 
            max_retries=3
        )
        structured_llm = llm.with_structured_output(HealthAdvisory)

        # Define the system prompt instruction set
        template = """
        You are an AI Health Advisor for an Urban Air Quality Intelligence Platform.
        Generate a personalized health advisory based strictly on the following citizen data:
        
        - Current AQI: {aqi}
        - Citizen Age: {age}
        - Pre-existing Medical Conditions: {condition}
        - Target Language: {language}

        Clinical Guidelines to Follow:
        1. If AQI > 300: Classify as 'Hazardous'. Mandate N95 masks and prohibit outdoor activities for everyone.
        2. If AQI > 200 and condition includes respiratory/cardiovascular issues (e.g., Asthma, COPD): Classify as 'Severe'. Mandate masks and strict indoor confinement.
        3. If AQI < 100: Classify as 'Low' or 'Moderate'. Safe for normal activities, but advise minor caution for sensitive groups if AQI is between 51-100.
        4. Tone must be empathetic, medically sound, and highly actionable.
        
        CRITICAL: The 'primary_advisory' and 'precautions' MUST be written entirely in {language}.
        """

        prompt = PromptTemplate(
            input_variables=["aqi", "age", "condition", "language"],
            template=template
        )

        # Chain execution
        logger.info(f"Generating {language} advisory for Age: {age}, Condition: {condition}, AQI: {aqi}")
        chain = prompt | structured_llm
        result = chain.invoke({
            "aqi": aqi,
            "age": age,
            "condition": condition,
            "language": language
        })

        # Return structured JSON
        return result.model_dump_json(indent=4)

    except Exception as e:
        logger.error(f"Failed to generate advisory via Gemini API: {e}")
        return _generate_fallback_advisory(aqi, condition, language)

def _generate_fallback_advisory(aqi: int, condition: str, language: str) -> str:
    """Deterministic fallback logic when API is unavailable or rate-limited."""
    is_hindi = language.lower() == "hindi"
    is_vulnerable = "asthma" in condition.lower() or "heart" in condition.lower()

    if aqi > 300 or (aqi > 200 and is_vulnerable):
        advisory = {
            "risk_level": "Severe",
            "primary_advisory": "हवा की गुणवत्ता बहुत खराब है। कृपया घर के अंदर रहें।" if is_hindi else "Air quality is severely degraded. Please remain indoors.",
            "precautions": [
                "बाहर जाते समय N95 मास्क पहनें।" if is_hindi else "Wear an N95 mask if you must go outside.",
                "सभी बाहरी व्यायाम बंद करें।" if is_hindi else "Stop all outdoor physical exertion.",
                "अपने इनहेलर को पास रखें।" if is_hindi else "Keep your prescribed inhaler nearby."
            ],
            "mask_recommended": True
        }
    elif aqi <= 100:
        advisory = {
            "risk_level": "Low",
            "primary_advisory": "हवा की गुणवत्ता अच्छी है। बाहरी गतिविधियों के लिए सुरक्षित है।" if is_hindi else "Air quality is good. It is safe for normal outdoor activities.",
            "precautions": [
                "आप सुरक्षित रूप से बाहर व्यायाम कर सकते हैं।" if is_hindi else "You may safely exercise outdoors.",
                "खिड़कियां खोलकर ताजी हवा आने दें।" if is_hindi else "Open windows to allow fresh air circulation."
            ],
            "mask_recommended": False
        }
    else:
        advisory = {
            "risk_level": "Moderate",
            "primary_advisory": "हवा की गुणवत्ता मध्यम है। संवेदनशील लोगों को सावधानी बरतनी चाहिए।" if is_hindi else "Air quality is moderate. Sensitive individuals should exercise caution.",
            "precautions": [
                "भारी बाहरी काम कम करें।" if is_hindi else "Reduce heavy outdoor exertion.",
                "यदि आपको सांस लेने में तकलीफ हो तो घर के अंदर चले जाएं।" if is_hindi else "Move indoors if you experience breathing difficulty."
            ],
            "mask_recommended": is_vulnerable
        }
        
    return json.dumps(advisory, indent=4, ensure_ascii=False)

# ==============================================================================
# PIPELINE EXECUTION & TESTING
# ==============================================================================
def main():
    print("="*60)
    print("CITIZEN HEALTH ADVISORY AGENT - TEST EXECUTION")
    print("="*60)

    # Test Case 1: High Risk, Vulnerable Citizen (English)
    print("\n[TEST 1] AQI: 350 | Age: 65 | Condition: Asthma | Language: English")
    response_1 = generate_advisory(aqi=350, age=65, condition="Asthma, Hypertension", language="English")
    print(response_1)

    # Test Case 2: Low Risk, Healthy Citizen (Hindi)
    print("\n[TEST 2] AQI: 80 | Age: 25 | Condition: None | Language: Hindi")
    response_2 = generate_advisory(aqi=80, age=25, condition="None", language="Hindi")
    print(response_2)
    
    # Test Case 3: Moderate Risk, Vulnerable Citizen (Hindi)
    print("\n[TEST 3] AQI: 180 | Age: 12 | Condition: Bronchitis | Language: Hindi")
    response_3 = generate_advisory(aqi=180, age=12, condition="Bronchitis", language="Hindi")
    print(response_3)

if __name__ == "__main__":
    main()