"""Groq model management utility for tracking available models and handling deprecations."""

# Updated: Latest Test Results
# Status as of latest test run:
#   ✅ Working: llama-3.1-8b-instant
#   ⚠️ Deprecated: llama-3.2-90b-vision-preview, mixtral-8x7b-32768, llama-3.1-70b-versatile, gemma-7b-it
#   ❌ Not Available: llama-3.1-405b-reasoning

# List of known Groq models
GROQ_MODELS = {
    # RECOMMENDED MODELS (Currently Working)
    "llama-3.1-8b-instant": {
        "name": "Llama 3.1 8B Instant",
        "status": "active",
        "size": "8B",
        "speed": "very_fast",
        "quality": "good",
        "context": "8000 tokens",
        "description": "Fastest and most reliable model. Perfect for RAG applications.",
        "recommended": True
    },
    
    # DEPRECATED MODELS (Do Not Use)
    "llama-3.2-90b-vision-preview": {
        "name": "Llama 3.2 90B Vision Preview",
        "status": "deprecated",
        "reason": "Model decommissioned by Groq",
        "description": "DEPRECATED - Use llama-3.1-8b-instant instead"
    },
    "llama-3.1-70b-versatile": {
        "name": "Llama 3.1 70B Versatile",
        "status": "deprecated",
        "reason": "Model decommissioned by Groq",
        "description": "DEPRECATED - Use llama-3.1-8b-instant instead"
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral 8x7B",
        "status": "deprecated",
        "reason": "Model decommissioned by Groq",
        "description": "DEPRECATED - Use llama-3.1-8b-instant instead"
    },
    "gemma-7b-it": {
        "name": "Gemma 7B IT",
        "status": "deprecated",
        "reason": "Model decommissioned by Groq",
        "description": "DEPRECATED - Use llama-3.1-8b-instant instead"
    },
    
    # UNAVAILABLE MODELS
    "llama-3.1-405b-reasoning": {
        "name": "Llama 3.1 405B Reasoning",
        "status": "unavailable",
        "reason": "404 - Model not found or not accessible",
        "description": "May be available on different Groq pricing tier"
    }
}

def get_available_models():
    """Return list of currently available (active) models"""
    return [m for m, d in GROQ_MODELS.items() if d.get("status") == "active"]

def get_recommended_model():
    """Return the recommended model for RAG applications"""
    for model_id, model_info in GROQ_MODELS.items():
        if model_info.get("recommended"):
            return model_id
    return "llama-3.1-8b-instant"

def print_available_models():
    """Print formatted list of all models with status"""
    print("\n" + "="*60)
    print("GROQ MODELS STATUS")
    print("="*60 + "\n")
    
    # Active models
    active = {m: d for m, d in GROQ_MODELS.items() if d.get("status") == "active"}
    if active:
        print("✅ ACTIVE MODELS:")
        for model_id, model_info in active.items():
            print(f"   {model_id}")
            print(f"      Speed: {model_info.get('speed', 'N/A')} | Quality: {model_info.get('quality', 'N/A')}")
            print(f"      {model_info.get('description', '')}")
            if model_info.get('recommended'):
                print(f"      ⭐ RECOMMENDED FOR RAG")
            print()
    
    # Deprecated models
    deprecated = {m: d for m, d in GROQ_MODELS.items() if d.get("status") == "deprecated"}
    if deprecated:
        print("⚠️ DEPRECATED MODELS (Don't Use):")
        for model_id in deprecated:
            print(f"   - {model_id}")
        print()
    
    # Unavailable models
    unavailable = {m: d for m, d in GROQ_MODELS.items() if d.get("status") == "unavailable"}
    if unavailable:
        print("❌ UNAVAILABLE MODELS:")
        for model_id in unavailable:
            print(f"   - {model_id}")
        print()
    
    print("="*60)

if __name__ == "__main__":
    print_available_models()
    recommended = get_recommended_model()
    print(f"\n🎯 Use this model in src/rag_chain.py:")
    print(f"   llm_model: str = \"{recommended}\"")
    print()

