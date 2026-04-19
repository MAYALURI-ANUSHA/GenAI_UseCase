#!/usr/bin/env python3
"""
Test script to find working Groq models.

Run this to discover which models are currently available:
    python test_groq_models.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Error: GROQ_API_KEY not found in .env file")
    sys.exit(1)

print("\n" + "="*60)
print("GROQ MODEL AVAILABILITY TEST")
print("="*60 + "\n")

# List of models to test
models_to_test = [
    # Latest models (most likely to work)
    "llama-3.2-90b-vision-preview",
    "llama-3.1-405b-reasoning",
    
    # Common models
    "mixtral-8x7b-32768",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    
    # Older models (likely deprecated)
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]

# Remove duplicates while preserving order
models_to_test = list(dict.fromkeys(models_to_test))

working_models = []
deprecated_models = []
unknown_models = []

try:
    from langchain_groq import ChatGroq
except ImportError:
    print("❌ Error: langchain_groq not installed")
    print("Run: pip install langchain-groq")
    sys.exit(1)

print(f"Testing {len(models_to_test)} models...\n")

for i, model in enumerate(models_to_test, 1):
    print(f"[{i}/{len(models_to_test)}] Testing: {model}...", end=" ", flush=True)
    
    try:
        llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0.7)
        # Try a simple invocation
        response = llm.invoke("Say 'hello' in one word")
        print("✅ WORKS")
        working_models.append(model)
    except Exception as e:
        error_msg = str(e).lower()
        if "decommissioned" in error_msg or "deprecated" in error_msg or "no longer" in error_msg:
            print("⚠️ DEPRECATED")
            deprecated_models.append(model)
        else:
            print(f"❌ ERROR")
            unknown_models.append((model, str(e)[:80]))

print("\n" + "="*60)
print("RESULTS")
print("="*60 + "\n")

if working_models:
    print("✅ WORKING MODELS (can use):")
    for model in working_models:
        marker = "⭐ RECOMMENDED" if model == working_models[0] else ""
        print(f"   - {model} {marker}")

if deprecated_models:
    print("\n⚠️ DEPRECATED MODELS (don't use):")
    for model in deprecated_models:
        print(f"   - {model}")

if unknown_models:
    print("\n❌ ERROR MODELS:")
    for model, error in unknown_models:
        print(f"   - {model}")
        print(f"     Error: {error[:60]}...")

print("\n" + "="*60)

if working_models:
    print(f"\n✅ USE THIS MODEL: {working_models[0]}")
    print(f"\nUpdate src/rag_chain.py:")
    print(f'    llm_model: str = "{working_models[0]}"')
    print(f"\nThen restart: streamlit run app.py")
else:
    print("\n❌ No working models found!")
    print("Check your Groq API key and internet connection.")
    print("Visit: https://console.groq.com/docs/models")

print("\n" + "="*60 + "\n")
