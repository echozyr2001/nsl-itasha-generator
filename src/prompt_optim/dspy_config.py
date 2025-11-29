"""DSPy configuration for prompt optimization."""
from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def configure_dspy():
    """
    Configure DSPy with appropriate language model.
    Supports Vertex AI via service account file (account.json) or API key.
    This should be called before running GEPA optimization.
    
    Returns:
        dspy.LM instance if successful, None otherwise
    """
    import dspy
    
    # Priority 1: Try Vertex AI with service account file (account.json)
    service_account_file = Path("account.json")
    if service_account_file.exists():
        try:
            # Read project_id and location from account.json
            with open(service_account_file, 'r') as f:
                account_info = json.load(f)
                project_id = account_info.get("project_id")
            
            # Set environment variables for Vertex AI authentication
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(service_account_file.resolve())
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            # Use "global" location as in other services
            location = "global"
            
            # Configure DSPy with Vertex AI
            # For Vertex AI, we need to use the vertex model name format
            # DSPy's LM class should automatically detect Vertex AI when GOOGLE_APPLICATION_CREDENTIALS is set
            try:
                # Try using Vertex AI model format
                # Note: Vertex AI model names may differ from standard Gemini API
                lm = dspy.LM(model="gemini/gemini-2.0-flash-exp")
                dspy.configure(lm=lm)
                print(f"DSPy configured with Vertex AI (project: {project_id}, location: {location})")
                return lm
            except Exception as e:
                print(f"Failed to configure Vertex AI model directly: {e}")
                # Try alternative approach with explicit Vertex AI configuration
                # Some DSPy versions may require different configuration
                print(f"   Using service account: {service_account_file}")
                print(f"   Project ID: {project_id}")
                print(f"   Location: {location}")
                # The environment variables should be enough for most cases
                # Try again with a simpler model name or default
                try:
                    lm = dspy.LM(model="gemini/gemini-2.0-flash-exp")
                    dspy.configure(lm=lm)
                    print("DSPy configured with Vertex AI (retry successful)")
                    return lm
                except:
                    print("Warning: Vertex AI configuration may need manual setup")
                    pass
        except Exception as e:
            print(f"Failed to configure Vertex AI: {e}")
            import traceback
            traceback.print_exc()
            pass
    
    # Priority 2: Try standard Gemini API with GOOGLE_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            # Configure with Gemini using DSPy's LM class
            os.environ["GOOGLE_API_KEY"] = api_key
            lm = dspy.LM(model="gemini/gemini-2.0-flash-exp")
            dspy.configure(lm=lm)
            print("DSPy configured with Gemini API (using GOOGLE_API_KEY)")
            return lm
        except Exception as e:
            print(f"Failed to configure Gemini API: {e}")
            pass
    
    # Priority 3: Fallback to OpenAI if available
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            lm = dspy.LM(model="gpt-4o-mini", api_key=openai_key)
            dspy.configure(lm=lm)
            print("DSPy configured with OpenAI")
            return lm
        except Exception as e:
            print(f"Failed to configure OpenAI: {e}")
            pass
    
    # If no LM configured, DSPy will use default settings
    print("Warning: No language model configured. GEPA optimization may not work properly.")
    print("Please ensure:")
    print("  1. account.json exists in project root, OR")
    print("  2. GOOGLE_API_KEY is set in .env file, OR")
    print("  3. OPENAI_API_KEY is set in .env file")
    return None

