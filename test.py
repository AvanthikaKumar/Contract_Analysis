import os
import requests
import json
import re

def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def check_env_var(name):
    value = os.getenv(name)
    if not value:
        return f"❌ Missing environment variable: {name}"
    return None

def diagnose_http_error(status, body):
    if status == 404:
        return "❌ 404 Not Found → Usually incorrect endpoint, deployment name, or API version."
    if status == 400:
        return "❌ 400 Bad Request → Malformed request, wrong API version, or invalid index/model."
    if status == 401:
        return "❌ 401 Unauthorized → Invalid API key."
    if status == 403:
        return "❌ 403 Forbidden → Key exists but does not have permissions."
    if status == 500:
        return "❌ 500 Internal Error → Service problem or region mismatch."
    return f"⚠ Unexpected status {status}: {body}"


# ==========================================================================
# Azure OpenAI Deep Diagnostic
# ==========================================================================
def diagnose_openai():
    header("Azure OpenAI Diagnostic")

    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT"
    ]

    for v in required_vars:
        missing = check_env_var(v)
        if missing:
            print(missing)
            return

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    version = os.getenv("AZURE_OPENAI_API_VERSION")
    deploy = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    # Check endpoint format
    if endpoint.endswith("/openai"):
        print("⚠ Endpoint ends with /openai — REMOVE it. Azure adds this internally.")
    
    if not re.match(r"https://.*\.openai\.azure\.com/?$", endpoint):
        print("❌ Endpoint format incorrect. Should look like:")
        print("   https://<resource>.openai.azure.com")
        return

    # Perform test request
    url = f"{endpoint}/openai/deployments/{deploy}/chat/completions?api-version={version}"
    headers = {"api-key": key, "Content-Type": "application/json"}
    data = {"messages": [{"role": "user", "content": "ping"}]}

    try:
        r = requests.post(url, headers=headers, json=data, timeout=10)
        if r.status_code == 200:
            print("✅ Azure OpenAI is working correctly.")
        else:
            print(diagnose_http_error(r.status_code, r.text))
            print("\n🔍 Raw Response:")
            print(r.text)
    except Exception as e:
        print(f"❌ Exception: {e}")


# ==========================================================================
# Azure Search Deep Diagnostic
# ==========================================================================
def diagnose_search():
    header("Azure AI Search Diagnostic")

    required_vars = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME"
    ]

    for v in required_vars:
        missing = check_env_var(v)
        if missing:
            print(missing)
            return

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_API_KEY")
    idx = os.getenv("AZURE_SEARCH_INDEX_NAME")

    if not endpoint.endswith(".search.windows.net"):
        print("❌ Search endpoint must be:")
        print("   https://<your-search-name>.search.windows.net")
        return

    url = f"{endpoint}/indexes/{idx}/docs?api-version=2024-07-01-Preview&$top=1"
    headers = {"api-key": key}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            print("✅ Azure AI Search is working correctly.")
        else:
            print(diagnose_http_error(r.status_code, r.text))
            print("\n🔍 Raw Response:")
            print(r.text)
    except Exception as e:
        print(f"❌ Exception: {e}")


# ==========================================================================
# Azure Document Intelligence Deep Diagnostic
# ==========================================================================
def diagnose_document_intelligence():
    header("Azure Document Intelligence Diagnostic")

    required_vars = [
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
    ]

    for v in required_vars:
        missing = check_env_var(v)
        if missing:
            print(missing)
            return

    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")

    if not endpoint.endswith(".cognitiveservices.azure.com"):
        print("❌ Wrong endpoint. It must be:")
        print("   https://<resource>.cognitiveservices.azure.com")
        return

    url = f"{endpoint}/formrecognizer/info?api-version=2024-11-30"
    headers = {"Ocp-Apim-Subscription-Key": key}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            print("✅ Document Intelligence is working correctly.")
        else:
            print(diagnose_http_error(r.status_code, r.text))
            print("\n🔍 Raw Response:")
            print(r.text)
    except Exception as e:
        print(f"❌ Exception: {e}")


# ==========================================================================
# RUN ALL
# ==========================================================================
if __name__ == "__main__":
    diagnose_openai()
    diagnose_search()
    diagnose_document_intelligence()