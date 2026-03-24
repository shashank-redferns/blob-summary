# Blob Summarizer — Azure Function

A production-ready Azure Function that listens for blob upload events via Event Grid,
extracts text with Document Intelligence, summarises with Azure OpenAI, and patches the
result back to a Salesforce record.

## Project Structure

```
blob-summarizer/
├── function_app.py        ← Entry point (v2 programming model)
├── host.json              ← Runtime config (timeout, logging)
├── local.settings.json    ← Local env vars (never commit this)
├── requirements.txt       ← Pinned Python dependencies
├── .funcignore            ← Excludes files from deployment zip
└── .gitignore
```

## Local Development

```bash
# 1. Create and activate virtual env
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fill in local.settings.json with real values, then:
func start
```

## Deploy to Azure

```bash
# One-time: login + create resources
az login
az group create --name rg-blob-summarizer --location eastus
az storage account create --name <storageaccname> --resource-group rg-blob-summarizer --sku Standard_LRS
az functionapp create \
  --resource-group rg-blob-summarizer \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name blob-summarizer-fn \
  --storage-account <storageaccname>

# Set all env vars in Azure (one per line)
az functionapp config appsettings set \
  --name blob-summarizer-fn \
  --resource-group rg-blob-summarizer \
  --settings \
    DOC_INTEL_ENDPOINT="..." \
    DOC_INTEL_KEY="..." \
    AZURE_OPENAI_ENDPOINT="..." \
    AZURE_OPENAI_KEY="..." \
    AZURE_OPENAI_DEPLOYMENT="..." \
    SALESFORCE_BASE_URL="..." \
    SALESFORCE_TOKEN="..."

# Deploy the function code
func azure functionapp publish blob-summarizer-fn
```

## Event Grid Subscription

After deploying, create an Event Grid subscription on your storage account so blob creation events fire the function:

```bash
az eventgrid event-subscription create \
  --name blob-created-sub \
  --source-resource-id /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.Storage/storageAccounts/<storage-account> \
  --endpoint-type azurefunction \
  --endpoint /subscriptions/<sub-id>/resourceGroups/rg-blob-summarizer/providers/Microsoft.Web/sites/blob-summarizer-fn/functions/blob_summarizer \
  --included-event-types Microsoft.Storage.BlobCreated
```

## Environment Variables Required

| Variable | Description |
|---|---|
| `DOC_INTEL_ENDPOINT` | Azure Document Intelligence endpoint URL |
| `DOC_INTEL_KEY` | Document Intelligence API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name of your GPT model |
| `SALESFORCE_BASE_URL` | Salesforce instance URL (e.g. `https://xxx.my.salesforce.com`) |
| `SALESFORCE_TOKEN` | Salesforce OAuth bearer token |
