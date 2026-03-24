import io
import logging
import os
from typing import List, Optional
from urllib.parse import urlparse

import azure.functions as func
import requests
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobClient, BlobServiceClient
from openai import OpenAI
from pypdf import PdfWriter, PdfReader

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MAX_CHARS = 12_000

app = func.FunctionApp()

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Trigger
# ──────────────────────────────────────────────
@app.event_grid_trigger(arg_name="event")
def blob_summarizer(event: func.EventGridEvent) -> None:
    """
    Fires whenever a blob is created in the configured storage container.
    Flow: Blob upload → Document Intelligence → Azure OpenAI summarise → Salesforce PATCH
    """
    record_id = None
    try:
        # ── 1. Extract blob URL from Event Grid payload ──────────────────
        # event.get_json() returns the inner `data` payload directly (not the full envelope)
        event_data: dict = event.get_json()
        blob_url: str = event_data["url"]
        logger.info("Processing blob: %s", blob_url)

        # ── 2. Read blob metadata (contains Salesforce recordId) ─────────
        conn_str = os.environ["BLOB_STORAGE_CONNECTION"]
        # Parse container & blob name from URL
        parsed = urlparse(blob_url)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        container_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""
        blob_client = BlobClient.from_connection_string(conn_str, container_name=container_name, blob_name=blob_name)
        props = blob_client.get_blob_properties()
        metadata: dict = props.metadata or {}

        record_id = metadata.get("id") or metadata.get("recordid")
        if not record_id:
            logger.warning("No recordId in blob metadata — skipping.")
            return

        # ── 3. Download blob bytes and send directly to Document Intelligence ──
        # (avoids the 4 MB URL-source size limit)
        logger.info("Downloading blob bytes for Document Intelligence")
        blob_data = blob_client.download_blob().readall()
        logger.info("Downloaded %d bytes", len(blob_data))

        # ── 4. Extract text via Document Intelligence ────────────────────
        doc_client = DocumentIntelligenceClient(
            endpoint=os.environ["DOC_INTEL_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["DOC_INTEL_KEY"]),
        )
        poller = doc_client.begin_analyze_document(
            "prebuilt-read",
            body=io.BytesIO(blob_data),
        )
        result = poller.result()

        full_text: str = result.content or ""
        page_count: int = len(result.pages)
        logger.info("Extracted %d chars across %d pages", len(full_text), page_count)

        if not full_text.strip():
            logger.warning("Document Intelligence returned empty content — skipping.")
            return

        # ── 4. Summarise with Llama on Azure AI Foundry ─────────────────
        ai_client = OpenAI(
            api_key=os.environ["AZURE_OPENAI_KEY"],
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

        final_summary = _summarise_document(ai_client, deployment, full_text)
        logger.info("Summary generated (%d chars)", len(final_summary))
        logger.info("=== SUMMARY ===\n%s\n===============", final_summary)

        # ── 5. Push result back to Salesforce (Apex REST) ────────────────
        _call_salesforce(record_id, final_summary, page_count, blob_url)
        logger.info("Salesforce updated for record %s", record_id)

    except Exception as exc:
        logger.exception("Unhandled error processing blob: %s", exc)
        _report_failure_to_salesforce(record_id, exc)
        raise  # re-raise so Azure marks the invocation as failed


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _split_text(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """Break *text* into fixed-size chunks."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _summarise(client: OpenAI, deployment: str, text: str) -> str:
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a precise medical summarizer."},
            {"role": "user", "content": f"Summarize this document clearly:\n{text}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def _summarise_document(client: OpenAI, deployment: str, full_text: str) -> str:
    """Map-reduce summarisation for arbitrarily large documents."""
    if len(full_text) <= MAX_CHARS:
        return _summarise(client, deployment, full_text)

    chunks = _split_text(full_text)
    logger.info("Document split into %d chunks", len(chunks))

    partial_summaries = [_summarise(client, deployment, chunk) for chunk in chunks]
    combined = "\n".join(partial_summaries)
    return _summarise(client, deployment, combined)


def _get_salesforce_token() -> str:
    """
    Exchange the stored refresh token for a fresh Salesforce access token.
    Uses the OAuth2 refresh-token grant with the Connected App credentials.

    Required env vars:
        SF_AUTH_URL       – e.g. https://login.salesforce.com  (or a My Domain URL)
        SF_CONSUMER_KEY   – Connected App client_id
        SF_CONSUMER_SECRET– Connected App client_secret
        SF_REFRESH_TOKEN  – Long-lived refresh token
    """
    auth_url = os.environ["SF_AUTH_URL"].rstrip("/") + "/services/oauth2/token"
    resp = requests.post(
        auth_url,
        data={
            "grant_type": "refresh_token",
            "client_id": os.environ["SF_CONSUMER_KEY"],
            "client_secret": os.environ["SF_CONSUMER_SECRET"],
            "refresh_token": os.environ["SF_REFRESH_TOKEN"],
        },
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    logger.info("Salesforce access token refreshed successfully")
    return token


def _call_salesforce(record_id: str, summary: str, page_count: int, doc_url: str) -> None:
    """
    POST to the authenticated Salesforce Apex REST endpoint.
    Endpoint: POST {SALESFORCE_BASE_URL}/services/apexrest/UpdateDocumentSummary/
    Payload:  { "fileName": "...", "summary": "...", "noOfPages": <int> }

    fileName is the full blob path within the container, e.g.:
        00001029_500f6000009QXdJAAW/merged/March.pdf
    """
    base_url = os.environ["SALESFORCE_BASE_URL"].rstrip("/")
    url = f"{base_url}/services/apexrest/UpdateDocumentSummary/"

    # Extract the blob path after the container name from the URL.
    # URL format: https://<account>.blob.core.windows.net/<container>/<blob-path>
    parsed = urlparse(doc_url)
    # path = /<container>/<blob-path>  →  strip leading slash, split off container
    path_parts = parsed.path.lstrip("/").split("/", 1)
    file_name = path_parts[1] if len(path_parts) > 1 else parsed.path.lstrip("/")

    access_token = _get_salesforce_token()

    payload = {
        "fileName": file_name,
        "summary": summary,
        "noOfPages": page_count,
    }
    logger.info("Calling Salesforce: %s | file=%s pages=%d", url, file_name, page_count)
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    logger.info("Salesforce response: %s", response.text[:200])


def _report_failure_to_salesforce(record_id: Optional[str], exc: Exception) -> None:
    """Best-effort failure notification; silently swallows its own errors."""
    if not record_id:
        return
    try:
        access_token = _get_salesforce_token()
        base_url = os.environ["SALESFORCE_BASE_URL"]
        url = f"{base_url}/services/data/v59.0/sobjects/Document__c/{record_id}"
        requests.patch(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"Status__c": "Failed", "Error__c": str(exc)},
            timeout=30,
        )
    except Exception as inner:
        logger.error("Could not report failure to Salesforce: %s", inner)


# ──────────────────────────────────────────────
# PDF Merge Function
# ──────────────────────────────────────────────
@app.route(route="merge-pdfs", methods=["POST"])
def merge_pdfs(req: func.HttpRequest) -> func.HttpResponse:
    """
    Merges two PDF blobs into one and uploads the result back to Azure Blob Storage.

    Expected JSON body:
    {
        "blob_url_1": "https://redfernstech.blob.core.windows.net/cases/path/to/file1.pdf",
        "blob_url_2": "https://redfernstech.blob.core.windows.net/cases/path/to/file2.pdf",
        "output_name": "merged_output.pdf"   // optional
    }

    Returns JSON:
    {
        "merged_url": "https://...",
        "pages": <total page count>
    }
    """
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON body", status_code=400)

    blob_url_1 = body.get("blob_url_1")
    blob_url_2 = body.get("blob_url_2")
    if not blob_url_1 or not blob_url_2:
        return func.HttpResponse(
            "Both 'blob_url_1' and 'blob_url_2' are required.", status_code=400
        )

    conn_str = os.environ["BLOB_STORAGE_CONNECTION"]

    def _download(blob_url: str) -> bytes:
        from urllib.parse import unquote
        parsed = urlparse(blob_url)
        parts = parsed.path.lstrip("/").split("/", 1)
        container = parts[0]
        blob_name = unquote(parts[1]) if len(parts) > 1 else ""
        client = BlobClient.from_connection_string(conn_str, container, blob_name)
        return client.download_blob().readall()

    logger.info("Downloading blob 1: %s", blob_url_1)
    data1 = _download(blob_url_1)
    logger.info("Downloading blob 2: %s", blob_url_2)
    data2 = _download(blob_url_2)

    # ── Merge with pypdf ──────────────────────────────────────────────────
    writer = PdfWriter()
    for data in (data1, data2):
        reader = PdfReader(io.BytesIO(data))
        for page in reader.pages:
            writer.add_page(page)

    merged_buffer = io.BytesIO()
    writer.write(merged_buffer)
    merged_buffer.seek(0)
    total_pages = len(writer.pages)
    logger.info("Merged PDF has %d pages", total_pages)

    # ── Upload merged PDF into the same virtual directory as blob 1 ──────
    # Structure: {container}/{case_folder}/{subfolder}/{file}
    # Merged goes to: {container}/{case_folder}/merged/{output_name}
    parsed1 = urlparse(blob_url_1)
    parts1 = parsed1.path.lstrip("/").split("/", 1)
    container_name = parts1[0]
    blob1_path = parts1[1] if len(parts1) > 1 else ""
    # Extract the top-level virtual directory (case folder) — first path segment
    case_folder = blob1_path.split("/")[0] if "/" in blob1_path else ""
    original_name = blob1_path.split("/")[-1] or "file1.pdf"
    output_filename = body.get("output_name") or f"merged_{original_name}"
    # Place under {case_folder}/merged/
    output_blob_path = f"{case_folder}/merged/{output_filename}" if case_folder else f"merged/{output_filename}"

    svc = BlobServiceClient.from_connection_string(conn_str)
    upload_client = svc.get_blob_client(container=container_name, blob=output_blob_path)
    upload_client.upload_blob(merged_buffer, overwrite=True, content_settings=None)

    account_name = upload_client.account_name
    merged_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{output_blob_path}"
    logger.info("Merged PDF uploaded: %s", merged_url)

    import json
    return func.HttpResponse(
        json.dumps({"merged_url": merged_url, "pages": total_pages}),
        mimetype="application/json",
        status_code=200,
    )
