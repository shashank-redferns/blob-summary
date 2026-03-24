import io
import logging
import os
from typing import List, Optional
from urllib.parse import urlparse

import azure.functions as func
import requests
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobClient
from openai import OpenAI

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
        conn_str = os.environ["AzureWebJobsStorage"]
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


def _call_salesforce(record_id: str, summary: str, page_count: int, doc_url: str) -> None:
    """
    POST to a custom Salesforce Apex REST service.
    The Apex class should be exposed at:
        @RestResource(urlMapping='/BlobSummary/*')
    and accept: recordId, summary, pageCount, docUrl, status
    """
    base_url = os.environ["SALESFORCE_BASE_URL"].rstrip("/")
    apex_class = os.environ.get("SALESFORCE_APEX_CLASS", "BlobSummary")
    url = f"{base_url}/services/apexrest/{apex_class}/"
    payload = {
        "recordId": record_id,
        "summary": summary,
        "pageCount": page_count,
        "docUrl": doc_url,
        "status": "Completed",
    }
    logger.info("Calling Salesforce Apex REST: %s | payload keys: %s", url, list(payload.keys()))
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {os.environ['SALESFORCE_TOKEN']}",
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
        base_url = os.environ["SALESFORCE_BASE_URL"]
        url = f"{base_url}/services/data/v59.0/sobjects/Document__c/{record_id}"
        requests.patch(
            url,
            headers={
                "Authorization": f"Bearer {os.environ['SALESFORCE_TOKEN']}",
                "Content-Type": "application/json",
            },
            json={"Status__c": "Failed", "Error__c": str(exc)},
            timeout=30,
        )
    except Exception as inner:
        logger.error("Could not report failure to Salesforce: %s", inner)
