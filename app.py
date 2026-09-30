import os
import io
import json
import time
import pandas as pd
import streamlit as st

from google import genai
from google.genai import types
from PIL import Image
from supabase import create_client
from dotenv import load_dotenv


# APP NAME (change this in one place)

APP_NAME = "ManifestIQ"
APP_TAGLINE = "AI document scanner for supply chain & logistics"
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")


# STREAMLIT PAGE (must be the first Streamlit command)

st.set_page_config(
    page_title=f"{APP_NAME} | Logistics Document Scanner",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "📦",
    layout="wide",
)


# LOAD ENVIRONMENT VARIABLES

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
# Used automatically if the main model stays busy after several retries
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.1-flash-lite")
RETRY_DELAYS = [2, 5, 10]  # seconds to wait between retries


# CLIENTS

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    st.error("GEMINI_API_KEY is not found in your .env file")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    st.error("SUPABASE_URL or SUPABASE_KEY is missing from your .env file")


# AI PROMPT

AI_PROMPT = """
You are a document extraction assistant for supply chain and logistics operations.

Read the entire uploaded document and extract ALL information that can be reliably read.
Typical documents: bill of lading, air waybill, packing list, commercial invoice,
delivery note / proof of delivery, purchase order, goods received note (GRN),
customs declaration, freight invoice, inventory or stock sheet, receipt.

Do not guess or invent information. If information is missing or unreadable, use null.
Return ONLY valid JSON.

Use this structure:

{
  "document": {
    "document_type": null,
    "document_number": null,
    "document_date": null,
    "reference_numbers": null,
    "po_number": null,
    "invoice_number": null,
    "tracking_number": null,
    "container_number": null,
    "seal_number": null,
    "shipper_name": null,
    "shipper_address": null,
    "consignee_name": null,
    "consignee_address": null,
    "notify_party": null,
    "carrier_name": null,
    "vehicle_or_vessel": null,
    "driver_name": null,
    "origin": null,
    "destination": null,
    "ship_date": null,
    "delivery_date": null,
    "incoterms": null,
    "payment_terms": null,
    "total_packages": null,
    "total_gross_weight": null,
    "total_net_weight": null,
    "total_volume": null,
    "currency": null,
    "subtotal": null,
    "freight_charges": null,
    "tax": null,
    "total": null,
    "delivery_status": null,
    "signed_by": null,
    "additional_information": null
  },
  "items": [
    {
      "item_number": 1,
      "sku": null,
      "description": null,
      "hs_code": null,
      "batch_or_lot": null,
      "quantity": null,
      "unit_of_measure": null,
      "packages": null,
      "weight": null,
      "unit_price": null,
      "amount": null,
      "condition_or_remarks": null
    }
  ]
}

Rules:
- Extract every product, shipment line, pallet, carton or charge individually.
- Extract all rows from tables and analyze every page.
- Preserve names, dates, numbers, units, currencies and codes exactly as shown.
- Keep weight and volume units with the value (e.g. "1,250 KG").
- For numeric money fields (subtotal, freight_charges, tax, total, unit_price, amount) return plain numbers without currency symbols or commas.
- Do not guess missing information; use null.
- If there are no items, return an empty array.
- Put anything that does not fit a field in additional_information.
"""


# SETTINGS

FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png"}

DOCUMENT_FIELDS = [
    "document_type", "document_number", "document_date", "reference_numbers",
    "po_number", "invoice_number", "tracking_number", "container_number",
    "seal_number", "shipper_name", "shipper_address", "consignee_name",
    "consignee_address", "notify_party", "carrier_name", "vehicle_or_vessel",
    "driver_name", "origin", "destination", "ship_date", "delivery_date",
    "incoterms", "payment_terms", "total_packages", "total_gross_weight",
    "total_net_weight", "total_volume", "currency", "subtotal",
    "freight_charges", "tax", "total", "delivery_status", "signed_by",
    "additional_information",
]

ITEM_FIELDS = [
    "item_number", "sku", "description", "hs_code", "batch_or_lot", "quantity",
    "unit_of_measure", "packages", "weight", "unit_price", "amount",
    "condition_or_remarks",
]

NUMERIC_DOC_FIELDS = {"subtotal", "freight_charges", "tax", "total"}
NUMERIC_ITEM_FIELDS = {"item_number", "quantity", "unit_price", "amount"}

# Columns used by the search box on the Records page
SEARCH_COLUMNS = [
    "document_number", "po_number", "invoice_number", "tracking_number",
    "container_number", "shipper_name", "consignee_name", "carrier_name",
    "origin", "destination",
]


# HELPERS

def to_number(value):
    """Turn '1,200.50' or '$1200' into a float. Returns None if it can't."""
    if value is None or isinstance(value, (int, float)):
        return value
    cleaned = "".join(ch for ch in str(value) if ch.isdigit() or ch in ".-")
    try:
        return float(cleaned)
    except ValueError:
        return None


def as_text(value):
    """Lists/dicts from the AI become JSON text so they fit a text column."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def is_temporary_error(error):
    """True for 'try again later' errors: overloaded (503), rate limit (429), server errors."""
    code = getattr(error, "code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    text = str(error).upper()
    return "UNAVAILABLE" in text or "RESOURCE_EXHAUSTED" in text or "OVERLOADED" in text


def generate_with_retry(contents, notify=None):
    """Call Gemini. Retries busy/overloaded errors, then tries the fallback model."""
    models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models.append(GEMINI_FALLBACK_MODEL)

    last_error = None
    for model in models:
        for delay in [0] + RETRY_DELAYS:
            if delay:
                if notify:
                    notify(f"The AI model is busy. Retrying in {delay} seconds...")
                time.sleep(delay)
            try:
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            except Exception as error:
                if not is_temporary_error(error):
                    raise
                last_error = error
        if notify and model != models[-1]:
            notify(f"{model} is still busy. Switching to {models[-1]}...")
    raise last_error


def fetch_all(table_name, order_col):
    """Fetch every row of a table (Supabase returns 1000 rows per request)."""
    rows, start, step = [], 0, 1000
    while True:
        response = (
            supabase.table(table_name)
            .select("*")
            .order(order_col, desc=(order_col == "created_at"))
            .range(start, start + step - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < step:
            break
        start += step
    return rows


def to_csv_bytes(df):
    """CSV bytes that open correctly in Excel (UTF-8 with BOM)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def build_combined(docs_df, items_df):
    """One row per line item, with its document's details alongside."""
    if items_df.empty:
        return docs_df.copy()
    combined = docs_df.merge(
        items_df,
        left_on="id",
        right_on="document_id",
        how="left",
        suffixes=("", "_item"),
    )
    return combined.drop(columns=["id_item", "created_at_item"], errors="ignore")


# PAGE 1: SCAN DOCUMENT

def scan_page():
    st.title("Scan Document")
    st.write(
        "Upload a bill of lading, packing list, commercial invoice, delivery note, "
        "purchase order, waybill or goods received note, and let AI extract and organize it."
    )

    left_col, right_col = st.columns([1, 1])

    file_bytes = None
    mime_type = None

    with left_col:
        st.subheader("Upload Logistics Document")

        uploaded_file = st.file_uploader(
            "Drop your document here (image or PDF)",
            type=["jpg", "jpeg", "png", "pdf"],
        )

        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                mime_type = "application/pdf"
                file_bytes = uploaded_file.getvalue()
                st.info(f"PDF ready: {uploaded_file.name}")
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Your Uploaded Document", width=400)

                save_format = image.format if image.format in FORMAT_TO_MIME else "JPEG"
                mime_type = FORMAT_TO_MIME[save_format]

                if save_format == "JPEG" and image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")

                buffer = io.BytesIO()
                image.save(buffer, format=save_format)
                file_bytes = buffer.getvalue()

    with right_col:
        st.subheader("Organized Data")

        if uploaded_file is None:
            st.info("Waiting for you to upload a document on the left.")
            return

        if st.button(
            "Extract Shipment Data",
            type="primary",
            disabled=client is None or supabase is None,
        ):
            with st.spinner("Reading your document..."):
                try:
                    notice = st.empty()
                    response = generate_with_retry(
                        [
                            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                            AI_PROMPT,
                        ],
                        notify=lambda message: notice.info(message),
                    )
                    notice.empty()

                    ai_text = response.text.strip()
                    if ai_text.startswith("```json"):
                        ai_text = ai_text.replace("```json", "", 1).strip()
                    if ai_text.endswith("```"):
                        ai_text = ai_text[:-3].strip()

                    extracted = json.loads(ai_text)
                    document = extracted.get("document", {}) or {}
                    items = extracted.get("items", []) or []

                    # Save document
                    document_data = {}
                    for field in DOCUMENT_FIELDS:
                        value = document.get(field)
                        if field in NUMERIC_DOC_FIELDS:
                            value = to_number(value)
                        else:
                            value = as_text(value)
                        document_data[field] = value

                    document_response = (
                        supabase.table("documents").insert(document_data).execute()
                    )
                    document_id = document_response.data[0]["id"]

                    # Save items
                    if items:
                        item_records = []
                        for item in items:
                            record = {"document_id": document_id}
                            for field in ITEM_FIELDS:
                                value = item.get(field)
                                if field in NUMERIC_ITEM_FIELDS:
                                    value = to_number(value)
                                else:
                                    value = as_text(value)
                                record[field] = value
                            item_records.append(record)

                        supabase.table("document_items").insert(item_records).execute()

                    st.success(
                        "Document extracted and saved! "
                        "Open **Records & Export** in the sidebar to view or download it."
                    )

                    st.markdown("# Document Analysis")

                    st.markdown("## Shipment Details")
                    document_display = {
                        key.replace("_", " ").title(): value
                        for key, value in document.items()
                        if value is not None
                    }
                    st.table(document_display)

                    if items:
                        st.markdown("## Line Items / Cargo")
                        st.dataframe(items, width="stretch")

                    st.caption(f"Database Document ID: {document_id}")

                except json.JSONDecodeError:
                    st.error("AI returned an invalid JSON response. Please try again.")
                except Exception as e:
                    if is_temporary_error(e):
                        st.warning(
                            "The AI service is overloaded right now, even after several retries. "
                            "Nothing was saved. Please wait a minute and click the button again."
                        )
                    else:
                        st.error(f"Something went wrong: {e}")


# PAGE 2: RECORDS & EXPORT

def records_page():
    st.title("Records & Export")
    st.write("Browse everything ManifestIQ has extracted and download it as CSV.")

    if supabase is None:
        st.error("Supabase is not connected. Check your .env file.")
        return

    top_left, top_right = st.columns([5, 1])
    with top_right:
        if st.button("Refresh", width="stretch"):
            st.rerun()

    try:
        docs = fetch_all("documents", "created_at")
        items = fetch_all("document_items", "id")
    except Exception as e:
        st.error(f"Could not load records: {e}")
        return

    if not docs:
        st.info("No documents saved yet. Scan a document first, then come back here.")
        return

    docs_df = pd.DataFrame(docs)
    items_df = pd.DataFrame(items) if items else pd.DataFrame(columns=["id", "document_id"])

    # Put the most useful columns first
    lead = [c for c in ["created_at", "document_type", "document_number", "document_date"] if c in docs_df.columns]
    docs_df = docs_df[lead + [c for c in docs_df.columns if c not in lead]]

    # Filters
    f1, f2 = st.columns([2, 2])
    with f1:
        search = st.text_input(
            "Search",
            placeholder="Document no., PO, tracking, container, shipper, consignee, carrier...",
        )
    with f2:
        types_available = sorted(docs_df["document_type"].dropna().astype(str).unique()) if "document_type" in docs_df else []
        chosen_types = st.multiselect("Document type", types_available)

    filtered = docs_df.copy()
    if chosen_types:
        filtered = filtered[filtered["document_type"].astype(str).isin(chosen_types)]
    if search.strip():
        cols = [c for c in SEARCH_COLUMNS if c in filtered.columns]
        haystack = filtered[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        filtered = filtered[haystack.str.contains(search.strip().lower(), regex=False)]

    filtered_items = (
        items_df[items_df["document_id"].isin(filtered["id"])]
        if "document_id" in items_df.columns
        else items_df
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", len(filtered))
    m2.metric("Line items", len(filtered_items))
    m3.metric("Document types", filtered["document_type"].nunique() if "document_type" in filtered else 0)

    tab_docs, tab_items, tab_detail = st.tabs(["Documents", "Line Items", "Document Details"])

    # Documents
    with tab_docs:
        st.dataframe(filtered.drop(columns=["id"], errors="ignore"), width="stretch", hide_index=True)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download documents (CSV)",
                data=to_csv_bytes(filtered),
                file_name="manifestiq_documents.csv",
                mime="text/csv",
                width="stretch",
            )
        with d2:
            st.download_button(
                "Download everything combined (CSV)",
                data=to_csv_bytes(build_combined(filtered, filtered_items)),
                file_name="manifestiq_combined.csv",
                mime="text/csv",
                width="stretch",
                help="One row per line item, with the document details repeated on each row.",
            )

    # Line items
    with tab_items:
        if filtered_items.empty:
            st.info("No line items for the current selection.")
        else:
            ref_cols = [c for c in ["id", "document_number", "document_type", "po_number", "tracking_number"] if c in filtered.columns]
            labelled = filtered_items.merge(
                filtered[ref_cols].rename(columns={"id": "document_id"}),
                on="document_id",
                how="left",
            )
            lead_cols = [c for c in ["document_number", "document_type", "po_number", "tracking_number"] if c in labelled.columns]
            labelled = labelled[lead_cols + [c for c in labelled.columns if c not in lead_cols]]
            st.dataframe(labelled.drop(columns=["id", "document_id"], errors="ignore"), width="stretch", hide_index=True)
            st.download_button(
                "Download line items (CSV)",
                data=to_csv_bytes(labelled),
                file_name="manifestiq_line_items.csv",
                mime="text/csv",
            )

    # One document in detail
    with tab_detail:
        if filtered.empty:
            st.info("No documents match your filters.")
        else:
            def label(row):
                number = row.get("document_number") or "No number"
                dtype = row.get("document_type") or "Unknown type"
                when = str(row.get("created_at") or "")[:10]
                return f"{number}  |  {dtype}  |  {when}  |  {str(row['id'])[:8]}"

            options = {label(row): row["id"] for _, row in filtered.iterrows()}
            chosen = st.selectbox("Choose a document", list(options.keys()))
            chosen_id = options[chosen]

            doc_row = filtered[filtered["id"] == chosen_id].iloc[0]
            doc_items = items_df[items_df["document_id"] == chosen_id] if "document_id" in items_df.columns else items_df.iloc[0:0]

            st.markdown("#### Shipment details")
            details = {
                key.replace("_", " ").title(): value
                for key, value in doc_row.drop(labels=["id"], errors="ignore").items()
                if pd.notna(value) and value != ""
            }
            st.table(pd.DataFrame({"Value": {k: str(v) for k, v in details.items()}}))

            st.markdown("#### Line items")
            if doc_items.empty:
                st.caption("This document has no line items.")
            else:
                st.dataframe(doc_items.drop(columns=["id", "document_id"], errors="ignore"), width="stretch", hide_index=True)

            st.download_button(
                "Download this document (CSV)",
                data=to_csv_bytes(build_combined(filtered[filtered["id"] == chosen_id], doc_items)),
                file_name=f"manifestiq_{str(chosen_id)[:8]}.csv",
                mime="text/csv",
            )


# NAVIGATION

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width="stretch")
    else:
        st.title(APP_NAME)
    st.caption(APP_TAGLINE)
    page = st.radio("Go to", ["Scan Document", "Records & Export"], label_visibility="collapsed")

if page == "Scan Document":
    scan_page()
else:
    records_page()