import os
import fitz  # PyMuPDF
import pdfplumber
import io
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import json
import streamlit as st
from collections import defaultdict
import pytesseract
from fuzzywuzzy import fuzz

# --- Configuration ---
load_dotenv()

# Set Tesseract path if you are on Windows and it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load API key from environment variables
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    st.error("FATAL: GOOGLE_API_KEY environment variable not set. Please create a .env file or set the environment variable.")
    st.stop()

# --- Prompts ---
TEXT_PROMPT = """
You are an expert accounts payable specialist. Your task is to analyze the following text content from an invoice and a purchase order and extract key information.

The INVOICE text may contain one or more distinct invoices. You must aggregate the data from all of them.

From the INVOICE text, extract:
- Invoice Number: List all unique invoice numbers found, separated by a comma.
- Purchase Order (PO) Number: Use the PO number that is common across all invoices.
- Date: Use the date from the latest invoice found.
- Vendor Name: Extract the vendor name. This is the name of the company issuing the invoice, typically found in the header, sender's address, or contact information section. It is not the customer's name (which is usually NADEC or The National Agricultural Development Company). Look for phrases like the company name followed by address, phone, or email. Common examples might include "International N&H Denmark ApS" or similar; ignore logos or brand names like "DANISCO" if a full legal entity name is present.
- A list of all line items: Find all line items across ALL invoices in the text. If an item with the same description appears on multiple invoices or multiple times, you MUST sum their quantities and calculate the total price accordingly.
- Total Amount: Sum the total amounts from ALL invoices found in the text.

From the PURCHASE ORDER text, extract:
- PO Number
- Date
- Vendor Name
- A list of all ordered items. Each item should have a 'description', 'quantity', and 'price'.
- Total Amount

Return your findings ONLY as a single, minified JSON object. The JSON structure must be:
{
  "invoice_data": {
    "invoice_no": "...", "po_no": "...", "date": "...", "vendor": "...",
    "items": [{"description": "...", "quantity": 1, "price": 0.00}],
    "total": 0.00
  },
  "po_data": {
    "po_no": "...", "date": "...", "vendor": "...",
    "items": [{"description": "...", "quantity": 1, "price": 0.00}],
    "total": 0.00
  }
}
"""

IMAGE_PROMPT = TEXT_PROMPT # Use the same powerful prompt for images

# --- Gemini API Interaction ---
def get_gemini_response(payload):
    model = genai.GenerativeModel('models/gemini-pro-latest')
    response = None
    try:
        generation_config = genai.types.GenerationConfig(temperature=0)
        response = model.generate_content(payload, generation_config=generation_config)
        json_text = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(json_text)
    except Exception as e:
        st.error(f"An error occurred with the Gemini API or its response: {e}")
        if response:
            st.write("Raw Gemini response:", response.text)
        return None

# --- Helpers ---
def get_text_from_pdf(file_path):
    """Attempts to extract text using pdfplumber, falls back to OCR if it fails."""
    try:
        text_content = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2)
                if page_text:
                    text_content += page_text + "\n"
        if text_content.strip():
            return text_content.strip()
    except Exception as e:
        st.warning(f"Text extraction with pdfplumber failed: {e}. Falling back to OCR.")
    
    try:
        text_content = ""
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text_content += pytesseract.image_to_string(img) + "\n"
        doc.close()
        return text_content.strip()
    except Exception as e:
        st.error(f"OCR with Tesseract failed: {e}")
        return ""

# --- Item Aggregation Logic ---
def normalize_and_aggregate_items(items):
    """
    Takes a list of item dicts, aggregates quantities for identical items,
    and returns a clean list for display.
    """
    if not isinstance(items, list):
        return []
    normalized = defaultdict(lambda: {"quantity": 0, "description": "", "price": 0.0})
    for item in items:
        if not isinstance(item, dict) or not item.get("description"): continue
        desc_key = item.get("description", "").strip().lower()
        if desc_key.startswith("culture "):
            desc_key = desc_key[len("culture "):]
        
        if not normalized[desc_key]["description"]:
            normalized[desc_key]["description"] = item.get("description")
        try:
            quantity = float(item.get("quantity", 0))
            unit_price = float(str(item.get("price", 0.0)).replace(',', '.'))
        except (ValueError, TypeError):
            quantity, unit_price = 0, 0.0
        normalized[desc_key]["quantity"] += quantity
        if unit_price > 0:
             normalized[desc_key]["price"] = unit_price
    return list(normalized.values())

# --- Streamlit UI ---
st.set_page_config(page_title="Invoice & PO Matching Tool", layout="wide")
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
    .card { background-color: #ffffff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); padding: 20px; margin-bottom: 20px; }
    .header { background-color: #1f2937; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .sidebar-card { background-color: #f9fafb; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    .table-header { background-color: #f9fafb; font-weight: 600; }
    .status-approved { color: #15803d; font-weight: bold; }
    .status-review { color: #b91c1c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📑 Invoice & PO Matching Tool</h1><p class="mt-2 text-sm">Upload an Invoice and Purchase Order to automatically compare and verify their details.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.subheader("Upload Documents")
    invoice_file = st.file_uploader("📄 Invoice", type=["pdf"], key="invoice")
    po_file = st.file_uploader("📑 Purchase Order", type=["pdf"], key="po")
    
    if st.button("🔍 Compare Documents", key="compare", use_container_width=True):
        if invoice_file and po_file:
            with open(invoice_file.name, "wb") as f:
                f.write(invoice_file.getbuffer())
            with open(po_file.name, "wb") as f:
                f.write(po_file.getbuffer())
            
            with st.spinner("Analyzing documents... This may take a moment."):
                invoice_text = get_text_from_pdf(invoice_file.name)
                po_text = get_text_from_pdf(po_file.name)
                
                if invoice_text and po_text:
                    payload = [TEXT_PROMPT, f"--- INVOICE TEXT ---\n{invoice_text}", f"--- PO TEXT ---\n{po_text}"]
                    st.session_state['analysis'] = get_gemini_response(payload)
                else:
                    st.error("Failed to extract text from one or both documents.")
                    st.session_state['analysis'] = None

            os.remove(invoice_file.name)
            os.remove(po_file.name)
        else:
            st.error("Please upload both an Invoice and a Purchase Order file.")
    st.markdown('</div>', unsafe_allow_html=True)

if 'analysis' in st.session_state and st.session_state['analysis']:
    analysis = st.session_state['analysis']
    invoice_data = analysis.get('invoice_data', {})
    po_data = analysis.get('po_data', {})

    col1, col2 = st.columns(2)

    def display_doc(title, data, doc_type):
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="text-xl font-semibold mb-3">{title}</h2>', unsafe_allow_html=True)
        
        doc_no_key = "invoice_no" if doc_type == "invoice" else "po_no"
        st.markdown(f'<p><strong>{doc_type.capitalize()} #:</strong> {data.get(doc_no_key, "N/A")}</p>', unsafe_allow_html=True)
        if doc_type == 'invoice':
            st.markdown(f'<p><strong>PO #:</strong> {data.get("po_no", "N/A")}</p>', unsafe_allow_html=True)
        
        st.markdown(f'<p><strong>Date:</strong> {data.get("date", "N/A")}</p>', unsafe_allow_html=True)
        st.markdown(f'<p><strong>Vendor:</strong> {data.get("vendor", "N/A")}</p>', unsafe_allow_html=True)
        st.markdown('<h3 class="text-lg font-medium mt-4">Items</h3>', unsafe_allow_html=True)
        
        items = normalize_and_aggregate_items(data.get("items", []))
        
        if items:
            table_html = '<table class="w-full border-collapse mt-2"><thead><tr class="table-header">'
            table_html += '<th class="p-2 text-left border">Description</th><th class="p-2 text-left border">Quantity</th><th class="p-2 text-left border">Price</th></tr></thead><tbody>'
            for item in items:
                table_html += f'<tr><td class="p-2 border">{item.get("description", "N/A")}</td>'
                table_html += f'<td class="p-2 border">{item.get("quantity", 0)}</td>'
                table_html += f'<td class="p-2 border">${item.get("price", 0.0):,.2f}</td></tr>'
            table_html += '</tbody></table>'
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.markdown('<p class="text-gray-500">No items found.</p>', unsafe_allow_html=True)
        
        total = data.get("total", 0.0)
        try:
            total_float = float(str(total).replace(',','.'))
        except (ValueError, TypeError):
            total_float = 0.0
        st.markdown(f'<h3 class="text-lg font-bold mt-4">Total: ${total_float:,.2f}</h3>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col1:
        display_doc("📄 Invoice Details", invoice_data, "invoice")
    with col2:
        display_doc("📑 Purchase Order Details", po_data, "po")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="text-xl font-semibold">🔎 Match/Mismatch Summary</h2>', unsafe_allow_html=True)

    def generate_match_summary(invoice_data, po_data):
        lines, issues = [], []

        def get_normalized_dict(items):
            normalized = defaultdict(lambda: {"quantity": 0, "total_price": 0.0, "description": ""})
            if not isinstance(items, list): return normalized
            for item in items:
                if not isinstance(item, dict) or not item.get("description"): continue
                desc_key = item.get("description", "").strip().lower()
                if desc_key.startswith("culture "):
                    desc_key = desc_key[len("culture "):]
                
                normalized[desc_key]["description"] = item.get("description")
                try:
                    quantity = float(item.get("quantity", 0))
                    unit_price = float(str(item.get("price", 0.0)).replace(',','.'))
                except (ValueError, TypeError):
                    quantity, unit_price = 0, 0.0
                
                normalized[desc_key]["quantity"] += quantity
                normalized[desc_key]["total_price"] += unit_price * quantity
            return normalized

        inv_po_no_raw = invoice_data.get("po_no")
        po_po_no_raw = po_data.get("po_no")
        inv_po_no = str(inv_po_no_raw).strip() if inv_po_no_raw is not None else "N/A"
        po_po_no = str(po_po_no_raw).strip() if po_po_no_raw is not None else "N/A"
        if inv_po_no == po_po_no and inv_po_no != "N/A":
            lines.append(f"• PO Number matches: **{po_po_no}** ✓")
        else:
            lines.append(f"• PO Number mismatch: Invoice ({inv_po_no}) vs PO ({po_po_no}) ✗")
            issues.append("PO number mismatch")

        inv_vendor_raw = invoice_data.get("vendor")
        po_vendor_raw = po_data.get("vendor")
        inv_vendor = str(inv_vendor_raw).strip().lower().replace(' ', '') if inv_vendor_raw else ""
        po_vendor = str(po_vendor_raw).strip().lower().replace(' ', '') if po_vendor_raw else ""
        if inv_vendor and po_vendor and (inv_vendor in po_vendor or po_vendor in inv_vendor):
            lines.append(f"• Vendor matches: **{invoice_data.get('vendor')}** ✓")
        else:
            lines.append(f"• Vendor mismatch: Invoice ({inv_vendor_raw or 'N/A'}) vs PO ({po_vendor_raw or 'N/A'}) ✗")
            issues.append("Vendor mismatch")

        # --- MODIFIED: Stricter Total Amount Check ---
        invoice_total = float(str(invoice_data.get("total", 0.0)).replace(',','.'))
        po_total = float(str(po_data.get("total", 0.0)).replace(',','.'))
        if abs(invoice_total - po_total) < 0.01:
            lines.append(f"• Total amount matches: **${invoice_total:,.2f}** ✓")
        else:
            # Any difference is now treated as a critical mismatch
            lines.append(f"• **Total amount mismatch**: Invoice (${invoice_total:,.2f}) vs PO (${po_total:,.2f}) ✗")
            issues.append("Total amount mismatch")

        normalized_invoice_items = get_normalized_dict(invoice_data.get("items", []))
        normalized_po_items = get_normalized_dict(po_data.get("items", []))

        lines.append("---")

        for inv_key, inv_item in normalized_invoice_items.items():
            if inv_key in normalized_po_items:
                po_item = normalized_po_items[inv_key]
                display_desc = inv_item.get('description')
                
                if inv_item['quantity'] > po_item['quantity'] + 0.001:
                    lines.append(f"• **Quantity mismatch** for '{display_desc}': Invoice ({inv_item['quantity']}) **exceeds** PO quantity ({po_item['quantity']}) ✗")
                    issues.append("Item quantity exceeds PO")
                elif inv_item['quantity'] < po_item['quantity'] - 0.001:
                    lines.append(f"• Quantity for '{display_desc}' is a **partial shipment**: Invoice ({inv_item['quantity']}) of PO ({po_item['quantity']}) ⚠️")
                else:
                    lines.append(f"• Quantity for '{display_desc}' matches fully. ✓")

                if inv_item['total_price'] > po_item['total_price'] + 0.01:
                    lines.append(f"• **Line total mismatch** for '{display_desc}': Invoice (${inv_item['total_price']:,.2f}) **exceeds** PO total (${po_item['total_price']:,.2f}) ✗")
                    issues.append("Line total exceeds PO")
            else:
                lines.append(f"• Item '{inv_item.get('description')}' on invoice could not be found on the PO. ✗")
                issues.append("Unmatched invoice item")
        
        if not issues:
            lines.append('<span class="status-approved">→ Status: APPROVED ✅</span>')
        else:
            lines.append('<span class="status-review">→ Status: NEEDS REVIEW ⚠️ - Critical discrepancies found.</span>')
        
        return "<br>".join(lines)

    match_summary = generate_match_summary(invoice_data, po_data)
    st.markdown(match_summary, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)