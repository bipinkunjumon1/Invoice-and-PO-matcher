📑 Invoice & PO Matching Tool
Automate Accounts Payable Matching with AI + OCR

An intelligent document processing application built using Streamlit and Google's Gemini API, designed to automate the two-way and three-way matching process for Accounts Payable (AP) departments.

The tool extracts key data from uploaded Invoices and Purchase Orders (POs) — even from scanned PDFs — and compares them side-by-side. It intelligently aggregates line items, detects mismatches, and provides a color-coded summary indicating whether the documents are Approved ✅ or Need Review ⚠️.

✨ Key Features
🧠 1. Intelligent Data Extraction (Gemini-Pro)

Leverages Google’s Gemini-Pro model to read and interpret both text-based and scanned PDFs.

Extracts key invoice and PO fields such as:

Invoice Number

PO Number

Date

Vendor Name

Line Items (Description, Quantity, Price)

Total Amount

🧾 2. Multi-Invoice Handling

Capable of parsing multiple invoices contained in a single PDF file.

Automatically aggregates totals and quantities across all invoices.

🧮 3. Item Aggregation & Normalization

Combines line items with matching descriptions (even if worded slightly differently).

Sums up quantities and calculates total prices automatically.

🔗 4. Two-Way & Three-Way Matching Logic

Compares critical fields across documents:

Field	Logic
PO Number	Must exactly match between Invoice and PO
Vendor Name	Fuzzy matching used to tolerate minor text differences
Total Amount	Strict numerical comparison
Line Items & Quantities	Detects overbilling, partial shipments, and mismatched pricing
💻 5. Clean, Intuitive Streamlit UI

Simple drag-and-drop upload for Invoice and PO PDFs.

Modern TailwindCSS-inspired design.

Two-column layout showing Invoice Details and PO Details side-by-side.

🟩 6. Status Indicators

✅ APPROVED — All details match perfectly.

⚠️ NEEDS REVIEW — Discrepancies detected (quantity, total, or missing item).

Detailed text-based report summarizing matches and mismatches.

🛠️ Prerequisites

Before running the app, make sure you have the following installed:

Requirement	Description
Python 3.8+	Required to run the Streamlit app
Google AI API Key	Get your Gemini API key
 and store it in .env
Tesseract OCR	Needed for extracting text from scanned image PDFs
pip & virtualenv	For managing dependencies
🧩 Installing Tesseract OCR

Windows:

Download the official installer from Tesseract GitHub
.

Update the path in your script if needed:

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


Linux:

sudo apt install tesseract-ocr


macOS:

brew install tesseract

🚀 Setup & Installation

Follow these steps to get started:

1️⃣ Clone the Repository
git clone https://github.com/<your-username>/invoice-po-matcher.git
cd invoice-po-matcher

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows

3️⃣ Add Your Gemini API Key

Create a .env file in the root directory and paste your API key:

GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run the Application
streamlit run app.py