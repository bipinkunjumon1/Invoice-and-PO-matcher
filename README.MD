# Intelligent Invoice & PO Matching Tool


An intelligent document processing application built with **Streamlit** and **Google's Gemini API** to automate the two-way and three-way matching process for Accounts Payable (AP). This tool streamlines workflows by extracting key data from PDF Invoices and Purchase Orders, performing intelligent comparisons, and providing a clear, actionable summary for AP teams.

<br>



---

## ✨ Core Features

* **🤖 AI-Powered Data Extraction:** Leverages the multi-modal capabilities of **Google's Gemini Pro Vision** to intelligently extract key fields (Invoice #, PO #, Dates, Vendor, Line Items, Totals) from both digital and scanned PDFs.
* **⚖️ Automated 2-Way & 3-Way Matching:** Implements robust logic to compare critical data points between invoices and purchase orders:
    * **PO Number & Vendor Name:** Ensures the invoice corresponds to the correct PO and supplier (with fuzzy matching for vendor names).
    * **Total Amounts:** Verifies that the invoice total matches the PO total.
    * **Line Item Details:** Aggregates and compares item descriptions, quantities, and prices to flag over-billing or quantity discrepancies.
* **📊 Intuitive UI & Verification:** A clean, user-friendly interface built with Streamlit provides a side-by-side comparison of extracted data, with color-coded highlights for easy identification of mismatches.
* **✅ Clear Status Approvals:** Each invoice is automatically assigned a clear status—**`APPROVED ✅`** or **`NEEDS REVIEW ⚠️`**—based on the matching results, allowing users to focus only on exceptions.
* **📄 Multi-Document Handling:** The model is prompted to correctly parse and aggregate data even when multiple invoices are present in a single text extraction.

---

## 🛠️ Technology Stack

* **Backend:** Python
* **AI Model:** Google Gemini Pro & Gemini Pro Vision
* **UI Framework:** Streamlit
* **PDF Processing:** PyMuPDF, Pytesseract (for OCR)
* **Environment:** python-dotenv, FuzzyWuzzy

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

* **Python 3.8** or higher.
* **Google AI API Key:** You must have a valid [Gemini API key](https://ai.google.dev/).
* **Tesseract OCR:** Required for reading image-based (scanned) PDFs.
    * **Windows:** Install from the [official Tesseract page](https://github.com/tesseract-ocr/tesseract) and ensure its installation directory is added to your system's PATH.
    * **Linux/macOS:** Install via a package manager.
        ```bash
        # Debian/Ubuntu
        sudo apt update && sudo apt install tesseract-ocr

        # macOS (using Homebrew)
        brew install tesseract
        ```

### 2. Installation & Setup

**A. Clone the Repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

**B. Create a Virtual Environment**
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**C. Install Dependencies**
```bash
pip install -r requirements.txt
```

**D. Configure Environment Variables**
Create a file named `.env` in the root of the project directory and add your Google Gemini API key.
```env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

### 3. Running the Application

Once the setup is complete, launch the Streamlit application with the following command:
```bash
streamlit run app.py
```
Your web browser should automatically open to the application's interface.

---

## 📋 Usage

1.  **Launch the app** using the command above.
2.  **Upload the Purchase Order (PO)** PDF using the file uploader on the left.
3.  **Upload the corresponding Invoice** PDF using the second file uploader.
4.  The application will process both documents, extract the relevant data, and perform the matching logic.
5.  **Review the results:**
    * The overall status (`APPROVED ✅` or `NEEDS REVIEW ⚠️`) will be displayed at the top.
    * A side-by-side table will show the extracted data from both documents.
    * Mismatched fields will be highlighted for immediate attention.
6. if output shows worng information, please try match again with the same documents once more.

---

