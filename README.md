# 🤖 Multimodal AI Financial Assistant

An intelligent financial document analyzer powered by AI vision and language models. Upload credit card statements, invoices, receipts, or expense reports (images or PDFs) and ask natural language questions about charges, totals, dates, and transactions.

## ✨ Features

- **Multimodal Document Processing**: Analyze both images and PDF documents
- **Vision-Language AI**: Uses Meta's Llama 4 Scout for image understanding and Llama 3.3 for text analysis
- **RAG-Enhanced Answers**: Retrieves relevant context from billing policies and previous documents
- **Structured Field Extraction**: Automatically extracts vendor, dates, line items, taxes, and totals
- **Conversational Interface**: Chat-based Q&A about your financial documents
- **Vector Store**: ChromaDB-powered semantic search for context retrieval
- **Observability**: Optional LangSmith integration for tracing and monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Groq API Key](https://console.groq.com/) (for LLM access)
- (Optional) [LangSmith API Key](https://smith.langchain.com/) (for tracing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/saket-tech/Multimodal-AI-assistant.git
cd Multimodal-AI-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
LANGSMITH_PROJECT=multimodal-financial-assistant
CHROMA_DIR=./chroma_db
```

5. Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

### Upload a Document
1. Click the file uploader and select an image (PNG, JPG, JPEG) or PDF
2. The document will be displayed/extracted automatically
3. For PDFs, structured fields are extracted and stored in the vector database

### Ask Questions
Type questions in the chat interface, such as:
- "Why was this charge deducted?"
- "What is the total amount due?"
- "List all line items and amounts"
- "What are the taxes and fees?"
- "Who is the vendor for this charge?"
- "What was my previous balance?"

### How It Works
1. **Document Processing**: Images are encoded to base64, PDFs are parsed with pdfplumber
2. **Field Extraction**: Structured data (vendor, dates, amounts) is extracted using LLM
3. **Context Retrieval**: Relevant billing policies and prior documents are fetched from ChromaDB
4. **AI Analysis**: Vision/text model analyzes document with RAG context and answers your question
5. **Conversational Memory**: Chat history is maintained for follow-up questions

## 🏗️ Architecture

```
app.py                  # Streamlit UI and main orchestration
├── extractor.py        # Structured field extraction from documents
├── vector_store.py     # ChromaDB vector store for RAG context
└── chroma_db/          # Persisted vector database (gitignored)
```

### Key Components

**Models Used:**
- **Vision Model**: `meta-llama/llama-4-scout-17b-16e-instruct` (for image analysis)
- **Text Model**: `llama-3.3-70b-versatile` (for PDF text analysis)
- **Embedding Model**: `all-MiniLM-L6-v2` (for semantic search)

**Vector Store:**
- **Database**: ChromaDB with cosine similarity
- **Pre-seeded Policies**: Built-in billing policies for common charges (AWS, Netflix, taxes, etc.)
- **Document Storage**: Extracted documents are stored for future retrieval

## 📦 Dependencies

```
groq==0.13.1                    # Groq LLM client
streamlit==1.41.1               # Web UI framework
pillow==11.1.0                  # Image processing
pdfplumber==0.11.4              # PDF text extraction
python-dotenv==1.0.1            # Environment variable management
chromadb==0.5.23                # Vector database
sentence-transformers==3.3.1   # Embedding model
langsmith>=0.3.45               # Optional observability
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | API key from Groq Console |
| `LANGSMITH_API_KEY` | No | LangSmith tracing (optional) |
| `LANGSMITH_PROJECT` | No | Project name for LangSmith |
| `CHROMA_DIR` | No | Vector DB path (default: `./chroma_db`) |

### Models

To use different models, modify these constants in `app.py`:
```python
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"
```

Check [Groq's model list](https://console.groq.com/docs/models) for available options.

## 🛡️ Security Notes

- `.env` file is gitignored to protect API keys
- `chroma_db/` is gitignored (can be large and contains indexed data)
- Never commit sensitive credentials to version control
- Use `.env.example` as a template for required variables

## 📝 Example Use Cases

1. **Credit Card Statement Analysis**
   - Upload: Monthly credit card statement PDF
   - Ask: "What was the total amount charged for AWS services this month?"

2. **Invoice Verification**
   - Upload: Vendor invoice image
   - Ask: "Does this invoice include sales tax? What's the breakdown?"

3. **Expense Report Review**
   - Upload: Expense report screenshot
   - Ask: "List all business meal charges and their amounts"

4. **Receipt Tracking**
   - Upload: Restaurant receipt photo
   - Ask: "What was the tip percentage on this receipt?"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for fast LLM inference
- [Meta](https://ai.meta.com/) for Llama models
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Groq, Llama, and Streamlit**
