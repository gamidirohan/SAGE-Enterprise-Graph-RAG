# SAGE Enterprise Graph RAG
SAGE (Structured Augmented Generation Engine) is a completed personal project designed as an enterprise-grade system that extracts structured data from unstructured documents (PDFs, text files, messages), stores it in a Neo4j knowledge graph, and enables natural language querying using a Retrieval-Augmented Generation (RAG) approach with multi-hop reasoning to provide more accurate and comprehensive answers.

The project is inspired by:
- [Neo4J - Enhancing the Accuracy of RAG Applications With Knowledge Graphs](https://neo4j.com/developer-blog/enhance-rag-knowledge-graph/?mkt_tok=NzEwLVJSQy0zMzUAAAGTBn-WDr1KcupEPExYL6rh_DaP3R0h5gWQFxWGRm6dXiew5-oAnYBbvXvedknjyhyojNebyUa0ywWZwIkZQRtiJ-9x6k22vY3ru2Ztp7PjlgN5Bbs)

> **Stack:** Python, FastAPI, LangChain, Neo4J, PyPDF2, Groq API, SentenceTransformers

---

## 🚀 How to Run

### **1️⃣ Prerequisites**
Ensure you have:
✅ [Neo4j](https://neo4j.com/download/) installed and running locally
✅ Python environment set up (`Python 3.8+`)
✅ [Groq API key](https://console.groq.com/) for LLM access

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Create a `.env` file with credentials**
```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### **4️⃣ Project Components**

| File | Description |
|------|-------------|
| `backend.py` | FastAPI backend server for SAGE Graph RAG |
| `message_processor.py` | Processes message files and generates QA pairs |
| `pipeline.py` | Processes PDF documents and extracts structured data |
| `graph_rag.py` | Core graph-based RAG implementation |
| `performance_comparison.py` | Framework for comparing SAGE vs traditional RAG |
| `generate_report.py` | Generates performance reports with visualizations |
| `run_performance_comparison.py` | Script to run performance comparisons |
| `qa_pairs.json` | Sample question-answer pairs for testing |
| `test_queries.json` | Test queries for performance evaluation |

---

## 📌 Data Processing Pipeline

### Document Processing (`pipeline.py`)
The pipeline processes documents (PDFs, text files) and extracts structured data for the knowledge graph:

1. **Text Extraction**: Extracts text from PDFs in the `files/` folder using PyPDF2
2. **Entity Extraction**: Uses LLM (DeepSeek R1 Distill Llama-70B) to identify:
   - People, organizations, locations
   - Projects, tasks, deadlines
   - Meetings, events, communications
3. **Relationship Extraction**: Identifies connections between entities
4. **Document Chunking**: Splits documents into semantic chunks for better retrieval
5. **Graph Storage**: Stores all extracted data in Neo4j with proper relationships

#### **🛠 Run the Document Pipeline**
```bash
python pipeline.py
```

### Message Processing (`message_processor.py`)
The message processor handles chat messages and conversation data:

1. **Message Extraction**: Processes message files from the `uploads/` directory
2. **Structured Data Extraction**: Extracts sender, receiver, content, and metadata
3. **Graph Integration**: Adds messages to the knowledge graph with proper relationships
4. **QA Pair Generation**: Creates question-answer pairs for testing and evaluation

#### **🛠 Run the Message Processor**
```bash
python message_processor.py --directory uploads
```

To process messages and generate QA pairs:
```bash
python message_processor.py --directory uploads --num-pairs 30 --output qa_pairs.json
```

### **🕵️‍♂️ Verify Neo4j Data**
After running the pipelines, check stored entities & relationships:
```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
```
Visit [http://localhost:7474/browser/](http://localhost:7474/browser/) to visualize the graph.

---

## 📖 SAGE Graph RAG Backend

### SAGE Architecture

SAGE (Structured Augmented Generation Engine) is built on a graph-based RAG architecture that leverages the knowledge graph structure to provide more accurate and comprehensive answers.

Key components:
1. **Vector Search**: Initial retrieval based on embedding similarity
2. **Graph Traversal**: Multi-hop exploration of relationships between entities
3. **Entity Recognition**: Identification of entities mentioned in queries
4. **Relationship-aware Prompting**: LLM prompts that leverage graph structure
5. **Chain-of-Thought Reasoning**: Step-by-step reasoning through graph relationships

![SAGE Architecture](https://mermaid.ink/img/pako:eNp1kk1PwzAMhv9KlBOgSf3YpGlw2A6cEBJiB8QOaRo8L6ImJXVaVVX_O0lXWAcb18Tv48R27G0kOWJkIrOWXDtQkjxYp6Ej2-Ej-QZVgYfkBh6h0bpFSJLb5O4eXpNkeZ_cLuEpSZI0XcIiTZcwQzWgNLCGDrSEHIUDVKgKlA5qVNZJaLXXYLRFaVFYVLVWDmrfCFDWoXLQGV2Bk9Cg9KAcNKLWDdRCOdBGQyeVhVbXQkCHfYdKCGhEI6AWrRDQG-3Bqq_Ry_8Ys_kcZrPZfA6z-WJxdZnfnOWz9Gx1Oj2ZnB5PJqPR6GiUjsfj0XjyC_tnNhqfTH_Yx6PRYe_Vwa_3_uAXfHiQHuTBwf9gv-DD_eFwOByWWGhTYhZmXlvMwlxrLLIwb7DIwjzAIgvzGoss_Gxh5rXFLMxKY5GFWWERZnLnUYZ5i1mYOdRhJjsUYe5Rh7nBIszk1qMO846yMJO9RRHmDnWYyc6hCPMWdZjJrQ7zHZowk3uLOsxKFGEmDw51mLcowkweHOow31GEmTw61GFWoQgzeXKow6xGEWby7FCH-YEizOTFoQ6zBkWYyatDHeYtijCTN4c6zDsUYSbvDnWY9yjCTD4c6jA_UISZfDrUYX6iCDP5cqjDrEURZvLtUIdZhyLM5MehDvMTRZjJr0Md5heKMJM_hzrMbxRhJgfqMOtRhJkcqcOsQRFmcqIOs1bFv9S_1Bc?type=png)

### Running SAGE Backend (`backend.py`)
The backend provides a FastAPI server with endpoints for:
- Chat interface with graph-based RAG
- Document processing and storage
- Graph debugging and visualization

```bash
python backend.py
```

This will start the FastAPI backend server on http://localhost:8000 with the following endpoints:
- `/api/chat` - Process chat messages using SAGE Graph RAG
- `/api/process-document` - Upload and process documents
- `/api/debug-graph` - Get information about the graph structure
- `/api/health` - Health check endpoint

## 📊 Performance Comparison Framework

### Performance Comparison (`performance_comparison.py`)

The performance comparison framework evaluates SAGE Graph RAG against traditional RAG approaches using various metrics and models.

#### Key Features:
- **Multi-model evaluation**: Test with different LLM and embedding models
- **Comprehensive metrics**: Quality, latency, preference, and similarity scores
- **Visualization**: Generate graphs and heatmaps for easy analysis
- **Batch processing**: Run multiple comparisons with different configurations

#### Running Performance Comparison

1. **Basic Comparison**
   ```bash
   python performance_comparison.py --queries qa_pairs.json --output results/comparison_results.json --llm-models "llama3-8b-8192" --embedding-models "all-mpnet-base-v2"
   ```

2. **Multi-Model Comparison**
   ```bash
   python performance_comparison.py --queries qa_pairs.json --output results/multi_model_results.json --llm-models "gemma2-9b-it,llama3-8b-8192,llama3-70b-8192" --embedding-models "all-mpnet-base-v2,multi-qa-mpnet-base-dot-v1"
   ```

3. **Generate Performance Report** (`generate_report.py`)
   ```bash
   python generate_report.py --results results/comparison_results.json --output results/performance_report.html
   ```

### Performance Metrics

The framework evaluates:

1. **Answer Quality**: LLM-based evaluation of answer quality on a scale of 1-10
2. **System Preference**: Which system (SAGE or Traditional RAG) provides better answers
3. **Latency**: Response time for each system
4. **Answer Similarity**: How similar the answers from both systems are (F1 score)

### Available Models

**LLM Models (via Groq API)**:
- `gemma2-9b-it` - Gemma 2 9B Instruct
- `llama-guard-3-8b` - Llama Guard 3 8B
- `mistral-saba-24b` - Mistral Saba 24B
- `llama3-8b-8192` - Llama 3 8B
- `compound-beta-mini` - Compound Beta Mini
- `deepseek-r1-distill-llama-70b` - DeepSeek R1 Distill Llama 70B
- `llama-3.3-70b-versatile` - Llama 3.3 70B Versatile
- `llama3-70b-8192` - Llama 3 70B
- `llama-3.1-8b-instant` - Llama 3.1 8B Instant

**Embedding Models**:
- `all-mpnet-base-v2` - MPNet Base v2 (Default)
- `all-MiniLM-L6-v2` - MiniLM L6 v2 (Faster, smaller)
- `multi-qa-mpnet-base-dot-v1` - MPNet Base optimized for question-answering
- `all-distilroberta-v1` - DistilRoBERTa v1 (Good balance of speed and quality)
- `paraphrase-multilingual-mpnet-base-v2` - Multilingual MPNet Base v2

### Batch Processing

For convenience, we've included batch files to run the performance comparison pipeline:

1. **Quick Comparison** (single model, 10 questions)
   ```bash
   run_quick_comparison.bat
   ```

2. **Comprehensive Comparison** (recommended)
   ```bash
   run_comprehensive_comparison.bat
   ```
   This batch file runs a complete evaluation with multiple models and generates detailed reports with visualizations.

### Running the Full Pipeline

To run the complete pipeline from data processing to performance evaluation:

1. Process message files:
   ```bash
   python message_processor.py --directory uploads --skip-qa
   ```

2. Generate QA pairs:
   ```bash
   python message_processor.py --skip-processing --num-pairs 30
   ```

3. Run performance comparison:
   ```bash
   python performance_comparison.py --queries qa_pairs.json --output results/comparison_results.json
   ```

4. Generate report:
   ```bash
   python generate_report.py --results results/comparison_results.json
   ```

### Sample Results

Our optimized SAGE Graph RAG with llama-3.3-70b-versatile and multi-qa-mpnet-base-dot-v1 achieved:

- **Higher Quality Answers**: SAGE Graph RAG scored 7.60/10 compared to Traditional RAG's 6.30/10
- **Better Performance**: SAGE was rated better in 8 out of 10 queries
- **Comparable Latency**: SAGE's latency (2.47s) was only slightly higher than Traditional RAG (2.11s)
- **Higher Answer Similarity**: The similarity between SAGE and Traditional RAG answers was 0.5636

---

## 🛠 GitHub Workflow

### **📌 First-time setup**
```bash
git init
git remote add origin https://github.com/gamidirohan/SAGE-Enterprise-Graph-RAG.git
git branch -M main
```

### **📌 Development Workflow**
```bash
# Pull latest changes
git pull origin graph_rag_backend_for_nextjs

# Add changes
git add .

# Commit with detailed message
git commit -m "Update: Cleaned up codebase, improved QA pairs, and enhanced documentation"

# Push to branch
git push origin graph_rag_backend_for_nextjs
```

---

## 🌟 Project Status and Roadmap

### ✅ Completed Features
- **Backend API**: FastAPI backend with comprehensive endpoints for chat, document processing, and graph debugging
- **Document Processing**: Pipeline for extracting structured data from PDFs and text files
- **Message Processing**: System for handling chat messages and generating QA pairs
- **Graph-based RAG**: SAGE implementation with multi-hop traversal and relationship-aware prompting
- **Performance Comparison**: Framework for evaluating SAGE against traditional RAG approaches
- **Multi-model Support**: Integration with various LLM and embedding models via Groq API
- **Visualization**: Comprehensive reporting with graphs and heatmaps

### 🔄 In Progress
- **Frontend Integration**: Connecting the backend with a NextJS frontend
- **Real-time Updates**: Implementing WebSocket for real-time chat updates
- **Conversation History**: Adding support for follow-up questions and context retention
- **Advanced Entity Extraction**: Improving entity and relationship extraction accuracy

### 🔮 Future Roadmap
- **Automated Knowledge Graph Expansion**: Dynamically expanding the graph based on user queries
- **Custom Embedding Models**: Training domain-specific embedding models
- **Explainability Features**: Visualizing reasoning paths in the graph
- **Multi-modal Support**: Adding support for images and other media types
- **Enterprise Integration**: Connecting with enterprise data sources and authentication systems

---
