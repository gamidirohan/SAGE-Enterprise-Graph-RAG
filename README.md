# PDF Chatbot with Knowledge Graph in Neo4j

This is an **<*ongoing*>** personal project aimed at building a pipeline to extract structured data from **unstructured PDFs**, store it in a **Neo4j knowledge graph**, and enable **natural language querying** using a **Retrieval-Augmented Generation (RAG) approach**.

The project is inspired by:
- [Neo4J - Enhancing the Accuracy of RAG Applications With Knowledge Graphs](https://neo4j.com/developer-blog/enhance-rag-knowledge-graph/?mkt_tok=NzEwLVJSQy0zMzUAAAGTBn-WDr1KcupEPExYL6rh_DaP3R0h5gWQFxWGRm6dXiew5-oAnYBbvXvedknjyhyojNebyUa0ywWZwIkZQRtiJ-9x6k22vY3ru2Ztp7PjlgN5Bbs)

> **Stack:** Python, Streamlit, LangChain, Neo4J, PyPDF2, DeepSeek R1

---

## 🚀 How to Run

### **1️⃣ Prerequisites**
Ensure you have:
✅ [Neo4j](https://neo4j.com/download/) installed and running locally  
✅ Python environment set up (`Python 3.8+`)

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Create a `.env` file with Neo4j credentials**
```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j
```

---

## 📌 The Pipeline

### `pipeline.py` -> Main script to process PDFs
1. **Extracts text** from PDFs in the `files/` folder using PyPDF2.
2. **Generates structured data** using an **LLM (DeepSeek R1 Distill Llama-70B)** to extract:
   - Entities
   - Relationships
   - Agile/SDLC details
   - Meetings, bills, signatures, etc.
3. **Generates a unique `doc_id`** based on **SHA-256 hashing of the PDF content** to prevent duplicates.
4. **Stores extracted data** in a **Neo4j graph database**.

#### **🛠 Run the pipeline**
```bash
streamlit run pipeline.py
```

#### **🕵️‍♂️ Verify Neo4j Data**
After running the pipeline, **check stored entities & relationships**:
```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
```
Visit [http://localhost:7474/browser/](http://localhost:7474/browser/) to visualize the graph.

---

## 📖 The Graph RAG (Upcoming Feature)
🚧 **In progress:** Natural language querying using LangChain & Neo4j Cypher search.

1. Query Neo4J with a natural language question.
2. Return answers based on the graph structure.

---

## 🛠 GitHub Workflow - Commands to Push Updates Regularly
Run these commands **regularly** to keep your GitHub repo updated:

### **📌 First-time setup**
```bash
git init
git remote add origin <your-github-repo-url>
git branch -M main
```

### **📌 Daily Workflow**
```bash
git pull origin main  # Get latest updates
git add .             # Add all changes
git commit -m "Updated pipeline and Graph RAG"  # Write meaningful commit messages
git push origin main
```

---

### 🌟 Future Improvements
- ✅ Integrate **Graph RAG** for natural language search in Neo4J.
- ✅ Improve **question flexibility** by mapping synonyms to graph nodes.
- ✅ Optimize **entity linking** using better LLM embeddings.

---