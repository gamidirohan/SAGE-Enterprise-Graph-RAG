# SAGE Enterprise Graph RAG - FastAPI Backend

This is the FastAPI backend for the SAGE Enterprise Graph RAG system. It provides API endpoints for document processing, chat functionality, and graph debugging.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your `.env` file is set up with the following variables:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_api_key
```

3. Start the FastAPI server:
```bash
uvicorn backend:app --reload
```

## API Endpoints

### Chat Endpoint
- **URL**: `/api/chat`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "message": "Your question here",
    "history": [] // Optional chat history
  }
  ```
- **Response**:
  ```json
  {
    "answer": "Response from the AI",
    "thinking": [] // Optional thinking process
  }
  ```

### Document Processing Endpoint
- **URL**: `/api/process-document`
- **Method**: POST
- **Request**: Form data with a file upload
- **Response**:
  ```json
  {
    "doc_id": "document_id",
    "sender": "sender_id",
    "receivers": ["receiver_id1", "receiver_id2"],
    "subject": "document_subject",
    "success": true,
    "message": "Status message"
  }
  ```

### Graph Debug Endpoint
- **URL**: `/api/debug-graph`
- **Method**: GET
- **Response**: Detailed information about the graph structure

### Health Check
- **URL**: `/api/health`
- **Method**: GET
- **Response**: `{"status": "ok"}`

## Integration with NextJS Frontend

To connect your NextJS frontend to this backend:

1. Use the Fetch API or Axios to make requests to the backend endpoints.
2. For document uploads, use FormData to send files.
3. For chat functionality, maintain chat history on the frontend and send the current message to the backend.

Example NextJS code for chat:
```javascript
const sendMessage = async (message) => {
  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      history: chatHistory,
    }),
  });
  
  const data = await response.json();
  return data;
};
```

Example NextJS code for document upload:
```javascript
const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/api/process-document', {
    method: 'POST',
    body: formData,
  });
  
  const data = await response.json();
  return data;
};
```
