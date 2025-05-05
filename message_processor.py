"""
Message Processor for SAGE Enterprise Graph RAG

This script:
1. Processes message files from the files1 directory
2. Extracts structured data from each message
3. Pushes the data to the Neo4j database
4. Generates question-answer pairs for evaluation
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import nltk
from nltk.tokenize import sent_tokenize
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Neo4j connection
def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize embedding model
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Function to generate embeddings
def generate_embedding(text):
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()

# Function to generate document ID
def generate_doc_id(content: str) -> str:
    """Generate a unique document ID based on the hashed content."""
    return hashlib.sha256(content.encode()).hexdigest()

# Function to chunk document
def chunk_document(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Split document into chunks of approximately max_chunk_size characters."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size <= max_chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_size
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to extract structured data from a message file
def extract_message_data(file_path: str) -> Dict[str, Any]:
    """Extract structured data from a message file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract sender, receiver, and message
        lines = content.strip().split('\n')

        # Parse the message format
        sender_id = None
        receiver_id = None
        message_text = None
        sent_time = None

        for line in lines:
            if line.startswith('Sender ID:'):
                sender_id = line.replace('Sender ID:', '').strip()
            elif line.startswith('Receiver ID:'):
                receiver_id = line.replace('Receiver ID:', '').strip()
            elif line.startswith('Message:'):
                message_text = line.replace('Message:', '').strip()
                # If message continues on multiple lines
                message_index = lines.index(line)
                if message_index < len(lines) - 1:
                    next_line_index = message_index + 1
                    while next_line_index < len(lines) and not lines[next_line_index].startswith('Sent Time:'):
                        message_text += '\n' + lines[next_line_index]
                        next_line_index += 1
            elif line.startswith('Sent Time:'):
                sent_time = line.replace('Sent Time:', '').strip()

        # Validate required fields
        if not sender_id or not receiver_id or not message_text:
            logger.error(f"Missing required fields in {file_path}")
            return None

        # Generate document ID
        doc_id = generate_doc_id(content)

        # Create a subject from the first few words of the message
        words = message_text.split()
        subject = ' '.join(words[:min(5, len(words))]) + '...'

        return {
            "doc_id": doc_id,
            "sender": sender_id,
            "receivers": [receiver_id],  # Single receiver per message
            "subject": subject,
            "content": message_text,
            "sent_time": sent_time
        }
    except Exception as e:
        logger.error(f"Error extracting data from {file_path}: {str(e)}")
        return None

# Function to store document in Neo4j
def store_in_neo4j(data: Dict[str, Any]) -> bool:
    """Store document and its chunks in Neo4j."""
    driver = get_neo4j_driver()
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
        groq_api_key=GROQ_API_KEY
    )

    try:
        with driver.session() as session:
            # First, check if document already exists
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                RETURN d
                """,
                doc_id=data["doc_id"]
            ).single()

            if result:
                logger.info(f"Document {data['doc_id']} already exists in the database.")
                return True

            # Create Document node
            document_summary = llm.invoke(f"Summarize this document, include the word json in the summary: {data['content']}").content
            embedding = generate_embedding(document_summary[:5000])
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.sender = $sender, d.subject = $subject, d.content = $content, d.embedding = $embedding, d.summary = $summary
                """,
                doc_id=data["doc_id"],
                sender=data["sender"],
                subject=data["subject"],
                content=data["content"],
                embedding=embedding,
                summary=document_summary
            )

            # Chunk and create Chunk nodes
            chunks = chunk_document(data["content"])
            for i, chunk in enumerate(chunks):
                chunk_summary = llm.invoke(f"Summarize this chunk, include the word json in the summary: {chunk}").content
                chunk_embedding = generate_embedding(chunk_summary)
                session.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.content = $content, c.embedding = $embedding, c.summary = $summary
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (c)-[:PART_OF]->(d)
                    """,
                    chunk_id=f"{data['doc_id']}-chunk-{i}",
                    content=chunk,
                    embedding=chunk_embedding,
                    summary=chunk_summary,
                    doc_id=data["doc_id"]
                )

            # Create Sender and Receiver nodes and relationships
            session.run(
                """
                MERGE (s:Person {id: $sender_id})
                MERGE (d:Document {doc_id: $doc_id})
                MERGE (s)-[:SENT]->(d)
                """,
                sender_id=data["sender"],
                doc_id=data["doc_id"]
            )

            for receiver in data["receivers"]:
                session.run(
                    """
                    MERGE (r:Person {id: $receiver_id})
                    MERGE (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:RECEIVED_BY]->(r)
                    """,
                    receiver_id=receiver,
                    doc_id=data["doc_id"]
                )

            logger.info(f"Successfully stored document {data['doc_id']} in Neo4j.")
            return True
    except Exception as e:
        logger.error(f"Error storing document in Neo4j: {str(e)}")
        return False
    finally:
        driver.close()

# Function to process all message files in a directory
def process_message_files(directory: str) -> List[Dict[str, Any]]:
    """Process all message files in the specified directory."""
    processed_data = []

    # Get all text files in the directory
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.endswith('.txt') and os.path.isfile(os.path.join(directory, f))]

    logger.info(f"Found {len(file_paths)} message files in {directory}")

    # Process files in batches to avoid overwhelming the database
    batch_size = 10
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")

        for file_path in batch:
            logger.info(f"Processing file: {file_path}")

            # Extract data from file
            data = extract_message_data(file_path)
            if data:
                # Store in Neo4j
                success = store_in_neo4j(data)
                if success:
                    processed_data.append(data)
                    logger.info(f"Successfully processed {file_path}")
                else:
                    logger.error(f"Failed to store data from {file_path} in Neo4j")
            else:
                logger.error(f"Failed to extract data from {file_path}")

    return processed_data

# Function to generate question-answer pairs
def generate_qa_pairs(num_pairs: int = 30) -> List[Dict[str, str]]:
    """Generate question-answer pairs based on the data in Neo4j."""
    driver = get_neo4j_driver()
    llm = ChatGroq(
        model_name="deepseek-r1-distill-llama-70b",
        temperature=0.7,  # Higher temperature for more diverse questions
        groq_api_key=GROQ_API_KEY
    )

    qa_pairs = []

    try:
        with driver.session() as session:
            # Get all documents
            documents = session.run(
                """
                MATCH (d:Document)
                RETURN d.doc_id AS doc_id, d.subject AS subject, d.content AS content, d.summary AS summary
                """
            ).data()

            if not documents:
                logger.error("No documents found in the database.")
                return []

            # Get all people
            people = session.run(
                """
                MATCH (p:Person)
                RETURN p.id AS id
                """
            ).data()

            person_ids = [p["id"] for p in people]

            # Generate different types of questions
            question_types = [
                # Document content questions
                "Generate a specific question about the content of this document: {summary}",
                # Person-related questions
                f"Generate a question about what {{person}} is working on or responsible for. Person ID: {{person}}",
                # Relationship questions
                "Generate a question about the relationship between {person1} and {person2}",
                # Project-related questions
                "Generate a question about Project Phoenix based on this document: {summary}",
                # Timeline/schedule questions
                "Generate a question about timelines, deadlines, or schedules mentioned in this document: {summary}"
            ]

            # Generate questions
            pairs_per_type = num_pairs // len(question_types) + 1

            for question_type in question_types:
                for _ in range(pairs_per_type):
                    if len(qa_pairs) >= num_pairs:
                        break

                    try:
                        if "{person}" in question_type:
                            # Person-related question
                            person = random.choice(person_ids)
                            prompt = question_type.format(person=person)
                        elif "{person1}" in question_type and "{person2}" in question_type:
                            # Relationship question
                            if len(person_ids) < 2:
                                continue
                            person1, person2 = random.sample(person_ids, 2)
                            prompt = question_type.format(person1=person1, person2=person2)
                        else:
                            # Document-related question
                            document = random.choice(documents)
                            prompt = question_type.format(summary=document["summary"])

                        # Generate question
                        question = llm.invoke(prompt).content.strip()

                        # Clean up the question
                        if question.startswith('"') and question.endswith('"'):
                            question = question[1:-1]

                        # Query the graph to get the answer
                        model = get_embedding_model()
                        query_embedding = model.encode(question)

                        vector_results = session.run(
                            """
                            MATCH (c:Chunk)-[:PART_OF]->(d:Document)
                            WHERE c.embedding IS NOT NULL
                            WITH c, d, c.embedding AS chunk_embedding, $query_embedding AS query_embedding
                            WITH c, d, gds.similarity.cosine(chunk_embedding, query_embedding) AS similarity
                            ORDER BY similarity DESC
                            LIMIT 3
                            RETURN c.summary AS chunk_summary, d.subject AS subject, similarity
                            """,
                            query_embedding=query_embedding.tolist()
                        ).data()

                        if not vector_results:
                            continue

                        # Extract context
                        context = "\n\n".join([f"{r['chunk_summary']} (From: {r['subject']})" for r in vector_results])

                        # Generate answer
                        answer_prompt = f"""
                        Answer the following question based on the provided context:

                        Question: {question}

                        Context:
                        {context}

                        If the context doesn't contain enough information to answer the question fully,
                        say so and provide what information you can based on the context.

                        Answer:
                        """

                        answer = llm.invoke(answer_prompt).content.strip()

                        # Add to QA pairs
                        qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "context": context
                        })

                        logger.info(f"Generated QA pair: {question}")

                        # Avoid rate limiting
                        time.sleep(1)

                    except Exception as e:
                        logger.error(f"Error generating QA pair: {str(e)}")
                        continue

            # Ensure we have exactly the requested number of pairs
            qa_pairs = qa_pairs[:num_pairs]

            return qa_pairs

    except Exception as e:
        logger.error(f"Error generating QA pairs: {str(e)}")
        return []
    finally:
        driver.close()

# Function to save QA pairs to a file
def save_qa_pairs(qa_pairs: List[Dict[str, str]], output_file: str = "qa_pairs.json"):
    """Save question-answer pairs to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2)
        logger.info(f"Successfully saved {len(qa_pairs)} QA pairs to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving QA pairs to {output_file}: {str(e)}")
        return False

# Main function
def main():
    """Main function to process messages and generate QA pairs."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process messages and generate QA pairs")
    parser.add_argument("--directory", type=str, default="uploads", help="Directory containing message files")
    parser.add_argument("--num-pairs", type=int, default=30, help="Number of QA pairs to generate")
    parser.add_argument("--output", type=str, default="qa_pairs.json", help="Output file for QA pairs")
    parser.add_argument("--skip-processing", action="store_true", help="Skip message processing")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA pair generation")
    args = parser.parse_args()

    # Process message files
    if not args.skip_processing:
        logger.info(f"Processing message files from {args.directory}...")
        processed_data = process_message_files(args.directory)
        logger.info(f"Processed {len(processed_data)} message files.")
    else:
        logger.info("Skipping message processing.")

    # Generate QA pairs
    if not args.skip_qa:
        logger.info(f"Generating {args.num_pairs} question-answer pairs...")
        qa_pairs = generate_qa_pairs(num_pairs=args.num_pairs)
        logger.info(f"Generated {len(qa_pairs)} question-answer pairs.")

        # Save QA pairs
        save_qa_pairs(qa_pairs, args.output)
    else:
        logger.info("Skipping QA pair generation.")

if __name__ == "__main__":
    main()
