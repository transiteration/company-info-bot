# Company Information Assistant Bot

This project is a company information assistant powered by [LangChain](https://python.langchain.com/), [Qdrant](https://qdrant.tech/), and [Telegram](https://core.telegram.org/bots). It allows employees to ask questions about company policies, projects, or history via Telegram, and get answers based on internal documents.

## Features

- **Document Ingestion:** Loads and indexes company documents into a Qdrant vector database.
- **RAG Pipeline:** Uses Retrieval-Augmented Generation (RAG) with Google Gemini for accurate answers.
- **Telegram Bot:** Employees interact with the assistant via Telegram.
- **Document Grading:** Filters out irrelevant documents before generating answers.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/transiteration/company-info-bot
cd company-info-bot
```

### 2. Install Dependencies

If running locally:

```bash
pip install -r requirements.txt
```

### 3. Prepare Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY="your-google-api-key"
TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
QDRANT_URL="http://localhost:6333"
```

### 4. Prepare Documents

Place your documents and files in the `documents/` folder. These will be indexed and used for answering questions.

### 5. Ingest Documents

If running locally:

```bash
python ingest.py
```

### 6. Start the Telegram Bot

If running locally:

```bash
python bot.py
```

The bot will start and listen for messages on Telegram.

---

## Running with Docker Compose

You can run both Qdrant and the bot using Docker Compose.

```bash
docker-compose up --build
```

- Place your `.env` file and `documents/` folder in the project root before starting.
- The ingestion script runs automatically before the bot starts.

---

## Usage

- Start a chat with your bot on Telegram.
- Ask questions like:
  - "What is our remote work policy?"
  - "Tell me about Project Alpha."
  - "Who founded our company?"

The bot will respond with concise, document-based answers.

## Project Structure

```
.
├── bot.py          # Telegram bot interface
├── graph.py        # RAG workflow and logic
├── ingest.py       # Document ingestion and indexing
├── documents/      # Folder for company documents (.txt)
├── .env            # Environment variables
└── README.md       # This file
```

## Notes

- The Google API key must have access to Gemini models for embeddings and chat.
- Only `.txt` files in the `documents/` folder will be ingested.
- This project can be improved and implemented with Microsoft Teams or Slack in the future.
- You can also add support for other document types beyond `.txt`.
