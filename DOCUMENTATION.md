# IQ Test Generator

## Overview

A REST API that generates age-appropriate IQ tests for adolescents (10-18 years) using AI. Built with Flask and powered by DeepSeek-R1 via Ollama.

### Key Features
- AI-powered test generation with section-specific prompts
- Asynchronous processing with thread pool execution
- Age-based file storage with automatic cleanup
- Real-time progress tracking
- Comprehensive scoring with IQ conversion

### Test Structure
Each test contains 3 cognitive assessment sections:

1. **Verbal Reasoning** (10-12 questions, 7-8 minutes)
   - Vocabulary, analogies, sentence completion, reading comprehension
2. **Mathematical Reasoning** (10-12 questions, 10 minutes)  
   - Number sequences, word problems, proportions, logical deduction
3. **Spatial/Visual Reasoning** (10-12 questions, 7-8 minutes)
   - Pattern completion, mental rotation, spatial folding, matrix reasoning

**Total**: 30-36 questions, ~25 minutes

## API Documentation

**Base URL**: `http://localhost:5000`

### Endpoints

#### Health Check
```http
GET /health
```
Returns server status and version.

#### Generate Test
```http
POST /generate
Content-Type: application/json

{
  "age": 14
}
```
Generates an IQ test for the specified age (10-18). Returns immediately with a test ID.

#### Check Status
```http
GET /status/{test_id}
```
Returns the current status of test generation. Progress shows "X/3" sections completed.

#### Score Test
```http
POST /score
Content-Type: application/json

{
  "test_id": "test-id-here",
  "answers": {
    "1": "A",
    "2": "B",
    "3": "C"
  }
}
```
Scores a completed test and returns IQ score with classification.

#### List Tests
```http
GET /tests
```
Returns all generated tests with their status.

#### List Scores
```http
GET /scores
```
Returns all test scores.

## Installation

### Prerequisites
- Python 3.8+
- Ollama installed and running
- DeepSeek-R1:7b model downloaded

### Setup
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Start Ollama: `ollama serve`
6. Pull model: `ollama pull deepseek-r1:7b`
7. Run application: `python app.py`

### Docker

1. Copy `.env.example` to `.env` and fill in your values (or create a `.env` with the variables below):
   ```
   PROVIDER=ollama            # or 'deepseek'
   OLLAMA_BASE_URL=http://host.docker.internal:11434
   DEEPSEEK_API_KEY=          # required when PROVIDER=deepseek
   ```
2. Build and start: `docker compose up -d`
3. Verify: `curl http://localhost:5000/health`
4. Stop: `docker compose down`

Generated test files are persisted to `./tests/` via a volume mount.

### Dependencies
- Flask 2.3.3
- Flask-CORS 4.0.0
- requests 2.31.0
- python-dotenv 1.0.1

## License

MIT License - see LICENSE file for details.
