from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import re
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Any, Optional, Tuple
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from section_prompts import get_section_prompts
from dotenv import load_dotenv

load_dotenv()

# Simple configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-r1:7b')
DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_CLOUD_MODEL = os.getenv('DEEPSEEK_CLOUD_MODEL', 'deepseek-chat')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
GEMINI_API_URL = os.getenv('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta')
PROVIDER = os.getenv('PROVIDER', 'ollama').lower()  # 'ollama', 'deepseek', or 'gemini'
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
TOP_P = float(os.getenv('TOP_P', 0.9))

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = SECRET_KEY

# Simple in-memory storage for tests with thread safety
tests_storage = {}
storage_lock = threading.RLock()

# Thread pool for asynchronous test generation
executor = ThreadPoolExecutor(max_workers=3)

# Background cleanup thread
cleanup_thread = None

# Configuration for cleanup
MAX_TEST_AGE_HOURS = 24  # Remove tests older than 24 hours
MAX_STORED_TESTS = 100   # Maximum number of tests to keep in memory

class IQTestGenerator:
    def __init__(self, provider: str = None, ollama_url: str = None, model: str = None):
        self.provider = (provider or PROVIDER).lower()
        self.ollama_url = ollama_url or OLLAMA_BASE_URL
        self.model = model or DEEPSEEK_MODEL
        self.session = requests.Session()
    
    def _generate_test_section_by_section(self, age: int, test_id: str, provider: Optional[str] = None) -> Dict[str, Any]:
        """Generate a complete IQ test section by section to avoid token limits"""
        current_provider = (provider or self.provider).lower()
        if current_provider not in ['ollama', 'deepseek', 'gemini']:
            return {
                "success": False,
                "error": "Provider must be 'ollama', 'deepseek', or 'gemini'"
            }
        
        if current_provider == 'deepseek' and not DEEPSEEK_API_KEY:
            return {
                "success": False,
                "error": "DEEPSEEK_API_KEY not configured. Please set it in your environment variables."
            }
        
        if current_provider == 'gemini' and not GEMINI_API_KEY:
            return {
                "success": False,
                "error": "GEMINI_API_KEY not configured. Please set it in your environment variables."
            }
        
        # Validate age parameter
        if not (10 <= age <= 18):
            return {
                "success": False,
                "error": "Age must be between 10 and 18"
            }
        
        # Get section prompts
        section_prompts = get_section_prompts(age)
        
        # Define section order (3 sections only)
        section_order = [
            "section_1", "section_2", "section_3"
        ]
        
        # Initialize test content
        test_content = ""
        
        # Generate each section
        for i, section_name in enumerate(section_order):
            # Get section prompt
            section_prompt = section_prompts[section_name]
            
            # Generate section content
            try:
                if current_provider == 'deepseek':
                    section_content = self._call_deepseek_cloud(section_prompt)
                elif current_provider == 'gemini':
                    section_content = self._call_gemini(section_prompt)
                else:
                    section_content = self._call_ollama(section_prompt)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to generate section {section_name}: {str(e)}",
                    "provider": current_provider
                }
            
            if not section_content:
                return {
                    "success": False,
                    "error": f"Failed to generate section: {section_name}",
                    "provider": current_provider
                }

            cleaned = re.sub(r'<think>.*?</think>', '', section_content, flags=re.DOTALL)
            
            # Append to test content
            test_content += cleaned + "\n\n"
            
            # Update progress in test record with thread safety
            with storage_lock:
                if test_id in tests_storage:
                    tests_storage[test_id].update({
                        "status": "generating",
                        "progress": f"{i+1}/{len(section_order)}",
                        "current_section": section_name,
                        "provider": current_provider
                    })
        
        # Save complete test to file
        is_valid, validation_errors = self._validate_test_schema(test_content)
        if not is_valid:
            error_message = "Schema validation failed"
            if validation_errors:
                error_message += f": {'; '.join(validation_errors)}"
            with storage_lock:
                if test_id in tests_storage:
                    tests_storage[test_id].update({
                        "status": "failed",
                        "progress": "validation_failed",
                        "current_section": "validation",
                        "error": error_message,
                        "provider": current_provider
                    })
            return {
                "success": False,
                "error": error_message,
                "provider": current_provider
            }

        try:
            filepath = self._save_test_to_file(test_content, age, test_id)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save test file: {str(e)}"
            }
        
        # Update test record with completion and cleanup old tests
        with storage_lock:
            tests_storage[test_id].update({
                "status": "completed",
                "progress": "completed",
                "current_section": "completed",
                "file_path": filepath,
                "completed_at": datetime.now().isoformat(),
                "provider": current_provider
            })
            
            # Cleanup old tests (excluding current test)
            self._cleanup_old_tests(exclude_test_id=test_id)
        
        return {
            "success": True,
            "test_id": test_id,
            "message": "Test generated successfully",
            "file_path": filepath,
            "provider": current_provider
        }
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with the given prompt"""
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            }
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            
            if 'response' in result and result['response']:
                return result['response']
            else:
                raise Exception("Empty or invalid response from Ollama API")
                
        except requests.exceptions.Timeout:
            raise Exception("Ollama API request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama API")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API request failed: {str(e)}")
        except ValueError as e:
            raise Exception(f"Invalid JSON response from Ollama API: {str(e)}")

    def _call_deepseek_cloud(self, prompt: str) -> str:
        """Call DeepSeek Cloud API with the given prompt"""
        if not DEEPSEEK_API_KEY:
            raise Exception("DEEPSEEK_API_KEY not configured. Please set it in your environment variables.")
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": DEEPSEEK_CLOUD_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "top_p": TOP_P
        }
        
        try:
            response = self.session.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=300)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise Exception("DeepSeek Cloud API request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to DeepSeek Cloud API")
        except requests.exceptions.RequestException as e:
            raise Exception(f"DeepSeek Cloud API request failed: {str(e)}")
        
        try:
            result = response.json()
        except ValueError as e:
            raise Exception(f"Invalid JSON response from DeepSeek Cloud API: {str(e)}")
        
        choices = result.get('choices', [])
        if not choices:
            raise Exception("Invalid response format from DeepSeek Cloud API")
        
        message = choices[0].get('message', {})
        content = message.get('content')
        if not content:
            raise Exception("Empty content returned from DeepSeek Cloud API")
        
        return content

    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini Cloud API with the given prompt"""
        if not GEMINI_API_KEY:
            raise Exception("GEMINI_API_KEY not configured. Please set it in your environment variables.")
        
        url = f"{GEMINI_API_URL}/models/{GEMINI_MODEL}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "topP": TOP_P
            }
        }
        
        try:
            response = self.session.post(url, headers=headers, json=data, timeout=300)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise Exception("Gemini API request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Gemini API")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Gemini API request failed: {str(e)}")
        
        try:
            result = response.json()
        except ValueError as e:
            raise Exception(f"Invalid JSON response from Gemini API: {str(e)}")
        
        candidates = result.get('candidates', [])
        if not candidates:
            raise Exception("No candidates returned from Gemini API")
        
        content = candidates[0].get('content', {})
        parts = content.get('parts', [])
        if not parts:
            raise Exception("Empty content returned from Gemini API")
        
        text = parts[0].get('text', '')
        if not text:
            raise Exception("Empty text returned from Gemini API")
        
        return text

    def _validate_test_schema(self, content: str) -> Tuple[bool, List[str]]:
        """Validate that generated IQ test content follows the expected schema."""
        errors: List[str] = []
        
        if not content or not content.strip():
            return False, ["Test content is empty"]

        expected_sections = [
            ("1", "Verbal Reasoning"),
            ("2", "Mathematical Reasoning"),
            ("3", "Spatial/Visual Reasoning")
        ]

        section_pattern = re.compile(r"Section\s+(\d+):\s+([^\n]+)")
        matches = list(section_pattern.finditer(content))

        if len(matches) != len(expected_sections):
            errors.append(f"Expected {len(expected_sections)} sections but found {len(matches)}")

        sections: List[Tuple[re.Match, str]] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            sections.append((match, content[start:end]))

        for idx, expected in enumerate(expected_sections):
            if idx >= len(sections):
                errors.append(f"Missing section {expected[0]}: {expected[1]}")
                continue

            match, section_text = sections[idx]
            section_number, section_title = expected

            found_number = match.group(1)
            found_header = match.group(0).strip()
            found_title = match.group(2).strip()

            if found_number != section_number:
                errors.append(f"Expected Section {section_number} but found Section {found_number}")

            if expected[1].lower() not in found_title.lower():
                errors.append(f"Section {section_number} header should include '{expected[1]}' but found '{found_title}'")

            header_end = match.end()
            section_body = content[header_end: sections[idx + 1][0].start()] if idx + 1 < len(sections) else content[header_end:]
            section_body = section_body.strip()

            answer_key_match = re.search(r"Answer\s*Key:\s*(.*)", section_body, re.DOTALL | re.IGNORECASE)
            if not answer_key_match:
                errors.append(f"Section {section_number} missing 'Answer Key'")
                continue

            questions_text = section_body[:answer_key_match.start()].strip()
            answer_text = answer_key_match.group(1).strip()

            question_pattern = re.compile(r"(\d+)\.\s*Question:\s*(.*?)(?=\n\d+\.\s*Question:|\nAnswer\s*Key:|\Z)", re.DOTALL | re.IGNORECASE)
            question_matches = list(question_pattern.finditer(questions_text))

            if len(question_matches) != 10:
                errors.append(f"Section {section_number} should contain 10 questions but found {len(question_matches)}")

            for q_idx, question_match in enumerate(question_matches, start=1):
                question_number = question_match.group(1).strip()
                if question_number != str(q_idx):
                    errors.append(f"Section {section_number} question numbering issue: expected {q_idx} but found {question_number}")

                question_block = question_match.group(2)
                options = re.findall(r"^[A-D]\)\s+.+", question_block, flags=re.MULTILINE)
                if len(options) != 4:
                    errors.append(f"Section {section_number} Question {question_number} should have 4 options but found {len(options)}")

            answer_lines = re.findall(r"^(\d+)\.\s*([A-D])\b", answer_text, flags=re.MULTILINE)
            if len(answer_lines) != 10:
                errors.append(f"Section {section_number} Answer Key should list 10 answers but found {len(answer_lines)}")
            else:
                for expected_num in range(1, 11):
                    if str(expected_num) not in [num for num, _ in answer_lines]:
                        errors.append(f"Section {section_number} Answer Key missing entry for question {expected_num}")
                        break

        return (len(errors) == 0, errors)
    
    def _save_test_to_file(self, content: str, age: int, test_id: str) -> str:
        """Save test content to file with security validation"""
        # Validate test_id to prevent path traversal
        if not re.match(r'^[a-zA-Z0-9\-_]+', test_id):
            raise ValueError("Invalid test_id format")
        
        # Validate age parameter
        if not isinstance(age, int) or not (10 <= age <= 18):
            raise ValueError("Invalid age parameter")
        
        # Create age-specific directory
        age_dir = os.path.join('tests', str(age))
        os.makedirs(age_dir, exist_ok=True)
        
        # Generate filename with current date and test_id for uniqueness
        current_date = datetime.now().strftime('%d_%m_%Y')
        filename = f"{current_date}_{test_id[:8]}.txt"
        filepath = os.path.join(age_dir, filename)
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def _cleanup_old_tests(self, exclude_test_id: str = None):
        """Remove old test records from memory"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=MAX_TEST_AGE_HOURS)
        
        # Remove old test records (excluding specified test)
        old_test_ids = []
        for test_id, test_record in tests_storage.items():
            if test_id == exclude_test_id:
                continue  # Skip the excluded test
            created_at = datetime.fromisoformat(test_record['created_at'])
            if created_at < cutoff_time:
                old_test_ids.append(test_id)
        
        for test_id in old_test_ids:
            # Also remove the associated file if it exists
            test_record = tests_storage.get(test_id)
            if test_record and 'file_path' in test_record:
                try:
                    if os.path.exists(test_record['file_path']):
                        os.remove(test_record['file_path'])
                except OSError:
                    pass  # Ignore file removal errors
            del tests_storage[test_id]
        
        # If we still have too many tests, remove the oldest ones (excluding specified test)
        if len(tests_storage) > MAX_STORED_TESTS:
            sorted_tests = sorted(
                [(tid, trecord) for tid, trecord in tests_storage.items() if tid != exclude_test_id], 
                key=lambda x: datetime.fromisoformat(x[1]['created_at'])
            )
            excess_count = len(tests_storage) - MAX_STORED_TESTS
            for i in range(excess_count):
                test_id = sorted_tests[i][0]
                test_record = tests_storage.get(test_id)
                if test_record and 'file_path' in test_record:
                    try:
                        if os.path.exists(test_record['file_path']):
                            os.remove(test_record['file_path'])
                    except OSError:
                        pass
                del tests_storage[test_id]


# Initialize generator
test_generator = IQTestGenerator()

def periodic_cleanup():
    """Periodic cleanup function that runs every hour"""
    import time
    while True:
        time.sleep(3600)  # Sleep for 1 hour
        try:
            test_generator._cleanup_old_tests()
        except Exception:
            pass  # Ignore cleanup errors in background thread

def start_cleanup_thread():
    """Start the background cleanup thread"""
    global cleanup_thread
    if cleanup_thread is None or not cleanup_thread.is_alive():
        cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
        cleanup_thread.start()

def create_error_response(message: str, status_code: int = 400) -> tuple:
    """Create standardized error response"""
    return jsonify({
        "success": False,
        "error": message
    }), status_code

def create_success_response(data: dict = None, message: str = None) -> dict:
    """Create standardized success response"""
    response = {"success": True}
    if data:
        response.update(data)
    if message:
        response["message"] = message
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/generate', methods=['POST'])
def generate_test():
    """Generate a new IQ test"""
    try:
        data = request.get_json()
        age = data.get('age', 14)
        provider = (data.get('provider', PROVIDER) if data else PROVIDER).lower()
        
        # Validate age
        if not (10 <= age <= 18):
            return create_error_response("Age must be between 10 and 18", 400)
        
        if provider not in ['ollama', 'deepseek', 'gemini']:
            return create_error_response("Provider must be 'ollama', 'deepseek', or 'gemini'", 400)
        
        if provider == 'deepseek' and not DEEPSEEK_API_KEY:
            return create_error_response("DEEPSEEK_API_KEY not configured. Please set it in your environment variables.", 400)
        
        if provider == 'gemini' and not GEMINI_API_KEY:
            return create_error_response("GEMINI_API_KEY not configured. Please set it in your environment variables.", 400)
        
        # Generate unique test ID
        test_id = str(uuid.uuid4())
        
        # Create test record with thread safety
        with storage_lock:
            tests_storage[test_id] = {
                "test_id": test_id,
                "age": age,
                "status": "generating",
                "progress": "0/3",
                "current_section": "initializing",
                "provider": provider,
                "created_at": datetime.now().isoformat()
            }
        
        # Submit generation task to thread pool
        future = executor.submit(
            test_generator._generate_test_section_by_section,
            age, test_id, provider
        )
        
        # Add callback to handle task completion/failure
        def task_done_callback(future_result):
            try:
                result = future_result.result()
                if not result.get("success", False):
                    # Update status to failed if generation failed
                    with storage_lock:
                        if test_id in tests_storage:
                            tests_storage[test_id].update({
                                "status": "failed",
                                "progress": "failed",
                                "current_section": "failed",
                                "error": result.get("error", "Unknown error"),
                                "provider": result.get("provider", provider)
                            })
            except Exception as e:
                # Update status to failed if task raised an exception
                with storage_lock:
                    if test_id in tests_storage:
                        tests_storage[test_id].update({
                            "status": "failed",
                            "progress": "failed",
                            "current_section": "failed",
                            "error": str(e),
                            "provider": provider
                        })
        
        future.add_done_callback(task_done_callback)
        
        return jsonify(create_success_response({
            "test_id": test_id,
            "status": "generating",
            "provider": provider
        }, "Test generation started"))
        
    except Exception as e:
        return create_error_response(str(e), 500)

@app.route('/status/<test_id>', methods=['GET'])
def get_test_status(test_id: str):
    """Get the status of a test generation"""
    # Validate test_id format
    if not test_id or not re.match(r'^[a-zA-Z0-9\-_]+', test_id):
        return create_error_response("Invalid test_id format", 400)
    
    with storage_lock:
        if test_id not in tests_storage:
            return jsonify({
                "success": False,
                "error": "Test not found"
            }), 404
        
        test_record = tests_storage[test_id].copy()
    
    return jsonify({
        "success": True,
        "test_id": test_id,
        "status": test_record["status"],
        "progress": test_record.get("progress", "unknown"),
        "current_section": test_record.get("current_section", "unknown"),
        "created_at": test_record["created_at"],
        "completed_at": test_record.get("completed_at"),
        "file_path": test_record.get("file_path"),
        "provider": test_record.get("provider"),
        "error": test_record.get("error")  # Include error if present
    })


@app.route('/tests', methods=['GET'])
def list_tests():
    """List all generated tests"""
    with storage_lock:
        return jsonify({
            "success": True,
            "tests": list(tests_storage.values())
        })


if __name__ == '__main__':
    # Start background cleanup thread
    start_cleanup_thread()
    app.run(host=HOST, port=PORT, debug=DEBUG)
