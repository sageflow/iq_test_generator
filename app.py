from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import re
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Any, Optional
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from section_prompts import get_section_prompts

# Simple configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-r1:7b')
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
TOP_P = float(os.getenv('TOP_P', 0.9))

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = SECRET_KEY

# Simple in-memory storage for tests and scores with thread safety
tests_storage = {}
scores_storage = []
storage_lock = threading.RLock()

# Thread pool for asynchronous test generation
executor = ThreadPoolExecutor(max_workers=3)

# Configuration for cleanup
MAX_TEST_AGE_HOURS = 24  # Remove tests older than 24 hours
MAX_STORED_TESTS = 100   # Maximum number of tests to keep in memory

class IQTestGenerator:
    def __init__(self, ollama_url: str = None, model: str = None):
        self.ollama_url = ollama_url or OLLAMA_BASE_URL
        self.model = model or DEEPSEEK_MODEL
        self.session = requests.Session()
    
    def _generate_test_section_by_section(self, age: int, test_id: str) -> Dict[str, Any]:
        """Generate a complete IQ test section by section to avoid token limits"""
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
            section_content = self._call_ollama(section_prompt)
            
            if not section_content:
                return {
                    "success": False,
                    "error": f"Failed to generate section: {section_name}"
                }
            
            # Append to test content
            test_content += section_content + "\n\n"
            
            # Update progress in test record with thread safety
            with storage_lock:
                if test_id in tests_storage:
                    tests_storage[test_id].update({
                        "status": "generating",
                        "progress": f"Completed section {i+1}/{len(section_order)}: {section_name}",
                        "current_section": section_name
                    })
        
        # Save complete test to file
        filepath = self._save_test_to_file(test_content, age, test_id)
        
        # Update test record with completion and cleanup old tests
        with storage_lock:
            tests_storage[test_id].update({
                "status": "completed",
                "progress": "completed",
                "current_section": "completed",
                "file_path": filepath,
                "completed_at": datetime.now().isoformat()
            })
            
            # Cleanup old tests
            self._cleanup_old_tests()
        
        return {
            "success": True,
            "test_id": test_id,
            "message": "Test generated successfully",
            "file_path": filepath
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
        
        response = self.session.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        
        if 'response' in result:
            return result['response']
        else:
            raise Exception("Invalid response format from Ollama API")
    
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
    
    def _cleanup_old_tests(self):
        """Remove old test records from memory"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=MAX_TEST_AGE_HOURS)
        
        # Remove old test records
        old_test_ids = []
        for test_id, test_record in tests_storage.items():
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
        
        # If we still have too many tests, remove the oldest ones
        if len(tests_storage) > MAX_STORED_TESTS:
            sorted_tests = sorted(
                tests_storage.items(), 
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


class IQTestScorer:
    def __init__(self):
        pass
    
    def score_test(self, answers: Dict[str, str], test_content: str) -> Dict[str, Any]:
        """Score a completed IQ test"""
        # Extract answer key from test content
        answer_key = self._extract_answer_key(test_content)
        
        if not answer_key:
            return {
                "success": False,
                "error": "Could not extract answer key from test"
            }
        
        # Calculate score
        correct_answers = 0
        total_questions = len(answer_key)
        
        for question_num, correct_answer in answer_key.items():
            user_answer = answers.get(question_num)
            if user_answer and user_answer.upper() == correct_answer.upper():
                correct_answers += 1
        
        # Calculate percentage and IQ score
        percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        # Proper IQ score calculation using psychometric scaling
        # Assuming a normal distribution with mean=50% and SD=15% of total questions
        raw_score = correct_answers
        mean_score = total_questions * 0.5  # 50% as baseline
        std_dev = total_questions * 0.15    # 15% standard deviation
        
        # Calculate z-score and convert to IQ scale (mean=100, SD=15)
        if std_dev > 0:
            z_score = (raw_score - mean_score) / std_dev
            iq_score = 100 + (z_score * 15)
        else:
            iq_score = 100  # Default to average if no variance
        
        # Clamp IQ score to reasonable range (60-160)
        iq_score = max(60, min(160, iq_score))
        
        # Determine classification
        if iq_score >= 130:
            classification = "Very Superior"
        elif iq_score >= 120:
            classification = "Superior"
        elif iq_score >= 110:
            classification = "High Average"
        elif iq_score >= 90:
            classification = "Average"
        elif iq_score >= 80:
            classification = "Low Average"
        else:
            classification = "Below Average"
        
        return {
            "success": True,
            "score": {
                "correct_answers": correct_answers,
                "total_questions": total_questions,
                "percentage": round(percentage, 2),
                "iq_score": round(iq_score, 1),
                "classification": classification
            }
        }
    
    def _extract_answer_key(self, test_content: str) -> Dict[str, str]:
        """Extract answer key from test content with improved regex"""
        answer_key = {}
        
        # More robust patterns for answer key extraction
        patterns = [
            r'Answer Key:\s*([^\n]+)',
            r'Answers:\s*([^\n]+)',
            r'Key:\s*([^\n]+)',
            r'Answer[\s]*Key[\s]*:?\s*([^\n]+)',
            r'Correct[\s]*Answers?:\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, test_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Parse individual answers with more flexible patterns
                # Handle formats like: 1:A, 2:B, 3:C or 1. A, 2. B, 3. C or Q1:A, Q2:B
                answer_patterns = [
                    r'(?:Q|Question)?\s*(\d+)\s*[:.]?\s*([A-D])',
                    r'(\d+)\s*[:.]?\s*([A-D])',
                    r'(\d+)\s*=\s*([A-D])'
                ]
                
                for answer_pattern in answer_patterns:
                    answers = re.findall(answer_pattern, match, re.IGNORECASE)
                    for question_num, answer in answers:
                        answer_key[question_num.strip()] = answer.upper()
        
        return answer_key

# Initialize generators
test_generator = IQTestGenerator()
test_scorer = IQTestScorer()

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
        
        # Validate age
        if not (10 <= age <= 18):
            return create_error_response("Age must be between 10 and 18", 400)
        
        # Generate unique test ID
        test_id = str(uuid.uuid4())
        
        # Create test record with thread safety
        with storage_lock:
            tests_storage[test_id] = {
                "test_id": test_id,
                "age": age,
                "status": "generating",
                "progress": "0/6",
                "current_section": "initializing",
                "created_at": datetime.now().isoformat()
            }
        
        # Submit generation task to thread pool
        future = executor.submit(
            test_generator._generate_test_section_by_section,
            age, test_id
        )
        
        return jsonify(create_success_response({
            "test_id": test_id,
            "status": "generating"
        }, "Test generation started"))
        
    except Exception as e:
        return create_error_response(str(e), 500)

@app.route('/status/<test_id>', methods=['GET'])
def get_test_status(test_id: str):
    """Get the status of a test generation"""
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
        "file_path": test_record.get("file_path")
    })

@app.route('/score', methods=['POST'])
def score_test():
    """Score a completed IQ test"""
    try:
        data = request.get_json()
        test_id = data.get('test_id')
        answers = data.get('answers', {})
        
        if not test_id or not re.match(r'^[a-zA-Z0-9\-_]+', test_id):
            return create_error_response("Invalid test_id format", 400)
        
        with storage_lock:
            if test_id not in tests_storage:
                return create_error_response("Test not found", 404)
            
            test_record = tests_storage[test_id].copy()
        
            if test_record["status"] != "completed":
                return create_error_response("Test not completed yet", 400)
        
        # Read test content from file
        file_path = test_record.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return create_error_response("Test file not found", 404)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            test_content = f.read()
        
        # Score the test
        result = test_scorer.score_test(answers, test_content)
        
        if result["success"]:
            # Store score with thread safety
            score_record = {
                "test_id": test_id,
                "score": result["score"],
                "scored_at": datetime.now().isoformat()
            }
            with storage_lock:
                scores_storage.append(score_record)
                # Limit stored scores to prevent memory issues
                if len(scores_storage) > 1000:
                    scores_storage.pop(0)  # Remove oldest score
            
            return jsonify({
                "success": True,
                "test_id": test_id,
                "score": result["score"]
            })
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return create_error_response(str(e), 500)

@app.route('/tests', methods=['GET'])
def list_tests():
    """List all generated tests"""
    with storage_lock:
        return jsonify({
            "success": True,
            "tests": list(tests_storage.values())
        })

@app.route('/scores', methods=['GET'])
def list_scores():
    """List all test scores"""
    with storage_lock:
        return jsonify({
            "success": True,
            "scores": scores_storage.copy()
        })

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
