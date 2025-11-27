"""
Test Script for AI Code Remediation Service
Tests the /local_fix endpoint with various vulnerability examples
"""

import requests
import json
import time
from typing import Dict, List
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# API endpoint
BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/local_fix"

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.CYAN}{text:^80}")
    print(f"{Fore.CYAN}{'=' * 80}\n")

def print_section(title: str, content: str):
    """Print a formatted section"""
    print(f"{Fore.YELLOW}{title}:")
    print(f"{Fore.WHITE}{content}\n")

def test_vulnerability(test_name: str, language: str, cwe: str, code: str):
    """
    Test a single vulnerability with the API
    
    Args:
        test_name: Name/description of the test
        language: Programming language
        cwe: CWE identifier
        code: Vulnerable code snippet
    """
    print_header(f"TEST: {test_name}")
    
    # Prepare request
    payload = {
        "language": language,
        "cwe": cwe,
        "code": code
    }
    
    print_section("Input", f"Language: {language}\nCWE: {cwe}")
    print_section("Vulnerable Code", code)
    
    # Send request with longer timeout for CPU inference
    try:
        start_time = time.time()
        response = requests.post(ENDPOINT, json=payload, timeout=300)  # 5 minutes for slow CPU
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"{Fore.GREEN}✓ Request successful!")
            print(f"{Fore.GREEN}Response time: {elapsed_time:.2f}s\n")
            
            # Display results
            print_section("Fixed Code", result['fixed_code'])
            print_section("Explanation", result['explanation'])
            print_section("Diff", result['diff'] if result['diff'] else "No diff available")
            
            # Display metrics
            print(f"{Fore.MAGENTA}Metrics:")
            print(f"  Model: {result['model_used']}")
            print(f"  Input Tokens: {result['token_usage']['input_tokens']}")
            print(f"  Output Tokens: {result['token_usage']['output_tokens']}")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
            print(f"  RAG Context Used: {result.get('rag_context_used', False)}")
            
            return True
            
        else:
            print(f"{Fore.RED}✗ Request failed!")
            print(f"{Fore.RED}Status code: {response.status_code}")
            print(f"{Fore.RED}Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"{Fore.RED}✗ Request timed out!")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error: {str(e)}")
        return False

def check_service_health():
    """Check if the service is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"{Fore.GREEN}✓ Service is healthy")
            print(f"  Model loaded: {health['model_loaded']}")
            print(f"  RAG available: {health['rag_available']}")
            return True
        else:
            print(f"{Fore.RED}✗ Service health check failed")
            return False
    except Exception as e:
        print(f"{Fore.RED}✗ Cannot connect to service: {str(e)}")
        print(f"{Fore.YELLOW}Make sure the service is running on {BASE_URL}")
        return False

def main():
    """Run all tests"""
    print_header("AI CODE REMEDIATION SERVICE - TEST SUITE")
    
    # Check service health
    print(f"{Fore.CYAN}Checking service health...")
    if not check_service_health():
        print(f"\n{Fore.RED}Please start the service first: python app.py")
        return
    
    print(f"\n{Fore.CYAN}Starting tests...\n")
    time.sleep(1)
    
    # Test cases
    test_cases = [
        {
            "name": "SQL Injection in Java",
            "language": "java",
            "cwe": "CWE-89",
            "code": """
public List<User> getUsersByName(String username) {
    String query = "SELECT * FROM users WHERE username = '" + username + "'";
    return jdbcTemplate.query(query, new UserRowMapper());
}
"""
        },
        {
            "name": "Hardcoded Credentials in Python",
            "language": "python",
            "cwe": "CWE-798",
            "code": """
import psycopg2

def connect_to_database():
    connection = psycopg2.connect(
        host="localhost",
        database="myapp",
        user="admin",
        password="SuperSecret123!"
    )
    return connection
"""
        },
        {
            "name": "Cross-Site Scripting (XSS) in JavaScript",
            "language": "javascript",
            "cwe": "CWE-79",
            "code": """
function displayUserComment(comment) {
    const commentDiv = document.getElementById('comments');
    commentDiv.innerHTML += '<p>' + comment + '</p>';
}

// User input directly inserted into HTML
const userComment = getUserInput();
displayUserComment(userComment);
"""
        },
        {
            "name": "Path Traversal in Python",
            "language": "python",
            "cwe": "CWE-22",
            "code": """
from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/download')
def download_file():
    filename = request.args.get('file')
    return send_file(f'/var/www/uploads/{filename}')
"""
        },
        {
            "name": "Command Injection in Java",
            "language": "java",
            "cwe": "CWE-78",
            "code": """
public String pingHost(String host) throws IOException {
    String command = "ping -c 4 " + host;
    Process process = Runtime.getRuntime().exec(command);
    
    BufferedReader reader = new BufferedReader(
        new InputStreamReader(process.getInputStream())
    );
    
    return reader.lines().collect(Collectors.joining("\\n"));
}
"""
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{Fore.BLUE}[Test {i}/{len(test_cases)}]")
        success = test_vulnerability(
            test_case["name"],
            test_case["language"],
            test_case["cwe"],
            test_case["code"]
        )
        results.append(success)
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print_header("TEST SUMMARY")
    total = len(results)
    passed = sum(results)
    failed = total - passed
    
    print(f"{Fore.CYAN}Total Tests: {total}")
    print(f"{Fore.GREEN}Passed: {passed}")
    print(f"{Fore.RED}Failed: {failed}")
    
    if passed == total:
        print(f"\n{Fore.GREEN}{'★' * 80}")
        print(f"{Fore.GREEN}{'ALL TESTS PASSED!':^80}")
        print(f"{Fore.GREEN}{'★' * 80}\n")
    else:
        print(f"\n{Fore.YELLOW}Some tests failed. Please review the output above.\n")

if __name__ == "__main__":
    main()