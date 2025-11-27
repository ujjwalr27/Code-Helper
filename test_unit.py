"""
Unit Tests for AI Code Remediation Service
Tests individual components: Model Handler, RAG Retriever, API endpoints
"""

import unittest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock heavy dependencies for faster testing
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

from fastapi.testclient import TestClient


class TestModelHandler(unittest.TestCase):
    """Test Model Handler functionality"""
    
    @patch('model_handler.AutoModelForCausalLM')
    @patch('model_handler.AutoTokenizer')
    def test_model_initialization(self, mock_tokenizer, mock_model):
        """Test model handler initializes correctly"""
        from model_handler import ModelHandler
        
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        handler = ModelHandler(model_name="test-model", use_gpu=False)
        
        self.assertEqual(handler.model_name, "test-model")
        self.assertEqual(handler.device, "cpu")
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
    
    @patch('model_handler.torch')
    def test_gpu_detection(self, mock_torch):
        """Test GPU detection logic"""
        from model_handler import ModelHandler
        
        mock_torch.cuda.is_available.return_value = True
        
        with patch('model_handler.AutoModelForCausalLM'), \
             patch('model_handler.AutoTokenizer'):
            handler = ModelHandler(use_gpu=True)
            self.assertTrue(handler.use_gpu)
    
    @patch('model_handler.AutoModelForCausalLM')
    @patch('model_handler.AutoTokenizer')
    def test_generate_fix(self, mock_tokenizer, mock_model):
        """Test fix generation returns expected structure"""
        from model_handler import ModelHandler
        
        # Setup mocks
        mock_tok_instance = Mock()
        mock_tok_instance.pad_token = None
        mock_tok_instance.eos_token = "<EOS>"
        mock_tok_instance.return_value = {
            'input_ids': Mock(shape=[1, 10])
        }
        mock_tok_instance.decode.return_value = "Fixed code here"
        
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance
        
        mock_model_instance = Mock()
        mock_model_instance.eval.return_value = None
        mock_model_instance.generate.return_value = Mock(shape=[1, 15])
        mock_model.from_pretrained.return_value = mock_model_instance
        
        with patch('model_handler.torch'):
            handler = ModelHandler()
            result = handler.generate_fix("test prompt")
        
        self.assertIn('text', result)
        self.assertIn('input_tokens', result)
        self.assertIn('output_tokens', result)


class TestRAGRetriever(unittest.TestCase):
    """Test RAG Retriever functionality"""
    
    def setUp(self):
        """Create temporary recipes directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.recipes_dir = os.path.join(self.temp_dir, 'recipes')
        os.makedirs(self.recipes_dir)
        
        # Create sample recipes
        with open(os.path.join(self.recipes_dir, 'sql_injection_cwe89.txt'), 'w') as f:
            f.write("SQL Injection remediation guide...")
        
        with open(os.path.join(self.recipes_dir, 'xss_cwe79.txt'), 'w') as f:
            f.write("XSS remediation guide...")
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    @patch('rag_retriever.SentenceTransformer')
    @patch('rag_retriever.faiss')
    def test_retriever_initialization(self, mock_faiss, mock_transformer):
        """Test RAG retriever loads recipes correctly"""
        from rag_retriever import RAGRetriever
        
        mock_transformer.return_value = Mock()
        mock_transformer.return_value.encode.return_value = [[0.1, 0.2]]
        
        retriever = RAGRetriever(recipes_dir=self.recipes_dir)
        
        self.assertEqual(len(retriever.recipes), 2)
        self.assertTrue(any('sql' in r['filename'].lower() for r in retriever.recipes))
    
    @patch('rag_retriever.SentenceTransformer')
    @patch('rag_retriever.faiss.IndexFlatL2')
    def test_cwe_extraction(self, mock_index, mock_transformer):
        """Test CWE extraction from filenames"""
        from rag_retriever import RAGRetriever
        
        mock_transformer.return_value = Mock()
        mock_transformer.return_value.encode.return_value = [[0.1, 0.2]]
        
        retriever = RAGRetriever(recipes_dir=self.recipes_dir)
        
        sql_recipe = [r for r in retriever.recipes if 'sql' in r['filename'].lower()][0]
        self.assertEqual(sql_recipe['cwe'], 'CWE-89')
    
    @patch('rag_retriever.SentenceTransformer')
    @patch('rag_retriever.faiss.IndexFlatL2')
    def test_retrieve_exact_match(self, mock_index, mock_transformer):
        """Test retrieval with exact CWE match"""
        from rag_retriever import RAGRetriever
        
        mock_transformer.return_value = Mock()
        mock_transformer.return_value.encode.return_value = [[0.1, 0.2]]
        
        retriever = RAGRetriever(recipes_dir=self.recipes_dir)
        
        result = retriever.retrieve("CWE-89", "java")
        self.assertIsNotNone(result)
        self.assertIn("SQL Injection", result)
    
    @patch('rag_retriever.SentenceTransformer')
    @patch('rag_retriever.faiss.IndexFlatL2')
    def test_get_stats(self, mock_index, mock_transformer):
        """Test statistics retrieval"""
        from rag_retriever import RAGRetriever
        
        mock_transformer.return_value = Mock()
        mock_transformer.return_value.encode.return_value = [[0.1, 0.2]]
        
        retriever = RAGRetriever(recipes_dir=self.recipes_dir)
        stats = retriever.get_stats()
        
        self.assertIn('total_recipes', stats)
        self.assertEqual(stats['total_recipes'], 2)


class TestAPIEndpoints(unittest.TestCase):
    """Test FastAPI endpoints"""
    
    def setUp(self):
        """Setup test client with mocked dependencies"""
        # Mock the startup event
        import app
        app.model_handler = Mock()
        app.rag_retriever = Mock()
        
        self.client = TestClient(app.app)
    
    def test_root_endpoint(self):
        """Test root health check endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('service', data)
        self.assertIn('status', data)
    
    def test_health_endpoint(self):
        """Test detailed health check"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    @patch('app.model_handler')
    def test_local_fix_endpoint_schema(self, mock_handler):
        """Test /local_fix validates request schema"""
        # Test invalid request (missing fields)
        response = self.client.post("/local_fix", json={})
        self.assertEqual(response.status_code, 422)  # Validation error
        
        # Test valid schema
        mock_handler.generate_fix.return_value = {
            'text': 'FIXED_CODE:\n```\nfixed\n```\nEXPLANATION:\nExplanation here',
            'input_tokens': 100,
            'output_tokens': 150
        }
        mock_handler.model_name = "test-model"
        
        response = self.client.post("/local_fix", json={
            "language": "python",
            "cwe": "CWE-89",
            "code": "vulnerable code"
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('fixed_code', data)
        self.assertIn('explanation', data)
        self.assertIn('token_usage', data)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_generate_diff(self):
        """Test diff generation"""
        from app import generate_diff
        
        original = "line1\nline2\nline3"
        fixed = "line1\nline2_modified\nline3"
        
        diff = generate_diff(original, fixed)
        
        self.assertIn('---', diff)
        self.assertIn('+++', diff)
        self.assertIn('line2', diff)
    
    def test_build_prompt(self):
        """Test prompt building"""
        from app import build_prompt
        
        prompt = build_prompt("python", "CWE-89", "vulnerable_code()")
        
        self.assertIn("python", prompt)
        self.assertIn("CWE-89", prompt)
        self.assertIn("vulnerable_code()", prompt)
        self.assertIn("FIXED_CODE:", prompt)
        self.assertIn("EXPLANATION:", prompt)
    
    def test_build_prompt_with_context(self):
        """Test prompt building with RAG context"""
        from app import build_prompt
        
        context = "Use parameterized queries for SQL"
        prompt = build_prompt("java", "CWE-89", "query = sql + input", context)
        
        self.assertIn(context, prompt)
        self.assertIn("Security Guidelines", prompt)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    @patch('app.model_handler')
    @patch('app.rag_retriever')
    def test_full_fix_workflow(self, mock_rag, mock_handler):
        """Test complete fix generation workflow"""
        from app import app as fastapi_app
        
        # Setup mocks
        mock_rag.retrieve.return_value = "Use prepared statements"
        mock_handler.generate_fix.return_value = {
            'text': 'FIXED_CODE:\n```\nPreparedStatement ps = conn.prepareStatement("SELECT * FROM users WHERE id=?");\nps.setInt(1, userId);\n```\nEXPLANATION:\nThe fix uses parameterized queries to prevent SQL injection.',
            'input_tokens': 120,
            'output_tokens': 180
        }
        mock_handler.model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        
        client = TestClient(fastapi_app)
        
        response = client.post("/local_fix", json={
            "language": "java",
            "cwe": "CWE-89",
            "code": 'String query = "SELECT * FROM users WHERE id=" + userId;'
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn('fixed_code', data)
        self.assertIn('explanation', data)
        self.assertIn('diff', data)
        self.assertIn('model_used', data)
        self.assertIn('token_usage', data)
        self.assertIn('latency_ms', data)
        
        # Verify token usage
        self.assertEqual(data['token_usage']['input_tokens'], 120)
        self.assertEqual(data['token_usage']['output_tokens'], 180)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGRetriever))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)