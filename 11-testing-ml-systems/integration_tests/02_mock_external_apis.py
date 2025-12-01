"""
Mocking External APIs for Testing

Learn how to mock external services (OpenAI, vector DBs, etc.) in tests.
Focus: pytest-mock, realistic mocking, avoiding API costs.

Install: poetry add --group dev pytest pytest-mock requests
Run: pytest integration_tests/02_mock_external_apis.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict
import requests


# ============================================================================
# 1. Why Mock External APIs?
# ============================================================================

def demo_why_mock_apis():
    """
    Why mocking external APIs is essential.
    """
    print("=" * 70)
    print("1. Why Mock External APIs?")
    print("=" * 70)
    print()
    
    print("💸 PROBLEM #1: Cost")
    print()
    print("   OpenAI API pricing:")
    print("   • GPT-4: $0.03 per 1K tokens input")
    print("   • GPT-4: $0.06 per 1K tokens output")
    print()
    print("   Test suite with 100 API calls:")
    print("   • Average: 1K tokens per call")
    print("   • Cost per run: $9.00")
    print("   • Daily runs: 10")
    print("   • Monthly cost: $2,700 💀")
    print()
    print("   With mocking:")
    print("   • Cost per run: $0")
    print("   • Monthly cost: $0 ✅")
    print()
    
    print("⏱️  PROBLEM #2: Speed")
    print()
    print("   Real API calls:")
    print("   • OpenAI API: 2-5 seconds per call")
    print("   • 100 test calls: 3-8 minutes")
    print("   • Developer waits → Context switch")
    print()
    print("   With mocking:")
    print("   • Mock returns instantly: 0.001 seconds")
    print("   • 100 test calls: 0.1 seconds")
    print("   • Developer stays focused ✅")
    print()
    
    print("🌐 PROBLEM #3: Reliability")
    print()
    print("   Real API calls:")
    print("   • Network issues → Test fails")
    print("   • Rate limits → Test fails")
    print("   • API downtime → Test fails")
    print("   • Flaky tests → Lost confidence")
    print()
    print("   With mocking:")
    print("   • No network required")
    print("   • No rate limits")
    print("   • No downtime")
    print("   • Reliable tests ✅")
    print()
    
    print("🔐 PROBLEM #4: CI/CD")
    print()
    print("   Real API calls in CI:")
    print("   • Need API keys in CI")
    print("   • Security risk")
    print("   • Key rotation breaks tests")
    print()
    print("   With mocking:")
    print("   • No API keys needed")
    print("   • No security risk")
    print("   • Tests always run ✅")
    print()


# ============================================================================
# 2. Simple Service with External API
# ============================================================================

class OpenAIService:
    """Service that calls OpenAI API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
    
    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "text-embedding-ada-002",
            "input": text
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        result = response.json()
        return result['data'][0]['embedding']


# ============================================================================
# 3. Basic Mocking with pytest-mock
# ============================================================================

def test_openai_generate_text_basic_mock(mocker):
    """Test text generation with basic mock."""
    # Create mock
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [
            {
                'message': {
                    'content': 'This is a mocked response'
                }
            }
        ]
    }
    
    # Patch requests.post
    mocker.patch('requests.post', return_value=mock_response)
    
    # Test
    service = OpenAIService(api_key="fake-key")
    result = service.generate_text("Hello")
    
    assert result == 'This is a mocked response'


def test_openai_embedding_basic_mock(mocker):
    """Test embedding creation with basic mock."""
    # Create mock
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'data': [
            {
                'embedding': [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        ]
    }
    
    # Patch requests.post
    mocker.patch('requests.post', return_value=mock_response)
    
    # Test
    service = OpenAIService(api_key="fake-key")
    embedding = service.create_embedding("Hello world")
    
    assert len(embedding) == 5
    assert embedding[0] == 0.1


def demo_basic_mocking():
    """Demo basic mocking."""
    print("\n" + "=" * 70)
    print("2. Basic Mocking with pytest-mock")
    print("=" * 70)
    print()
    
    print("🔧 BASIC MOCK PATTERN:")
    print()
    print("   def test_api_call(mocker):")
    print("       # 1. Create mock response")
    print("       mock_response = Mock()")
    print("       mock_response.status_code = 200")
    print("       mock_response.json.return_value = {...}")
    print("       ")
    print("       # 2. Patch the function")
    print("       mocker.patch('requests.post', return_value=mock_response)")
    print("       ")
    print("       # 3. Test")
    print("       service = OpenAIService('fake-key')")
    print("       result = service.generate_text('Hello')")
    print("       ")
    print("       # 4. Assert")
    print("       assert result == 'expected'")
    print()


# ============================================================================
# 4. Verifying Mock Calls
# ============================================================================

def test_openai_api_called_with_correct_params(mocker):
    """Test that API is called with correct parameters."""
    mock_post = mocker.patch('requests.post')
    mock_post.return_value = Mock(
        status_code=200,
        json=Mock(return_value={
            'choices': [{'message': {'content': 'Response'}}]
        })
    )
    
    service = OpenAIService(api_key="test-key")
    service.generate_text("Hello", max_tokens=50)
    
    # Verify API was called
    assert mock_post.called
    assert mock_post.call_count == 1
    
    # Verify call arguments
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"
    
    # Verify JSON payload
    json_data = call_args[1]['json']
    assert json_data['model'] == 'gpt-4'
    assert json_data['max_tokens'] == 50
    assert json_data['messages'][0]['content'] == 'Hello'
    
    # Verify headers
    headers = call_args[1]['headers']
    assert headers['Authorization'] == 'Bearer test-key'


def test_openai_api_not_called_on_cached_result(mocker):
    """Test that API is not called when result is cached."""
    # This would be for a caching implementation
    mock_post = mocker.patch('requests.post')
    
    # Simulate cached result
    # (In real code, you'd have caching logic)
    
    # Verify API was NOT called
    assert not mock_post.called


def demo_verifying_calls():
    """Demo verifying mock calls."""
    print("\n" + "=" * 70)
    print("3. Verifying Mock Calls")
    print("=" * 70)
    print()
    
    print("✅ VERIFY MOCK WAS CALLED:")
    print()
    print("   mock_func = mocker.patch('module.function')")
    print("   ")
    print("   # Call function")
    print("   service.do_something()")
    print("   ")
    print("   # Verify call")
    print("   assert mock_func.called")
    print("   assert mock_func.call_count == 1")
    print()
    
    print("🔍 VERIFY CALL ARGUMENTS:")
    print()
    print("   call_args = mock_func.call_args")
    print("   assert call_args[0][0] == 'expected_arg'")
    print("   assert call_args[1]['key'] == 'expected_value'")
    print()
    
    print("💡 WHY VERIFY CALLS:")
    print()
    print("   Ensures:")
    print("   • API called with correct parameters")
    print("   • API not called when cached")
    print("   • Retry logic works")
    print("   • Rate limiting respected")
    print()


# ============================================================================
# 5. Mocking Error Cases
# ============================================================================

def test_openai_api_error_handling(mocker):
    """Test handling of API errors."""
    # Mock API error
    mock_response = Mock()
    mock_response.status_code = 429  # Rate limit
    
    mocker.patch('requests.post', return_value=mock_response)
    
    service = OpenAIService(api_key="test-key")
    
    # Should raise exception
    with pytest.raises(Exception, match="API error: 429"):
        service.generate_text("Hello")


def test_openai_network_error_handling(mocker):
    """Test handling of network errors."""
    # Mock network error
    mocker.patch(
        'requests.post',
        side_effect=requests.exceptions.ConnectionError("Network error")
    )
    
    service = OpenAIService(api_key="test-key")
    
    # Should raise exception
    with pytest.raises(requests.exceptions.ConnectionError):
        service.generate_text("Hello")


def test_openai_timeout_handling(mocker):
    """Test handling of timeout errors."""
    # Mock timeout
    mocker.patch(
        'requests.post',
        side_effect=requests.exceptions.Timeout("Request timed out")
    )
    
    service = OpenAIService(api_key="test-key")
    
    # Should raise exception
    with pytest.raises(requests.exceptions.Timeout):
        service.generate_text("Hello")


def demo_mocking_errors():
    """Demo mocking error cases."""
    print("\n" + "=" * 70)
    print("4. Mocking Error Cases")
    print("=" * 70)
    print()
    
    print("🚨 MOCK API ERRORS:")
    print()
    print("   # Rate limit error")
    print("   mock_response = Mock()")
    print("   mock_response.status_code = 429")
    print("   mocker.patch('requests.post', return_value=mock_response)")
    print()
    print("   # Network error")
    print("   mocker.patch(")
    print("       'requests.post',")
    print("       side_effect=ConnectionError('Network error')")
    print("   )")
    print()
    print("   # Timeout")
    print("   mocker.patch(")
    print("       'requests.post',")
    print("       side_effect=Timeout('Timeout')")
    print("   )")
    print()
    
    print("💡 ERRORS TO TEST:")
    print()
    print("   • 429: Rate limit")
    print("   • 500: Server error")
    print("   • 401: Unauthorized")
    print("   • ConnectionError: Network issues")
    print("   • Timeout: Request timeout")
    print()


# ============================================================================
# 6. Realistic Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_successful(mocker):
    """Fixture providing successful OpenAI mock."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [
            {
                'message': {
                    'content': 'Mocked AI response'
                }
            }
        ]
    }
    
    return mocker.patch('requests.post', return_value=mock_response)


@pytest.fixture
def mock_openai_rate_limited(mocker):
    """Fixture providing rate-limited OpenAI mock."""
    mock_response = Mock()
    mock_response.status_code = 429
    
    return mocker.patch('requests.post', return_value=mock_response)


def test_with_successful_mock(mock_openai_successful):
    """Test with successful API mock."""
    service = OpenAIService(api_key="test-key")
    result = service.generate_text("Hello")
    
    assert result == 'Mocked AI response'


def test_with_rate_limited_mock(mock_openai_rate_limited):
    """Test with rate-limited API mock."""
    service = OpenAIService(api_key="test-key")
    
    with pytest.raises(Exception, match="429"):
        service.generate_text("Hello")


def demo_mock_fixtures():
    """Demo mock fixtures."""
    print("\n" + "=" * 70)
    print("5. Realistic Mock Fixtures")
    print("=" * 70)
    print()
    
    print("🔧 REUSABLE MOCK FIXTURES:")
    print()
    print("   @pytest.fixture")
    print("   def mock_openai_successful(mocker):")
    print("       mock = Mock()")
    print("       mock.status_code = 200")
    print("       mock.json.return_value = {...}")
    print("       return mocker.patch('requests.post', return_value=mock)")
    print()
    print("   def test_with_mock(mock_openai_successful):")
    print("       # Mock automatically applied!")
    print("       service = OpenAIService('key')")
    print("       result = service.generate_text('Hello')")
    print()
    
    print("💡 BENEFITS:")
    print()
    print("   • Reusable across tests")
    print("   • Consistent mock behavior")
    print("   • Less code duplication")
    print("   • Easy to maintain")
    print()


# ============================================================================
# 7. Best Practices
# ============================================================================

def demo_mocking_best_practices():
    """Demo mocking best practices."""
    print("\n" + "=" * 70)
    print("6. Mocking Best Practices")
    print("=" * 70)
    print()
    
    print("✅ DO:")
    print()
    print("   • Mock external APIs (OpenAI, databases)")
    print("   • Mock expensive operations")
    print("   • Mock non-deterministic operations")
    print("   • Test error cases")
    print("   • Use fixtures for reusable mocks")
    print("   • Verify mock was called correctly")
    print()
    
    print("❌ DON'T:")
    print()
    print("   • Mock your own code (test real code)")
    print("   • Mock too much (integration tests need real flow)")
    print("   • Make mocks too complex")
    print("   • Forget to test without mocks (E2E tests)")
    print()
    
    print("🎯 WHEN TO MOCK:")
    print()
    print("   Unit tests:")
    print("   ✅ Always mock external APIs")
    print("   ✅ Fast, isolated tests")
    print()
    print("   Integration tests:")
    print("   ⚠️  Sometimes mock (depends on test type)")
    print("   • Mock expensive calls")
    print("   • Use real calls for critical paths")
    print()
    print("   E2E tests:")
    print("   ❌ Don't mock")
    print("   • Test real system")
    print("   • Use staging APIs")
    print()


# ============================================================================
# Run Demonstrations
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n🧪 Mocking External APIs\n")
    
    demo_why_mock_apis()
    demo_basic_mocking()
    demo_verifying_calls()
    demo_mocking_errors()
    demo_mock_fixtures()
    demo_mocking_best_practices()
    
    print("\n" + "=" * 70)
    print("✅ Key Takeaways")
    print("=" * 70)
    print("""
1. Why Mock APIs:
   💸 Cost: Save $2,700/month in API costs
   ⚡ Speed: 0.1s vs 8 minutes for 100 tests
   🔒 Reliability: No network, rate limits, downtime
   🚀 CI/CD: No API keys needed

2. Basic Mocking:
   mock_response = Mock()
   mock_response.status_code = 200
   mock_response.json.return_value = {...}
   mocker.patch('requests.post', return_value=mock_response)

3. Verify Mock Calls:
   assert mock.called
   assert mock.call_count == 1
   call_args = mock.call_args
   assert call_args[1]['json']['key'] == 'value'

4. Mock Error Cases:
   ✅ Rate limits (429)
   ✅ Server errors (500)
   ✅ Network errors (ConnectionError)
   ✅ Timeouts (Timeout)

5. Mock Fixtures:
   @pytest.fixture
   def mock_api(mocker):
       return mocker.patch(...)
   
   def test_with_mock(mock_api):
       # Reusable mock!

Mocking Checklist:
```
Setup:
□ Install pytest-mock
□ Create mock fixtures
□ Mock external APIs

Success Cases:
□ Mock successful API responses
□ Verify correct parameters passed
□ Test response parsing

Error Cases:
□ Mock rate limits (429)
□ Mock server errors (500)
□ Mock network errors
□ Mock timeouts
□ Verify error handling

Best Practices:
□ Mock external APIs only
□ Keep mocks simple
□ Use fixtures for reusability
□ Test without mocks too (E2E)
```

pytest-mock Examples:
```python
# Basic mock
def test_api(mocker):
    mock = mocker.patch('requests.post')
    mock.return_value = Mock(status_code=200)
    # test...

# Mock with side effect
def test_error(mocker):
    mocker.patch(
        'requests.post',
        side_effect=ConnectionError('Network error')
    )
    # test...

# Verify calls
def test_verify(mocker):
    mock = mocker.patch('requests.post')
    # call function...
    assert mock.called
    assert mock.call_count == 1
    mock.assert_called_once_with(...)
```

Cost Savings:
Without mocking: $2,700/month
With mocking: $0/month
ROI: ∞ ✅

Next Steps:
→ 03_database_tests.py (Test with databases)
→ 04_performance_tests.py (Load testing)
""")


if __name__ == "__main__":
    main()
