"""Tests for multi-backend LLM client interface."""

import time
from unittest.mock import patch, MagicMock, PropertyMock
import pytest
from llm_client import BaseLLMClient, OpenAICompatibleClient, AnthropicClient


REQUIRED_KEYS = {
    'response', 'tokens_generated', 'processing_time_ms',
    'model_key', 'model_name', 'model_family', 'model_size_b',
    'quantization', 'inference_params', 'ollama_version',
}

# Test model config used by all tests
TEST_MODEL_CONFIG = {
    'test-openai-model': {
        'backend': 'openai',
        'api_model_name': 'gpt-4o-mini',
        'family': 'gpt',
        'size_b': 0.0,
        'context_length': 128000,
    },
    'test-anthropic-model': {
        'backend': 'anthropic',
        'api_model_name': 'claude-3-5-haiku-20241022',
        'family': 'claude',
        'size_b': 0.0,
        'context_length': 200000,
    },
}


class TestBaseLLMClientNoOps:
    """Base class provides sensible no-op defaults."""

    def test_ensure_model_loaded_is_noop(self):
        class DummyClient(BaseLLMClient):
            def generate(self, **kwargs):
                return {}
        client = DummyClient()
        client.ensure_model_loaded('any-model')  # should not raise

    def test_check_models_available_returns_empty(self):
        class DummyClient(BaseLLMClient):
            def generate(self, **kwargs):
                return {}
        client = DummyClient()
        assert client.check_models_available() == {}

    def test_list_available_models_returns_empty(self):
        class DummyClient(BaseLLMClient):
            def generate(self, **kwargs):
                return {}
        client = DummyClient()
        assert client.list_available_models() == []

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseLLMClient()


class TestOpenAICompatibleClient:
    """OpenAI-compatible client returns all required keys."""

    @patch('llm_client.OpenAICompatibleClient.__init__', return_value=None)
    def test_returns_all_required_keys(self, mock_init):
        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        client.backend_name = 'openai'
        client._version = 'openai/1.30.0'

        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '0.85'
        mock_response.usage.completion_tokens = 3
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_response

        with patch('config.MODELS_TO_TEST', TEST_MODEL_CONFIG):
            result = client.generate(
                model_key='test-openai-model',
                prompt='test prompt',
                temperature=0.0,
                max_tokens=50,
            )

        assert REQUIRED_KEYS.issubset(result.keys()), \
            f"Missing keys: {REQUIRED_KEYS - set(result.keys())}"
        assert result['response'] == '0.85'
        assert result['model_family'] == 'gpt'
        assert isinstance(result['inference_params'], dict)

    @patch('llm_client.OpenAICompatibleClient.__init__', return_value=None)
    def test_seed_set_for_temp_zero(self, mock_init):
        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        client.backend_name = 'openai'
        client._version = 'openai/1.30.0'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '0.5'
        mock_response.usage.completion_tokens = 1
        client.client = MagicMock()
        client.client.chat.completions.create.return_value = mock_response

        with patch('config.MODELS_TO_TEST', TEST_MODEL_CONFIG):
            result = client.generate(
                model_key='test-openai-model',
                prompt='test',
                temperature=0.0,
            )

        assert result['inference_params']['seed'] == 42

    @patch('llm_client.OpenAICompatibleClient.__init__', return_value=None)
    def test_unknown_model_raises(self, mock_init):
        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        client.backend_name = 'openai'

        with patch('config.MODELS_TO_TEST', TEST_MODEL_CONFIG):
            with pytest.raises(ValueError, match="Unknown model key"):
                client.generate(model_key='nonexistent', prompt='test')


class TestAnthropicClient:
    """Anthropic client returns all required keys."""

    @patch('llm_client.AnthropicClient.__init__', return_value=None)
    def test_returns_all_required_keys(self, mock_init):
        client = AnthropicClient.__new__(AnthropicClient)
        client._version = 'anthropic/0.30.0'

        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '0.9'
        mock_response.usage.output_tokens = 2
        client.client = MagicMock()
        client.client.messages.create.return_value = mock_response

        with patch('config.MODELS_TO_TEST', TEST_MODEL_CONFIG):
            result = client.generate(
                model_key='test-anthropic-model',
                prompt='test prompt',
                temperature=0.5,
                max_tokens=50,
            )

        assert REQUIRED_KEYS.issubset(result.keys()), \
            f"Missing keys: {REQUIRED_KEYS - set(result.keys())}"
        assert result['response'] == '0.9'
        assert result['model_family'] == 'claude'
        assert result['inference_params']['seed'] is None  # Anthropic doesn't support seed

    @patch('llm_client.AnthropicClient.__init__', return_value=None)
    def test_empty_response_handled(self, mock_init):
        client = AnthropicClient.__new__(AnthropicClient)
        client._version = 'anthropic/0.30.0'

        mock_response = MagicMock()
        mock_response.content = []
        mock_response.usage.output_tokens = 0
        client.client = MagicMock()
        client.client.messages.create.return_value = mock_response

        with patch('config.MODELS_TO_TEST', TEST_MODEL_CONFIG):
            result = client.generate(
                model_key='test-anthropic-model',
                prompt='test',
            )

        assert result['response'] == ''
        assert result['tokens_generated'] == 0
