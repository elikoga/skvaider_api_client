import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from skvaider_api_client.client import APIClient, Config

@pytest.fixture
def mock_config():
    return Config(skvaider_token="tok", skvaider_url="http://mock")

@pytest.mark.asyncio
async def test_client_init(mock_config):
    client = APIClient(mock_config)
    assert client.token == "tok"
    assert client.url == "http://mock"
    assert client.headers["Authorization"] == "Bearer tok"

@pytest.mark.asyncio
async def test_list_models(mock_config):
    client = APIClient(mock_config)
    
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": []}
    mock_response.raise_for_status = MagicMock()

    client.client.get = AsyncMock(return_value=mock_response)

    res = await client.list_models()
    assert res == {"models": []}
    client.client.get.assert_called_with("http://mock/models")

@pytest.mark.asyncio
async def test_get_completion(mock_config):
    client = APIClient(mock_config)
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": []}
    client.client.post = AsyncMock(return_value=mock_response)

    await client.get_completion("model-1", "hello", max_tokens=50)
    
    client.client.post.assert_called_once()
    args, kwargs = client.client.post.call_args
    assert args[0] == "http://mock/chat/completions"
    assert kwargs["json"]["model"] == "model-1"
    assert kwargs["json"]["messages"][0]["content"] == "hello"
    assert kwargs["json"]["max_tokens"] == 50

@pytest.mark.asyncio
async def test_get_batch_completion(mock_config):
    client = APIClient(mock_config)
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": []}
    client.client.post = AsyncMock(return_value=mock_response)

    prompts = ["p1", "p2"]
    await client.get_batch_completion("model-1", prompts, max_tokens=10)
    
    client.client.post.assert_called_once()
    args, kwargs = client.client.post.call_args
    assert args[0] == "http://mock/completions"
    assert kwargs["json"]["prompt"] == prompts
