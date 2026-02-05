import pytest
from unittest.mock import patch, AsyncMock, mock_open, ANY
import sys
from skvaider_api_client import cli

@pytest.mark.asyncio
async def test_cli_benchmark_command():
    test_args = [
        "skvaider-api-client",  # prog name
        "--config", "test_config.toml",
        "benchmark",
        "--model", "test-model",
        "--batch-sizes", "1,2"
    ]
    
    with patch.object(sys, "argv", test_args):
        # mock open config
        with patch("builtins.open", mock_open(read_data='skvaider_token="t"\nskvaider_url="u"')):
            # mock command execution
            with patch("skvaider_api_client.cli.benchmark_command", new_callable=AsyncMock) as mock_cmd:
                await cli._main()
                mock_cmd.assert_called_once()
                # verify args passed
                call_args = mock_cmd.call_args
                config = call_args[0][0]
                assert config.skvaider_token == "t"
                assert call_args[0][1] == "test-model" # model_id
                assert call_args[0][4] == [1, 2] # batch_sizes

@pytest.mark.asyncio
async def test_cli_benchmark_batch_command():
    test_args = ["prog", "benchmark-batch", "--model", "m", "--batch-sizes", "5"]
    with patch.object(sys, "argv", test_args):
        with patch("builtins.open", mock_open(read_data='skvaider_token="t"\nskvaider_url="u"')):
            with patch("skvaider_api_client.cli.benchmark_batch_command", new_callable=AsyncMock) as mock_cmd:
                await cli._main()
                mock_cmd.assert_called_once_with(
                    ANY, "m", "dataset.txt", 100, [5], "benchmark_batch_results.json"
                )

@pytest.mark.asyncio
async def test_cli_benchmark_mixed_command():
    test_args = ["prog", "benchmark-mixed", "--model", "m", "--requests-at-once", "9"]
    with patch.object(sys, "argv", test_args):
        with patch("builtins.open", mock_open(read_data='skvaider_token="t"\nskvaider_url="u"')):
            with patch("skvaider_api_client.cli.benchmark_mixed_command", new_callable=AsyncMock) as mock_cmd:
                await cli._main()
                mock_cmd.assert_called_once()
                assert mock_cmd.call_args[0][5] == 9 # requests_at_once
