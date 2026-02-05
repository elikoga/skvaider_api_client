import pytest
from unittest.mock import patch, mock_open, MagicMock, AsyncMock
from skvaider_api_client.cli import _main, main
from skvaider_api_client.benchmarks.base import BaseBenchmark
import argparse


@pytest.fixture
def mock_config_data():
    return {
        "skvaider_token": "test-token-123",
        "skvaider_url": "http://localhost:8000/v1"
    }


@pytest.fixture
def mock_config_file(mock_config_data):
    """Mock TOML config file"""
    import toml
    return toml.dumps(mock_config_data)


@pytest.mark.asyncio
async def test_cli_no_command(capsys, mock_config_file):
    """Test CLI with no command shows help"""
    with patch("sys.argv", ["skvaider-api-client", "--config", "config.toml"]):
        with patch("builtins.open", mock_open(read_data=mock_config_file)):
            await _main()
            captured = capsys.readouterr()
            # Should print help when no command is provided
            assert "usage:" in captured.out or len(captured.out) > 0


@pytest.mark.asyncio
async def test_cli_config_not_found(capsys):
    """Test CLI with missing config file"""
    with patch("sys.argv", ["skvaider-api-client", "--config", "missing.toml"]):
        await _main()
        captured = capsys.readouterr()
        assert "Config file not found" in captured.out


@pytest.mark.asyncio
async def test_cli_parallel_benchmark(mock_config_file):
    """Test CLI running parallel benchmark"""
    with patch("sys.argv", ["skvaider-api-client", "--config", "config.toml", "benchmark"]):
        with patch("builtins.open", mock_open(read_data=mock_config_file)):
            # Mock the run method at the class level
            with patch("skvaider_api_client.benchmarks.parallel.ParallelBenchmark.run", new_callable=AsyncMock) as mock_run:
                await _main()
                
                # Verify run was called
                mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_cli_batch_api_benchmark(mock_config_file):
    """Test CLI running batch-api benchmark"""
    with patch("sys.argv", ["skvaider-api-client", "--config", "config.toml", "benchmark-batch"]):
        with patch("builtins.open", mock_open(read_data=mock_config_file)):
            # Mock the run method at the class level
            with patch("skvaider_api_client.benchmarks.batch_api.BatchApiBenchmark.run", new_callable=AsyncMock) as mock_run:
                await _main()
                
                # Verify run was called
                mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_cli_mixed_benchmark(mock_config_file):
    """Test CLI running mixed benchmark"""
    with patch("sys.argv", ["skvaider-api-client", "--config", "config.toml", "benchmark-mixed"]):
        with patch("builtins.open", mock_open(read_data=mock_config_file)):
            # Mock the run method at the class level
            with patch("skvaider_api_client.benchmarks.mixed.MixedBenchmark.run", new_callable=AsyncMock) as mock_run:
                await _main()
                
                # Verify run was called
                mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_cli_custom_config_path(mock_config_file):
    """Test CLI with custom config path"""
    custom_path = "custom-config.toml"
    with patch("sys.argv", ["skvaider-api-client", "--config", custom_path, "benchmark"]):
        with patch("builtins.open", mock_open(read_data=mock_config_file)) as m:
            # Mock the run method at the class level
            with patch("skvaider_api_client.benchmarks.parallel.ParallelBenchmark.run", new_callable=AsyncMock):
                await _main()
                
                # Verify the custom config path was opened (along with dataset.txt)
                # Check that the first call was to the custom config path
                assert m.call_args_list[0][0][0] == custom_path


def test_main_wrapper():
    """Test the synchronous main() wrapper"""
    with patch("asyncio.run") as mock_run:
        main()
        # Just verify asyncio.run was called once
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_cli_benchmark_receives_args(mock_config_file):
    """Test that benchmark receives both config and parsed args"""
    with patch("sys.argv", ["skvaider-api-client", "--config", "config.toml", "benchmark"]):
        with patch("builtins.open", mock_open(read_data=mock_config_file)):
            # Mock the __init__ to capture arguments
            captured_args = {}
            
            original_init = BaseBenchmark.__init__
            
            def capture_init(self, config, args):
                captured_args['config'] = config
                captured_args['args'] = args
                # Skip the rest of init to avoid network calls
            
            with patch.object(BaseBenchmark, '__init__', capture_init):
                with patch("skvaider_api_client.benchmarks.parallel.ParallelBenchmark.run", new_callable=AsyncMock):
                    await _main()
                    
                    # Verify config is Config object
                    assert hasattr(captured_args['config'], 'skvaider_url')
                    assert captured_args['config'].skvaider_url == "http://localhost:8000/v1"
                    
                    # Verify args is argparse.Namespace
                    assert isinstance(captured_args['args'], argparse.Namespace)
                    assert captured_args['args'].command == "benchmark"
