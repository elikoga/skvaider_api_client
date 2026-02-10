import httpx

from .config import Config


class APIClient:
    token: str
    url: str
    timeout: httpx.Timeout
    headers: dict
    client: httpx.AsyncClient

    def __init__(self, config: Config):
        self.token = config.skvaider_token
        self.url = config.skvaider_url
        self.timeout = httpx.Timeout(300.0, read=300.0)
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.client = httpx.AsyncClient(timeout=self.timeout, headers=self.headers)

    async def list_models(self):
        response = await self.client.get(f"{self.url}/models")
        response.raise_for_status()
        return response.json()

    async def get_completion(self, model_id: str, prompt: str, max_tokens: int = 100) -> dict:
        # Use max_tokens instead of max_completion_tokens for compatibility
        json_data = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = await self.client.post(f"{self.url}/chat/completions", json=json_data)
        response.raise_for_status()
        return response.json()

    async def get_batch_completion(self, model_id: str, prompts: list[str], max_tokens: int = 100) -> dict:
        """Use /completions endpoint with prompt as a list for true batching"""
        json_data = {
            "model": model_id,
            "prompt": prompts,  # List of prompts for batching
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = await self.client.post(f"{self.url}/completions", json=json_data)
        response.raise_for_status()
        return response.json()
