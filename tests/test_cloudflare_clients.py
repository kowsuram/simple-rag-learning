"""Tests for the Cloudflare Workers AI request adapters."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from src.agent import HealthcareAgent
from src.embedder import OllamaEmbedder


class TestCloudflareClients(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            ai_provider="cloudflare",
            cloudflare_ai_base_url="https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1",
            cloudflare_headers={"Authorization": "Bearer test-token"},
            embedding_model="@cf/baai/bge-base-en-v1.5",
            embedding_dim=768,
            ollama_base_url="http://localhost:11434",
            llm_model="@cf/meta/llama-3.1-8b-instruct",
        )

    def test_embedding_uses_cloudflare_openai_compatible_api(self):
        response = Mock()
        response.json.return_value = {"data": [{"embedding": [0.25] * 768}]}

        with (
            patch("src.embedder.config", self.config),
            patch("src.embedder.requests.post", return_value=response) as post,
        ):
            result = OllamaEmbedder().get_embedding("test text", verbose=False)

        self.assertEqual(len(result), 768)
        self.assertEqual(
            post.call_args.args[0],
            f"{self.config.cloudflare_ai_base_url}/embeddings",
        )
        self.assertEqual(
            post.call_args.kwargs["headers"],
            self.config.cloudflare_headers,
        )
        self.assertEqual(post.call_args.kwargs["json"]["input"], "test text")

    def test_chat_uses_cloudflare_openai_compatible_api(self):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Cloudflare response"}}]
        }
        messages = [{"role": "user", "content": "hello"}]

        with (
            patch("src.agent.config", self.config),
            patch("src.agent.requests.post", return_value=response) as post,
        ):
            result = HealthcareAgent()._generate_chat_response(messages, temperature=0.1)

        self.assertEqual(result, "Cloudflare response")
        self.assertEqual(
            post.call_args.args[0],
            f"{self.config.cloudflare_ai_base_url}/chat/completions",
        )
        self.assertEqual(
            post.call_args.kwargs["headers"],
            self.config.cloudflare_headers,
        )

    def test_chat_error_includes_cloudflare_status_and_message_without_token(self):
        response = Mock()
        response.status_code = 429
        response.json.return_value = {
            "errors": [{"message": "Daily usage limit exceeded"}]
        }
        http_error = requests.exceptions.HTTPError(response=response)
        failed_response = Mock()
        failed_response.raise_for_status.side_effect = http_error
        failed_response.json.side_effect = AssertionError("JSON must not be parsed after HTTP failure")

        with (
            patch("src.agent.config", self.config),
            patch("src.agent.requests.post", return_value=failed_response),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "HTTP 429.*Daily usage limit exceeded",
            ) as error:
                HealthcareAgent()._generate_chat_response(
                    [{"role": "user", "content": "hello"}],
                    temperature=0.1,
                )

        self.assertNotIn("test-token", str(error.exception))


if __name__ == "__main__":
    unittest.main()
