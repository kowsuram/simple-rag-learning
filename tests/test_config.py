"""Tests for database connection configuration."""

import os
import unittest
from unittest.mock import patch

from src.config import AppConfig


class TestAppConfig(unittest.TestCase):
    def test_cloudflare_headers_route_through_configured_gateway(self):
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "cloudflare",
                "CLOUDFLARE_ACCOUNT_ID": "test-account",
                "CLOUDFLARE_API_TOKEN": "test-token",
                "CLOUDFLARE_AI_GATEWAY_ID": "healthcare-gateway",
            },
            clear=True,
        ):
            config = AppConfig()

        self.assertEqual(
            config.cloudflare_headers,
            {
                "Authorization": "Bearer test-token",
                "cf-aig-gateway-id": "healthcare-gateway",
            },
        )

    def test_cloudflare_requires_gateway_id(self):
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "cloudflare",
                "CLOUDFLARE_ACCOUNT_ID": "test-account",
                "CLOUDFLARE_API_TOKEN": "test-token",
            },
            clear=True,
        ):
            config = AppConfig()

        with self.assertRaisesRegex(ValueError, "CLOUDFLARE_AI_GATEWAY_ID"):
            _ = config.cloudflare_headers

    def test_cloudflare_credentials_select_cloudflare_defaults(self):
        with patch.dict(
            os.environ,
            {
                "CLOUDFLARE_ACCOUNT_ID": "test-account",
                "CLOUDFLARE_API_TOKEN": "test-token",
            },
            clear=True,
        ):
            config = AppConfig()

        self.assertEqual(config.ai_provider, "cloudflare")
        self.assertEqual(config.llm_model, "@cf/meta/llama-3.2-3b-instruct")
        self.assertEqual(config.embedding_model, "@cf/baai/bge-base-en-v1.5")
        self.assertEqual(config.embedding_dim, 768)

    def test_database_url_takes_precedence_over_local_settings(self):
        database_url = "postgresql://user:password@db.example/neondb?sslmode=require"
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": database_url,
                "DB_HOST": "localhost",
                "DB_PORT": "5433",
                "DB_NAME": "healthcaredb",
            },
            clear=True,
        ):
            config = AppConfig()

        self.assertEqual(config.postgres_dsn, database_url)

    def test_local_database_settings_still_build_dsn(self):
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "localhost",
                "DB_PORT": "5433",
                "DB_NAME": "healthcaredb",
                "DB_USER": "postgres",
                "DB_PASSWORD": "local-password",
            },
            clear=True,
        ):
            config = AppConfig()

        self.assertEqual(
            config.postgres_dsn,
            "postgresql://postgres:local-password@localhost:5433/healthcaredb",
        )


if __name__ == "__main__":
    unittest.main()
