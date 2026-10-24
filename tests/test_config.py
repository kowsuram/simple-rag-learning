"""Tests for database connection configuration."""

import os
import unittest
from unittest.mock import patch

from src.config import AppConfig


class TestAppConfig(unittest.TestCase):
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
