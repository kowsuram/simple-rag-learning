"""Tests for optional Langfuse tracing behavior."""

import os
import unittest
from unittest.mock import Mock, patch

from src.observability import flush_langfuse, trace_observation


class TestLangfuseTracing(unittest.TestCase):
    def test_tracing_is_disabled_without_credentials(self):
        def operation(value):
            return value + 1

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.observability.observe") as observe,
        ):
            traced_operation = trace_observation("test")(operation)

        self.assertIs(traced_operation, operation)
        observe.assert_not_called()

    def test_tracing_decorator_disables_input_and_output_capture_by_default(self):
        decorator = Mock(return_value=lambda func: func)
        with (
            patch.dict(
                os.environ,
                {
                    "LANGFUSE_PUBLIC_KEY": "public",
                    "LANGFUSE_SECRET_KEY": "secret",
                },
                clear=True,
            ),
            patch("src.observability.observe", decorator),
        ):
            traced_operation = trace_observation("test", as_type="chain")(lambda: "result")

        self.assertTrue(callable(traced_operation))
        decorator.assert_called_once_with(
            name="test",
            as_type="chain",
            capture_input=False,
            capture_output=False,
        )

    def test_tracing_decorator_can_capture_input_and_output(self):
        decorator = Mock(return_value=lambda func: func)
        with (
            patch.dict(
                os.environ,
                {
                    "LANGFUSE_PUBLIC_KEY": "public",
                    "LANGFUSE_SECRET_KEY": "secret",
                },
                clear=True,
            ),
            patch("src.observability.observe", decorator),
        ):
            traced_operation = trace_observation(
                "test",
                as_type="generation",
                capture_input=True,
                capture_output=True,
            )(lambda: "result")

        self.assertTrue(callable(traced_operation))
        decorator.assert_called_once_with(
            name="test",
            as_type="generation",
            capture_input=True,
            capture_output=True,
        )

    def test_flush_only_runs_when_credentials_are_configured(self):
        client = Mock()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.observability.get_client", return_value=client),
        ):
            flush_langfuse()
        client.flush.assert_not_called()

        with (
            patch.dict(
                os.environ,
                {
                    "LANGFUSE_PUBLIC_KEY": "public",
                    "LANGFUSE_SECRET_KEY": "secret",
                },
                clear=True,
            ),
            patch("src.observability.get_client", return_value=client),
        ):
            flush_langfuse()
        client.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
