import os
import signal
import sys
import unittest

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from core import interrupt_handler


class InterruptShutdownTests(unittest.TestCase):
    def setUp(self):
        interrupt_handler.shutdown_requested.clear()

    def tearDown(self):
        interrupt_handler.shutdown_requested.clear()

    def test_direct_handler_call_sets_shutdown_event(self):
        interrupt_handler.install()
        self.assertFalse(interrupt_handler.shutdown_requested.is_set())
        interrupt_handler._handler(signal.SIGINT, None)
        self.assertTrue(interrupt_handler.shutdown_requested.is_set())

    def test_real_sigint_signal_sets_shutdown_event(self):
        interrupt_handler.install()
        self.assertFalse(interrupt_handler.shutdown_requested.is_set())
        signal.raise_signal(signal.SIGINT)
        self.assertTrue(interrupt_handler.shutdown_requested.is_set())

    def test_real_sigterm_signal_sets_shutdown_event(self):
        interrupt_handler.install()
        self.assertFalse(interrupt_handler.shutdown_requested.is_set())
        signal.raise_signal(signal.SIGTERM)
        self.assertTrue(interrupt_handler.shutdown_requested.is_set())

    def test_context_manager_sets_event_and_restores_previous_handler(self):
        previous = signal.getsignal(signal.SIGINT)
        with interrupt_handler.installed() as event:
            self.assertIs(event, interrupt_handler.shutdown_requested)
            signal.raise_signal(signal.SIGINT)
            self.assertTrue(event.is_set())
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)


if __name__ == "__main__":
    unittest.main()
