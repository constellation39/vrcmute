import asyncio
import time

from pythonosc import udp_client
from pythonosc.osc_message_builder import ArgValue


class VRCClient:
    """A client for sending messages to VRChat using OSC protocol."""

    CHATBOX_TYPING = "/chatbox/typing"
    CHATBOX_INPUT = "/chatbox/input"

    def __init__(
        self,
        address: str = "127.0.0.1",
        port: int = 9000,
        interval: float = 1.333,
        period: float = 0.1,
    ):
        """
        Initialize the VRCClient.

        Args:
            address (str): The IP address to connect to. Defaults to "127.0.0.1".
            port (int): The port to connect to. Defaults to 9000.
            interval (float): The minimum interval between messages in seconds. Defaults to 1.333.
            period (float): The period in seconds for periodic processing. Defaults to 0.1.

        Raises:
            ValueError: If interval or period is not positive.
        """
        if interval <= 0:
            raise ValueError("Interval must be a positive number.")
        if period <= 0:
            raise ValueError("Period must be a positive number.")

        self._osc_client = udp_client.SimpleUDPClient(address, port)
        self.interval = interval
        self.period = period
        self.last_send_chatbox_time = 0
        self._lock = asyncio.Lock()
        self._pending_message: ArgValue | None = None
        self._pending_message_event = asyncio.Event()
        self._running = asyncio.Event()

    async def is_interval_reached(self) -> bool:
        """Check if the minimum interval has been reached since the last message."""
        return time.time() - self.last_send_chatbox_time >= self.interval

    async def send_chatbox(self, content: ArgValue) -> None:
        """
        Send a message to the VRChat chatbox.

        If the minimum interval hasn't been reached, it will queue the message for later sending.

        Args:
            content (ArgValue): The message to send.

        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("Chatbox content cannot be empty.")

        async with self._lock:
            if not await self.is_interval_reached():
                self._pending_message = content
                self._pending_message_event.set()
                await self.send_message(self.CHATBOX_TYPING, True)
                return

            await self._send_chatbox_immediately(content)

    async def _send_chatbox_immediately(self, content: ArgValue) -> None:
        """
        Send a chatbox message immediately.

        This method assumes the caller has already checked the interval and holds the lock.

        Args:
            content (ArgValue): The message to send.
        """
        self.last_send_chatbox_time = time.time()
        self._pending_message = None
        self._pending_message_event.clear()
        await self.send_message(self.CHATBOX_TYPING, False)
        await self.send_message(self.CHATBOX_INPUT, content)

    async def send_message(self, address: str, value: ArgValue) -> None:
        """
        Send an OSC message.

        Args:
            address (str): The OSC address to send the message to.
            value: The value to send. Must be a bool, str, int, or float.

        Raises:
            ConnectionError: If there's an error sending the message.
        """
        try:
            self._osc_client.send_message(address, value)
        except Exception as e:
            raise ConnectionError(f"Failed to send OSC message: {e}") from e

    async def start_periodic_processing(self) -> None:
        """
        Start periodically processing pending messages.
        """
        self._running.set()
        while self._running.is_set():
            await self._pending_message_event.wait()
            async with self._lock:
                if await self.is_interval_reached() and self._pending_message:
                    await self._send_chatbox_immediately(self._pending_message)
            await asyncio.sleep(self.period)

    def stop_periodic_processing(self) -> None:
        """
        Stop the periodic processing.
        """
        self._running.clear()
        self._pending_message_event.set()  # To exit the wait

    async def __aenter__(self):
        self._periodic_task = asyncio.create_task(self.start_periodic_processing())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.stop_periodic_processing()
        await self._periodic_task
