import asyncio
import re
import time
import wave
from dataclasses import dataclass
from typing import Any

import dashscope
import numpy as np
import sounddevice as sd
from dashscope.audio.asr import Recognition
from dashscope.audio.asr import RecognitionCallback
from dashscope.audio.asr import RecognitionResult

from src.logger import logger


# Configuration
@dataclass
class Config:
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    INTERVAL: float = 0.2
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5


class AudioProcessor:
    """Handles audio processing and resampling operations."""

    @staticmethod
    def read_wave_file(filename: str) -> tuple[np.ndarray, int]:
        with wave.open(filename) as f:
            assert f.getnchannels() == 1, "Only mono audio is supported"
            assert f.getsampwidth() == 2, "Only 16-bit audio is supported"
            num_samples = f.getnframes()
            samples = f.readframes(num_samples)
            return np.frombuffer(samples, dtype=np.int16), f.getframerate()

    @staticmethod
    async def resample_audio(
        samples: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        if orig_sr == target_sr:
            return samples
        import librosa

        return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)


class TextCleaner:
    """Handles text cleaning and formatting operations."""

    FILLER_WORDS = {
        "yeah",
        "oh",
        "yes",
        "the",
        "and",
        "um",
        "uh",
        "er",
        "ah",
        "like",
        "so",
    }
    PUNCTUATION_PATTERN = r"[.,。，!！?？:：;；\s]"

    @classmethod
    def clean_content(cls, content: str) -> str:
        content = content.lower()
        for word in cls.FILLER_WORDS:
            content = re.sub(r"\b" + word + r"\b", "", content, flags=re.IGNORECASE)
        return re.sub(cls.PUNCTUATION_PATTERN, "", content)


class SpeechRecognitionHandler:
    """Manages speech recognition operations using DashScope."""

    def __init__(self, api_key: str, config: Config, callback: RecognitionCallback):
        self.config = config
        dashscope.api_key = api_key
        self.recognition = self._initialize_recognition(callback)

    def _initialize_recognition(self, callback: RecognitionCallback) -> Recognition:
        return Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=self.config.SAMPLE_RATE,
            callback=callback,
            speaker_count=1,
            # language_hints=["zh", "en"],
        )

    async def start(self) -> None:
        try:
            self.recognition.start()
        except dashscope.common.error.DashScopeException as e:
            raise RuntimeError(f"Failed to start recognition: {e}")

    def stop(self) -> None:
        if self.recognition and self.recognition._running:
            self.recognition.stop()

    async def reconnect(self):
        logger.info("Attempting to reconnect to speech recognition service...")
        for attempt in range(self.config.MAX_RETRIES):
            try:
                await self.start()
                logger.info("Successfully reconnected to speech recognition service.")
                return True
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(self.config.RETRY_DELAY)
        logger.error("Failed to reconnect after maximum attempts.")
        return False

    async def process_audio(self, audio_data: np.ndarray) -> None:
        if not self.recognition or not self.recognition._running:
            if await self.reconnect():
                self.recognition.send_audio_frame(audio_data)
            else:
                logger.warning(
                    "Unable to process audio: Recognition service is not available.",
                )
        else:
            self.recognition.send_audio_frame(audio_data)


class VRCMute:
    """Main system class that coordinates all components."""

    def __init__(self, api_key: str, vrc_client: Any, config: Config = Config()):
        self.config = config
        self.vrc_client = vrc_client
        self.callback = CustomRecognitionCallback(vrc_client)
        self.recognition_handler = SpeechRecognitionHandler(
            api_key,
            config,
            self.callback,
        )
        self.is_running = False
        self.audio_queue = asyncio.Queue()

    async def start(self, wave_filename: str | None = None) -> None:
        self.is_running = True
        self.callback.start_time = time.time()

        if wave_filename:
            await self._process_file(wave_filename)
        else:
            await self._process_microphone()

    async def _process_file(self, wave_filename: str) -> None:
        try:
            samples, sample_rate = AudioProcessor.read_wave_file(wave_filename)
            if sample_rate != self.config.SAMPLE_RATE:
                samples = await AudioProcessor.resample_audio(
                    samples,
                    sample_rate,
                    self.config.SAMPLE_RATE,
                )

            chunk_size = int(self.config.SAMPLE_RATE * self.config.INTERVAL)
            for i in range(0, len(samples), chunk_size):
                if not self.is_running:
                    break
                chunk = samples[i : i + chunk_size]
                await self.recognition_handler.process_audio(chunk)
                await asyncio.sleep(self.config.INTERVAL)

        except Exception as e:
            await self._handle_error(f"File processing error: {e}")

    def setup_microphone(self) -> None:
        try:
            self._print_microphone_info()
            sd.default.samplerate = self.config.SAMPLE_RATE
            sd.default.channels = self.config.CHANNELS
        except sd.PortAudioError as error:
            logger.warning(f"Unable to access microphone: {error}")

    def _print_microphone_info(self) -> None:
        devices = sd.query_devices()
        default_input = sd.default.device[0]  # 获取默认输入设备的索引

        if default_input is not None:
            device_info = devices[default_input]
            logger.info(f"Used Microphone: {device_info['name']}")
            logger.info(f"Channels: {device_info['max_input_channels']}")
            logger.info(f"Sample Rate: {device_info['default_samplerate']} Hz")
        else:
            logger.warning("No default input device found.")

    async def _process_microphone(self) -> None:
        def callback(
            indata: np.ndarray,
            frames: int,
            time: float,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                logger.info(f"Warning: {status}")
            if self.is_running:
                asyncio.run(self.recognition_handler.process_audio(indata.copy()))

        try:    
            with sd.InputStream(
                callback=callback,
                channels=self.config.CHANNELS,
                samplerate=self.config.SAMPLE_RATE,
                blocksize=int(self.config.SAMPLE_RATE * self.config.INTERVAL),
                dtype=np.int16,
            ):
                while self.is_running:
                    await asyncio.sleep(1)

        except Exception as e:
            await self._handle_error(f"Microphone processing error: {e}")

    async def _process_chunk(self, chunk_data: np.ndarray) -> None:
        await self.recognition_handler.process_audio(chunk_data)

    async def _handle_error(self, error_message: str) -> None:
        logger.info(f"Error: {error_message}")
        for _ in range(self.config.MAX_RETRIES):
            try:
                await self.recognition_handler.start()
                return
            except Exception as e:
                logger.info(f"Retry failed: {e}")
                await asyncio.sleep(self.config.RETRY_DELAY)

        self.stop()

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self.recognition_handler.stop()

    async def _worker(self) -> None:
        while self.is_running:
            try:
                chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                await self._process_chunk(chunk)
                self.audio_queue.task_done()
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in worker: {e}")
                await asyncio.sleep(1)

    async def __aenter__(self):
        await self.recognition_handler.start()
        self.worker_task = asyncio.create_task(self._worker())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("exit")
        self.stop()
        await self.audio_queue.join()
        self.worker_task.cancel()
        try:
            await self.worker_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Error while waiting for worker task: {e}")


class CustomRecognitionCallback(RecognitionCallback):
    """Handles recognition results and communicates with VRC client."""

    def __init__(self, osc_client: Any):
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.osc_client = osc_client
        self.loop = asyncio.get_event_loop()

    def on_complete(self) -> None:
        self.end_time = time.time()
        duration = self.end_time - self.start_time if self.start_time else 0
        logger.info(f"Recognition completed. Duration: {duration:.2f}s")

    def on_error(self, result: RecognitionResult) -> None:
        logger.info(f"Recognition error: {result}")

    def on_event(self, result: RecognitionResult) -> None:
        asyncio.run_coroutine_threadsafe(self._async_on_event(result), self.loop)

    async def _async_on_event(self, result: RecognitionResult) -> None:
        content = result.output.sentence["text"].strip()
        if not content:
            return

        temp_content = TextCleaner.clean_content(content)
        if not temp_content:
            return

        logger.info(f"Recognized text: {content}")
        await self.osc_client.send_chatbox(
            [content, True, content.endswith((".", "。", "？"))],
        )
