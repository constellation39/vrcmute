import asyncio
import os

from src.logger import logger
from src.vrcclient import VRCClient
from src.vrcmute import VRCMute


async def _main():
    async with VRCClient(
        address="127.0.0.1",
        port=9000,
        interval=1.333,
        period=0.1,
    ) as osc_client:
        wave_filename = "./hello_world_male2.wav"
        wave_filename = None
        try:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

            async with VRCMute(
                vrc_client=osc_client,
                api_key=api_key,
            ) as vrc_mute:
                if wave_filename:
                    await vrc_mute.start(wave_filename)
                else:
                    vrc_mute.setup_microphone()
                    await vrc_mute.start()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, exiting.")


def main():
    logger.info("Starting VRChat Mute...")
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, exiting.")


if __name__ == "__main__":
    main()
