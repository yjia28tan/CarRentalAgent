import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.gateway import LLMError, create_llm_gateway
from src.utils.config import get_settings


async def main() -> None:
    s = get_settings()
    print("endpoint host:", urlparse(s.llm_endpoint).hostname, "| api_version:", s.llm_api_version,
          "| deployment:", s.llm_model)
    try:
        await create_llm_gateway().complete("Say hi.", [{"role": "user", "content": "hi"}], [])
        print("OK")
    except LLMError as exc:
        cause = exc.__cause__
        print("status:", getattr(cause, "status_code", None))
        print("message:", str(getattr(cause, "message", cause))[:400])


asyncio.run(main())
