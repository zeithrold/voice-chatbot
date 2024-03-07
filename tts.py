from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
from datetime import datetime
from loguru import logger
import hmac
import hashlib
import base64
import numpy as np
import json
import websockets.client
from websockets.exceptions import ConnectionClosedError, InvalidStatusCode
import time

STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2


class TTSClient:
    def __init__(
        self,
        app_id: str,
        api_key: str,
        api_secret: str,
        endpoint="wss://ws-api.xfyun.cn/v2/tts",
    ) -> None:
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.endpoint = endpoint
        self.common_args = {"app_id": self.app_id}

    def prepare_data(self, text: str, sampling_rate=16000):
        business_args = {
            "aue": "raw",
            "auf": f"audio/L16;rate={sampling_rate}",
            "vcn": "xiaoyan",
            "tte": "utf8",
        }
        result = {
            "common": self.common_args,
            "business": business_args,
            "data": {
                "status": 2,
                "text": str(base64.b64encode(text.encode("utf-8")), "UTF8"),
            },
        }
        logger.debug(f"Data: {result}")
        return result

    def create_url(self):
        parse_result = urlparse(self.endpoint)
        host = parse_result.hostname
        # RFC1123 Timestamp
        date = format_date_time(time.mktime(datetime.now().timetuple()))
        path = parse_result.path

        sign_raw_str = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        logger.debug(f"Sign raw string: {sign_raw_str}")
        sign_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            sign_raw_str.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign_sha = base64.b64encode(sign_sha).decode("utf-8")
        auth_raw_str = 'api_key="%s", algorithm="%s", headers="%s", signature="%s"' % (
            self.api_key,
            "hmac-sha256",
            "host date request-line",
            sign_sha,
        )
        logger.debug(f"Authorization: {auth_raw_str}")
        auth = base64.b64encode(auth_raw_str.encode("utf-8")).decode("utf-8")
        params = {
            "authorization": auth,
            "date": date,
            "host": host,
        }
        url = f"{self.endpoint}?{urlencode(params)}"
        logger.debug(f"URL: {url}")
        return url

    def parse_result(self, result: bytes) -> np.ndarray:
        return np.frombuffer(result, dtype=np.int16)

    async def generate(self, text: str, sampling_rate=16000):
        logger.debug("Generate URL")
        url = self.create_url()
        logger.debug("Preparing Data")
        data = self.prepare_data(text, sampling_rate)
        result = bytearray()
        try:
            async with websockets.client.connect(url) as ws:
                logger.debug("Sending Data")
                await ws.send(json.dumps(data))
                while True:
                    try:
                        message = await ws.recv()
                        message = json.loads(message)
                        logger.debug(f"Received message: {message}")
                        audio = message["data"]["audio"]
                        logger.debug(f"Received audio length: {len(audio)}")
                        audio = base64.b64decode(audio)
                        status = message["data"]["status"]
                        result += audio
                        if status == STATUS_LAST_FRAME:
                            break
                    except ConnectionClosedError:
                        break
        except InvalidStatusCode as e:
            logger.error(f"Error: {e}")
            raise e
        logger.success("Audio generation finished")
        return sampling_rate, self.parse_result(bytes(result))


__all__ = ["TTSClient"]
