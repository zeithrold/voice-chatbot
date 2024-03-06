from urllib.parse import urlparse, urlencode
from wsgiref.handlers import format_date_time
from datetime import datetime
import hmac
import hashlib
import base64
import numpy as np
import json
import websockets.client
from websockets.exceptions import ConnectionClosedError
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
        self.business_args = {
            "aue": "raw",
            "auf": "audio/L16;rate=16000",
            "vcn": "xiaoyan",
            "tte": "utf8",
        }

    def prepare_data(self, text: str):
        return {
            "common": self.common_args,
            "business": self.business_args,
            "data": {
                "status": 2,
                "text": str(base64.b64encode(text.encode("utf-8")), "UTF8"),
            },
        }

    def create_url(self):
        parse_result = urlparse(self.endpoint)
        host = parse_result.hostname
        # RFC1123 Timestamp
        date = format_date_time(time.mktime(datetime.now().timetuple()))
        path = parse_result.path

        sign_raw_str = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        sign_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            sign_raw_str.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        auth_origin = 'api_key="%s", algorithm="%s", headers="%s", signature="%s"' % (
            self.api_key,
            "hmac-sha256",
            "host date request-line",
            sign_sha,
        )
        auth = base64.b64encode(auth_origin.encode("utf-8")).decode("utf-8")
        params = {
            "authorization": auth,
            "date": date,
            "host": host,
        }
        url = f"{self.endpoint}?{urlencode(params)}"
        return url

    def parse_result(self, result: bytes) -> np.ndarray:
        return np.frombuffer(result, dtype=np.int16)

    async def generate(self, text: str):
        url = self.create_url()
        data = self.prepare_data(text)
        result = bytearray()
        async with websockets.client.connect(url) as ws:
            await ws.send(json.dumps(data))
            while True:
                try:
                    message = await ws.recv()
                    message = json.loads(message)
                    audio = message["data"]["audio"]
                    audio = base64.b64decode(audio)
                    status = message["data"]["status"]
                    result += audio
                    if status == STATUS_LAST_FRAME:
                        break
                except ConnectionClosedError:
                    break
        return self.parse_result(bytes(result))
