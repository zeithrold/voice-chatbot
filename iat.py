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

'''
XFYun's IAT Client, which is used to convert speech to text.
'''
class IATClient:
    def __init__(
        self,
        app_id: str,
        api_key: str,
        api_secret: str,
        endpoint="wss://ws-api.xfyun.cn/v2/iat",
    ) -> None:
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.endpoint = endpoint
        self.common_args = {"app_id": self.app_id}
        self.business_args = {
            "domain": "iat",
            "language": "zh_cn",
            "accent": "mandarin",
            "vinfo": 1,
            "vad_eos": 10000,
        }

    # To convert ndarray audio data to PCM-style bytes
    # Gradio Audio Module returns a tuple of (sampling_rate, np.ndarray)
    # And the np.ndarray is the audio data, which is range from -32768 to 32767 matching PCM range.
    def encode_pcm(self, source: np.ndarray):
        return source.astype(np.int16).tobytes()

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

    def prepare_data(self, audio: bytes, chunk_size=1280, sampling_rate=16000):
        status = STATUS_FIRST_FRAME
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if i + chunk_size >= len(audio):
                status = STATUS_LAST_FRAME
            data = {
                "status": status,
                "format": f"audio/L16;rate={sampling_rate}",
                "audio": base64.b64encode(chunk).decode("utf-8"),
                "encoding": "raw",
            }
            payload = {"data": data}
            if status == STATUS_FIRST_FRAME:
                payload["common"] = self.common_args
                payload["business"] = self.business_args
            yield payload
            status = STATUS_CONTINUE_FRAME

    async def dictate(self, audio: tuple[int, np.ndarray], interval=0.4):
        url = self.create_url()
        sampling_rate, source = audio
        pcm = self.encode_pcm(source)
        async with websockets.client.connect(url) as ws:
            for payload in self.prepare_data(pcm, sampling_rate=sampling_rate):
                await ws.send(json.dumps(payload))
                time.sleep(interval)
            try:
                async for message in ws:
                    yield message
            except ConnectionClosedError as e:
                print(f"Connection closed: {e.code} {e.reason}")
