import gradio as gr
import os
from loguru import logger
from zhipuai import ZhipuAI
from zhipuai.api_resource.chat.chat import Chat
import yaml
import json
from iat import IATClient
from tts import TTSClient
import numpy as np
from scipy.signal import resample

logger.debug("Loading config")
config_env = os.environ.get("CONFIG", "")
if config_env:
    logger.debug("Using environment variable for config")
    config = json.loads(config_env)
else:
    logger.debug("Reading config from file")
    with open("config.yaml", "r") as f:
        try:
            config = yaml.safe_load(f)["config"]
        except yaml.YAMLError as e:
            logger.error(e)
            raise e

zhipuai_config = config["zhipuai"]
xfyun_config = config["xfyun"]

zhipuai = ZhipuAI(api_key=zhipuai_config["apikey"])
iat = IATClient(
    xfyun_config["iat"]["appid"],
    xfyun_config["iat"]["apikey"],
    xfyun_config["iat"]["apisecret"],
)
tts = TTSClient(
    xfyun_config["tts"]["appid"],
    xfyun_config["tts"]["apikey"],
    xfyun_config["tts"]["apisecret"],
)


def build_zhipuai_history(history: list[list[str]]):
    result = [{"role": "system", "content": config["zhipuai"]["prompt"]}]
    for history_element in history:
        user_message, assistant_message = history_element
        if user_message != None:
            result += [{"role": "user", "content": user_message}]
        if assistant_message != None:
            result += [{"role": "assistant", "content": assistant_message}]
    return result


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def bot(history):
    zhipuai_history = build_zhipuai_history(history)
    res = zhipuai.chat.completions.create(
        model="glm-4", messages=zhipuai_history, stream=True
    )
    history[-1][1] = ""
    for chunk in res:
        history[-1][1] += chunk.choices[0].delta.content
        yield history


async def generate_text(audio: tuple[int, np.ndarray]):
    logger.debug(f"Generating text from audio")
    logger.debug(f"Sampling rate: {audio[0]}, resampling to 16000")
    audio = (16000, resample(audio[1], 16000))
    result_list = []
    async for result in iat.dictate(audio):
        logger.debug(f"Result: {result}")
        result_list.append(result)
    return "".join(result_list)


async def generate_audio(history: list[list[str]]):
    logger.debug(f"Generating audio from text")
    text = history[-1][-1]
    result = await tts.generate(text)
    return result

with gr.Blocks() as demo:
    title = gr.Markdown("# 老王元宇宙受害者")

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )
        submit_button = gr.Button(value="提交", variant="primary")

    with gr.Row():
        with gr.Column():
            user_title = gr.Markdown("## 用户语音识别")
            user_audio = gr.Audio(type="numpy", sources=['microphone'])
            user_audio_submit = gr.Button(value="上传用户语音并转换", variant="primary")
        with gr.Column():
            user_title = gr.Markdown("## 机器人语音合成")
            bot_audio = gr.Audio()
            bot_audio_submit = gr.Button(value="将机器人最后一个回复转换为语音", variant="primary")

    user_audio_submit.click(generate_text, [user_audio], outputs=txt)
    bot_audio_submit.click(generate_audio, [chatbot], outputs=bot_audio)

    txt_msg = submit_button.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )

    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

demo.launch()
