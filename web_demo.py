Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pharell AI - interface de chat web interactive basée sur gradio."""
import os
from argparse import ArgumentParser

import gradio as gr
import mdtex2html

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


DEFAULT_CKPT_PATH = 'Qwen/Qwen-7B-Chat'


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Thème personnalisé "Pharell AI" : dégradé violet / rose / orange
# ----------------------------------------------------------------------------
pharell_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="linear-gradient(135deg, #1e1033 0%, #3b1568 40%, #7b2ff7 70%, #ff6ec7 100%)",
    body_background_fill_dark="linear-gradient(135deg, #0f0821 0%, #241046 40%, #4b1a8f 70%, #b3428f 100%)",
    button_primary_background_fill="linear-gradient(90deg, #ff6ec7 0%, #7b2ff7 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #ff8ad4 0%, #9452ff 100%)",
    button_primary_text_color="white",
    block_background_fill="rgba(255,255,255,0.08)",
    block_border_color="#a259ff",
    block_title_text_color="#ffb3ec",
    block_label_text_color="#ffb3ec",
)

CUSTOM_CSS = """
#chatbot { background: rgba(20, 10, 40, 0.55); border-radius: 18px; border: 1px solid #a259ff55; }
.gradio-container { font-family: 'Poppins', 'Segoe UI', sans-serif; }
h1, h2, h3 { background: linear-gradient(90deg, #ff6ec7, #7b2ff7, #58c7ff);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
footer { visibility: hidden; }
"""


def _launch_demo(args, model, tokenizer, config):

    def predict(_query, _chatbot, _task_history):
        print(f"User: {_parse_text(_query)}")
        _chatbot.append((_parse_text(_query), ""))
        full_response = ""

        for response in model.chat_stream(tokenizer, _query, history=_task_history, generation_config=config):
            _chatbot[-1] = (_parse_text(_query), _parse_text(response))

            yield _chatbot
            full_response = _parse_text(response)

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Pharell AI: {_parse_text(full_response)}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks(theme=pharell_theme, css=CUSTOM_CSS) as demo:
        gr.Markdown("""\
<p align="center"><img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg" style="height: 80px; border-radius: 50%; box-shadow: 0 0 25px #ff6ec7;"/><p>""")
        gr.Markdown("""<center><font size=8>🌈 Pharell AI 🌈</center>""")
        gr.Markdown(
            """\
<center><font size=3>Bienvenue sur <b>Pharell AI</b>, votre assistant conversationnel intelligent. \
(欢迎使用 Pharell AI 聊天机器人。)</center>""")

        chatbot = gr.Chatbot(label='Pharell AI', elem_id="chatbot", elem_classes="control-height")
        query = gr.Textbox(lines=2, label='💬 Votre message')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Effacer l'historique", variant="secondary")
            submit_btn = gr.Button("🚀 Envoyer", variant="primary")
            regen_btn = gr.Button("🤔 Régénérer", variant="secondary")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

        gr.Markdown("""\
<font size=2>Note : cette démo Pharell AI est fournie à titre indicatif. \
Nous encourageons vivement les utilisateurs à ne pas générer ou diffuser sciemment du contenu nuisible, \
incluant les discours de haine, la violence, la pornographie, la tromperie, etc.""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer, config = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer, config)


if __name__ == '__main__':
    main()
