import os, base64, io
from together import Together
from PIL import Image
import gradio as gr
from openai import OpenAI
client = OpenAI(api_key=os.getenv("TOGETHER_API_KEY"),base_url="https://api.together.xyz/v1",)

SYSTEM_PROMPT = (
    "You are an expert botanist. Identify the plant in the image, "
    "then give cultivation tips, toxicity/safety info and a fun fact."
)

def infer(img, extra_prompt):
    if img is None:
        # 如果没有图片输入，就礼貌地返回一个提示，而不是让程序崩溃。
        return "请先上传一张图片再提交！"
    # 将图片转 base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    resp = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        stream=False,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": extra_prompt or ""}
                ]
            }
        ],
    )
    return resp.choices[0].message.content

demo = gr.Interface(
    fn=infer,
    inputs=[gr.Image(type="pil"), gr.Textbox(lines=2, label="额外提问（可选）")],
    outputs=gr.Markdown(),
    title="🌿 Welcome to Protamind",
    description="请上传植物照片"
)
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get('PORT', 7860)))