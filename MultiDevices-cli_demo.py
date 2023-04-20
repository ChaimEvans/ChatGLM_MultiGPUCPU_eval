import os
import platform
import signal
import MultiDevices
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("D:\chatGLM-6B-model\chatGLM-6B", trust_remote_code=True)
model = AutoModel.from_pretrained("D:\chatGLM-6B-model\chatGLM-6B", trust_remote_code=True).half()
MultiDevices.GPU_precision = 'int8'
MultiDevices.embeddings = 'cpu'
#默认为使用6G显存+16G内存，修改请参阅README.md
MultiDevices.layers={
                        'cuda:0': '1-20',
                        'cpu':'21-28'
                    }
MultiDevices.final_layernorm = 'cpu'
model = MultiDevices.ConfigMultiDevices(model)
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()
