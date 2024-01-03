import gradio as gr
import Backend
import time

chain  = Backend.Malware_context
def Malware(query):
    return chain.run(query)

with gr.Blocks() as demo:
    gr.Markdown('''<h1><center>Chat CSEC: Malware Expert </center></h1>''')
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = Malware(history[-1][0])
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.0005)
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)
    
demo.queue()
demo.launch()