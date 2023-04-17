from gradio import Interface

def greet(name: str):
    return 'Hello {}.'.format(name)

Interface(fn=greet, inputs='text', outputs='text').launch()
