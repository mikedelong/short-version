from gradio import Interface

Interface(fn=lambda x: 'Hello {}.'.format(x), inputs='text', outputs='text').launch()
