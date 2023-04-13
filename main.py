"""
Summarize text?
"""

from logging import FileHandler
from logging import INFO
from logging import StreamHandler
from logging import basicConfig
from logging import getLogger
from pathlib import Path
from sys import stdout

from arrow import now
from pandas import set_option
from torch import manual_seed
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def text_generation(arg_text, arg_model, arg_tokenizer):
    outputs = arg_model.generate(
        arg_tokenizer(arg_text, return_tensors='pt').input_ids,
        do_sample=True,
        max_length=100,
        pad_token_id=arg_tokenizer.eos_token_id, )
    return arg_tokenizer.batch_decode(outputs, skip_special_tokens=True)


DATA_FOLDER = './data/'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_PATH = Path('./logs/')
MODEL = [
    'gpt2',
][0]

if __name__ == '__main__':
    time_start = now()
    LOG_PATH.mkdir(exist_ok=True)

    set_option('display.max_colwidth', None)  # was -1 and caused a warning
    run_start_time = now().strftime('%Y-%m-%d_%H-%M-%S')
    file_root_name = 'main'
    LOGFILE = '{}/log-{}-{}.log'.format(LOG_PATH, run_start_time, file_root_name)

    handlers = [FileHandler(LOGFILE), StreamHandler(stdout)]
    # noinspection PyArgumentList
    basicConfig(datefmt=DATE_FORMAT, format=LOG_FORMAT, handlers=handlers, level=INFO, )

    logger = getLogger()
    logger.info('started')

    input_text = 'The quick brown fox jumped over the lazy dog.'
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    for seed in range(10):
        manual_seed(seed=seed)
        result = text_generation(arg_text=input_text, arg_model=model, arg_tokenizer=tokenizer)[0]
        result = result.replace('\n', ' ')
        result = result.replace(input_text, '')
        result = ' '.join(result.split())
        logger.info('seed: %d result: %s', seed, result)

    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))
