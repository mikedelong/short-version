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


def text_generation(input_text, seed, arg_model, arg_tokenizer):
    input_ids = arg_tokenizer(input_text, return_tensors='pt').input_ids
    manual_seed(seed)  # Max value: 18446744073709551615
    outputs = arg_model.generate(input_ids, do_sample=True, max_length=100)
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    input_text = 'The quick brown fox jumped over the lazy dog.'
    seed = 1
    result = text_generation(input_text=input_text, seed=seed, arg_model=model, arg_tokenizer=tokenizer)
    logger.info(result[0])

    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))
