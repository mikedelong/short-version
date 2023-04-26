"""
Abstractive text summarization demo
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
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer

DATA_FOLDER = './data/'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
ENCODING = 'utf-8'
FILE_ROOT_NAME = 'abstractive'
INPUT_FILE = 'plant-manifesto.txt'
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_PATH = Path('./logs/')
MAX_LENGTH = 300
MODE_READ = 'r'
MODEL_NAME = 'google/pegasus-xsum'
SIZES = {
    'google/pegasus-xsum': '2.28Gb'
}

if __name__ == '__main__':
    time_start = now()
    LOG_PATH.mkdir(exist_ok=True)

    set_option('display.max_colwidth', None)  # was -1 and caused a warning
    run_start_time = now().strftime('%Y-%m-%d_%H-%M-%S')
    LOGFILE = '{}/log-{}-{}.log'.format(LOG_PATH, run_start_time, FILE_ROOT_NAME)

    handlers = [FileHandler(LOGFILE), StreamHandler(stdout)]
    # noinspection PyArgumentList
    basicConfig(datefmt=DATE_FORMAT, format=LOG_FORMAT, handlers=handlers, level=INFO, )

    logger = getLogger()
    logger.info('model: %s', MODEL_NAME)
    input_file = DATA_FOLDER + INPUT_FILE
    with open(file = input_file, encoding=ENCODING, mode=MODE_READ) as input_fp:
        text = input_fp.readlines()
    text = ' '.join(text)
    logger.info('text length: %d text start: %s...', len(text), text[:40].replace('\n', ''))

    tokenizer = PegasusTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
    logger.info('loaded pretrained tokenizer.')
    tokens = tokenizer(text=text, truncation=True, padding='longest', return_tensors='pt')
    model = PegasusForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
    logger.info('loaded pretrained model.')
    summary = model.generate(**tokens, max_length=1000,)
    logger.info('got summary.')
    logger.info('summary: %s', tokenizer.decode(token_ids=summary[0], skip_special_tokens=True))

    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))
