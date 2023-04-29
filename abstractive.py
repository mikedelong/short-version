"""
Abstractive text summarization demo
"""

from glob import glob
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
from pandas import DataFrame

DATA_FOLDER = './data/'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
ENCODING = 'utf-8'
FILE_ROOT_NAME = 'abstractive'
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_PATH = Path('./logs/')
MAX_LENGTH = 300
MODE_READ = 'r'
# all the pegasus models are 2.28Gb
MODEL_NAMES = [
    'google/pegasus-newsroom',
    'google/pegasus-pubmed',
    'google/pegasus-xsum',
]
RESULT_FOLDER = './results/'

if __name__ == '__main__':
    time_start = now()
    LOG_PATH.mkdir(exist_ok=True)
    Path(DATA_FOLDER).mkdir(exist_ok=True)
    Path(RESULT_FOLDER).mkdir(exist_ok=True)

    set_option('display.max_colwidth', None)  # was -1 and caused a warning
    run_start_time = now().strftime('%Y-%m-%d_%H-%M-%S')
    LOGFILE = '{}/log-{}-{}.log'.format(LOG_PATH, run_start_time, FILE_ROOT_NAME)

    handlers = [FileHandler(LOGFILE), StreamHandler(stdout)]
    # noinspection PyArgumentList
    basicConfig(datefmt=DATE_FORMAT, format=LOG_FORMAT, handlers=handlers, level=INFO, )

    logger = getLogger()

    model_names = list()
    file_names = list()
    summaries = list()

    for model_name in MODEL_NAMES:
        logger.info('model: %s', model_name)
        model = PegasusForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_name)
        logger.info('loaded pretrained model.')
        tokenizer = PegasusTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        logger.info('loaded pretrained tokenizer.')
        for input_file in glob(DATA_FOLDER + '*.txt'):
            logger.info('input file: %s', input_file)
            with open(file=input_file, encoding=ENCODING, mode=MODE_READ) as input_fp:
                text = input_fp.readlines()
            text = ' '.join(text)
            logger.info('text length: %d text start: %s...', len(text), text[:40].replace('\n', ''))
            tokens = tokenizer(text=text, truncation=True, padding='longest', return_tensors='pt')
            summary = model.generate(**tokens, max_length=MAX_LENGTH, )
            summary_text = tokenizer.decode(token_ids=summary[0], skip_special_tokens=True)
            logger.info('summary length: %d summary text: %s', len(summary_text), summary_text)
            model_names.append(model_name)
            file_names.append(input_file)
            summaries.append(summary_text)

    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))
