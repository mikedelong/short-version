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
from transformers import pipeline


def read_text(filename: str) -> str:
    with open(file=filename, encoding='utf-8', mode='r') as input_fp:
        lines = input_fp.readlines()
    lines = ' '.join(lines).replace('\n', ' ')
    return ' '.join(lines.split())


DATA_FOLDER = './data/'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
INPUT_FILE = [
    'communist-manifesto.txt',
    'worldwatch2003.txt',
    ][0]
INPUT_LENGTH = [
    2 * 1024 // 6,
    3 * 1024 // 8,
    2 * 1024 // 3,
    7 * 1024 // 10,
    3 * 1024 // 4,
    4 * 1024 // 5,
][3]
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_PATH = Path('./logs/')
MAX_LENGTH = [
    448,
    224,
    112,
    56,
    30,
][0]
MODEL = [
    'google/pegasus-cnn_dailymail',
    'google/pegasus-large',
    'google/pegasus-pubmed',
    'google/pegasus-xsum',
    'lidiya/bart-large-xsum-samsum',
    'philschmid/bart-large-cnn-samsum',
    'sshleifer/distilbart-cnn-12-6',
    't5-base',
][6]
TASK = 'summarization'

if __name__ == '__main__':
    time_start = now()
    LOG_PATH.mkdir(exist_ok=True)

    set_option('display.max_colwidth', None)  # was -1 and caused a warning
    run_start_time = now().strftime('%Y-%m-%d_%H-%M-%S')
    file_root_name = 'summarize'
    LOGFILE = '{}/log-{}-{}.log'.format(LOG_PATH, run_start_time, file_root_name)

    handlers = [FileHandler(LOGFILE), StreamHandler(stdout)]
    # noinspection PyArgumentList
    basicConfig(datefmt=DATE_FORMAT, format=LOG_FORMAT, handlers=handlers, level=INFO, )

    logger = getLogger()
    logger.info('started')

    input_file = DATA_FOLDER + INPUT_FILE
    logger.info('reading input data from %s', input_file)
    data = read_text(filename=input_file)
    logger.info(data[:100] + '...')
    logger.info('input data has %d tokens', len(data.split()))

    processor = pipeline(max_length=MAX_LENGTH, model=MODEL, task=TASK)
    for start in range(0, len(data.split()), INPUT_LENGTH):
        input_slice = ' '.join(data.split()[start:start+INPUT_LENGTH])
        logger.info('segment length: %d', len(input_slice.split()))
        result = processor(input_slice)
        summary_text =  result[0]['summary_text']
        logger.info('%d tokens: %s', len(summary_text.split()), summary_text)

    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))
