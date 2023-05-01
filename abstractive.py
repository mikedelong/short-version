"""
Abstractive text summarization demo
"""

from glob import glob
from logging import FileHandler
from logging import INFO
from logging import StreamHandler
from logging import basicConfig
from logging import getLogger
from os.path import exists
from pathlib import Path
from sys import stdout

from arrow import now
from pandas import DataFrame
from pandas import read_csv
from pandas import set_option
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer

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
OUTPUT_FILE_NAMES = list()
OUTPUT_MODEL_NAMES = list()
OUTPUT_SUMMARIES = list()
RESULT_FOLDER = './results/'
RESULT_FILE = 'abstractive.csv'

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

    prior_filename = RESULT_FOLDER + RESULT_FILE
    if exists(path=prior_filename):
        logger.info('reading prior results from %s', prior_filename)
        prior_df = read_csv(filepath_or_buffer=prior_filename)
        logger.info('read %d results from %s', len(prior_df), prior_filename)
    else:
        prior_df = DataFrame(columns=['model', 'file', 'summary', ])

    # TODO only generate cases we haven't already generated
    input_files = list(glob(DATA_FOLDER + '*.txt'))
    for model_name in MODEL_NAMES:
        logger.info('model: %s', model_name)
        model = PegasusForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_name)
        logger.info('loaded pretrained model.')
        tokenizer = PegasusTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        logger.info('loaded pretrained tokenizer.')
        for input_file in input_files:
            logger.info('input file: %s', input_file)
            with open(file=input_file, encoding=ENCODING, mode=MODE_READ) as input_fp:
                text = input_fp.readlines()
            text = ' '.join(text)
            logger.info('text length: %d text start: %s...', len(text), text[:40].replace('\n', ''))
            tokens = tokenizer(text=text, truncation=True, padding='longest', return_tensors='pt')
            summary = model.generate(**tokens, max_length=MAX_LENGTH, )
            summary_text = tokenizer.decode(token_ids=summary[0], skip_special_tokens=True)
            logger.info('summary length: %d summary text: %s', len(summary_text), summary_text)
            OUTPUT_MODEL_NAMES.append(model_name)
            OUTPUT_FILE_NAMES.append(input_file)
            OUTPUT_SUMMARIES.append(summary_text)
            result_filename = RESULT_FOLDER + RESULT_FILE
            logger.info('writing: %s', result_filename)
            # TODO concat the results and write the updated results
            DataFrame(
                data={'model': OUTPUT_MODEL_NAMES, 'file': OUTPUT_FILE_NAMES, 'summary': OUTPUT_SUMMARIES}).to_csv(
                index=False, path_or_buf=result_filename, )

    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))
