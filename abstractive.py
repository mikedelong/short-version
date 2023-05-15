"""
Abstractive text summarization demo
All the pegasus models are 2.28Gb
"""

from glob import glob
from json import load
from logging import FileHandler
from logging import INFO
from logging import Logger
from logging import StreamHandler
from logging import basicConfig
from logging import getLogger
from os.path import exists
from pathlib import Path
from sys import stdout

from arrow import now
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import set_option
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer


def configure_logging() -> Logger:
    set_option('display.max_colwidth', None)  # was -1 and caused a warning
    logfile = '{}/log-{}-{}.log'.format(LOG_PATH, now().strftime('%Y-%m-%d_%H-%M-%S'), FILE_ROOT_NAME)

    handlers = [FileHandler(logfile), StreamHandler(stdout)]
    # noinspection PyArgumentList
    basicConfig(datefmt=DATE_FORMAT, format=LOG_FORMAT, handlers=handlers, level=INFO, )
    return getLogger()


def get_prior_data(filename: str) -> DataFrame:
    if exists(path=filename):
        logger.info('reading prior results from %s', filename)
        result_df = read_csv(filepath_or_buffer=filename)
        logger.info('read %d results from %s', len(result_df), filename)
    else:
        result_df = DataFrame(columns=['model', 'file', 'summary', ])
    return result_df


def get_settings() -> dict:
    with open(encoding='utf-8', file=SETTINGS_FILE, mode=MODE_READ, ) as settings_fp:
        result = load(fp=settings_fp)

    return result


def main():
    time_start = now()
    LOG_PATH.mkdir(exist_ok=True)
    DATA_FOLDER.mkdir(exist_ok=True)
    RESULT_FOLDER.mkdir(exist_ok=True)

    settings = get_settings()
    model_names = settings['MODEL_NAMES'] if 'MODEL_NAMES' in settings.keys() else None
    if not model_names:
        logger.error(msg='Model name is missing from settings.')
    prior_filename = RESULT_FOLDER.name + RESULT_FILE
    prior_df = get_prior_data(filename=prior_filename)

    input_files = list(glob(DATA_FOLDER.name + '*.txt'))
    for model_name in model_names:
        model_df = prior_df[prior_df['model'] == model_name]
        not_done_files = {name for name in input_files if name not in model_df['file'].values}
        if len(not_done_files):
            logger.info('model: %s', model_name)
            model = PegasusForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_name)
            logger.info('loaded pretrained model.')
            tokenizer = PegasusTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
            logger.info('loaded pretrained tokenizer.')
            for input_file in not_done_files:
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
                result_filename = RESULT_FOLDER.name + RESULT_FILE
                result_df = DataFrame(
                    data={'model': OUTPUT_MODEL_NAMES, 'file': OUTPUT_FILE_NAMES, 'summary': OUTPUT_SUMMARIES})
                result_df = concat([prior_df, result_df])
                logger.info('writing: %d rows to %s', len(result_df), result_filename)
                result_df.to_csv(index=False, path_or_buf=result_filename, )

    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))


DATA_FOLDER = Path('./data/')
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
ENCODING = 'utf-8'
FILE_ROOT_NAME = 'abstractive'
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_PATH = Path('./logs/')
MAX_LENGTH = 300
MODE_READ = 'r'
OUTPUT_FILE_NAMES = list()
OUTPUT_MODEL_NAMES = list()
OUTPUT_SUMMARIES = list()
RESULT_FOLDER = Path('./results/')
RESULT_FILE = 'abstractive.csv'
SETTINGS_FILE = './abstractive.json'

logger = configure_logging()

if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logger.error(exc_info=True, msg='error:')
