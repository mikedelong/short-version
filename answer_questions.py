"""
Answer simple questions
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
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

DATA_FOLDER = './data/'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
FILE_ROOT_NAME = 'answer_questions'
INPUT_TEXT = 'Everyone is going to the mall but Alice. My name is Bob; where am I going?'
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_PATH = Path('./logs/')
MAX_LENGTH = 300
# google/flan-t5-small is 308M
MODEL_NAME = 'google/flan-t5-small'
SIZES = {
    'google/flan-t5-small': '308M'
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
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
    logger.info('input: %s', INPUT_TEXT)
    token_ids = tokenizer(text=INPUT_TEXT, max_length=MAX_LENGTH, return_tensors='pt', truncation=True, ).input_ids
    output_ids = model.generate(token_ids, max_length=MAX_LENGTH, )
    output = tokenizer.decode(token_ids=output_ids[0], ).replace('<pad> ', '').replace('</s>', '')
    logger.info('output: %s', output)

    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))
