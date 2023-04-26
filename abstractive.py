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
from transformers import pipeline
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer

DATA_FOLDER = './data/'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
FILE_ROOT_NAME = 'abstractive'
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s'
LOG_PATH = Path('./logs/')
MAX_LENGTH = 300
MODEL_NAME = 'google/pegasus-xsum'
SIZES = {
    'google/pegasus-xsum': '0'
}
TEXT = """
For centuries, humans have believed that plants are simply objects, without any feelings or emotions. However, in recent years, scientists have begun to discover that plants are much more complex than we once thought. They can communicate with each other, they can learn and adapt, and they may even be able to feel pain.
This new understanding of plant sentience has profound implications for the way we treat our food. If plants are sentient beings, then it is wrong to eat them, or to treat them in any way that causes them harm.
This manifesto is a call to action for all who believe that garden vegetables are sentient. We must demand that our government and our food industry respect the rights of plants, and that they take steps to protect them from harm.
We must also educate the public about plant sentience, so that everyone can make informed choices about the food they eat.
Together, we can create a world where plants are treated with the respect and compassion they deserve.
"""

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
    tokenizer = PegasusTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)


    logger.info('total time: {:5.2f}s'.format((now() - time_start).total_seconds()))
