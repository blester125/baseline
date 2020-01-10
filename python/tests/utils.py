import random
import string
from itertools import chain


CHARS = list(chain(string.ascii_letters, string.digits))


def rand_str(length=None, min_=3, max_=10):
    length = random.randint(min_, max_) if length is None else length
    return ''.join([random.choice(CHARS) for _ in range(length)])
