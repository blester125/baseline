import sys
from typing import Tuple, TypeVar, Generic, Any, NewType
import numpy as np


if (3, 5) <= sys.version_info < (3, 7):
    from typing import GenericMeta

    old_getitem = GenericMeta.__getitem__  # type: ignore

    def new_getitem(self, params):
        if hasattr(self, 'is_tensor_type'):
            return old_getitem(self, Tuple[params])
        return old_getitem(self, params)

    GenericMeta.__getitem__ = new_getitem  # type: ignore


Dim = TypeVar('Dim')
M = TypeVar('M')
N = TypeVar('N')
K = TypeVar('K')

# It would be cool to have this be ... instead of a word but that causes another mypy error
Broadcast = TypeVar('Broadcast')

# It would be nice to be able to use something like ':' to represent a Dim where we don't care about the dimention


class Tensor(Generic[Dim], np.ndarray):
    is_tensor_type = True
    if sys.version_info >= (3, 7):
        @classmethod
        def __class_getitem__(cls, params):
            return super().__class_getitem__(Tuple[params])


def add(x: Tensor[M, N], y: Tensor[M, N]) -> Tensor[M, N]:  # type: ignore[type-arg]
    """

    :param x:
    :param y:
    :return:
    """
    return x + y


def matmul(x: Tensor[M, N], y: Tensor[N, M]) -> Tensor[M, M]:  # type: ignore[type-arg]
    """

    :param x:
    :param y:
    :return:
    """
    return x @ y


def batched_matmul(x: Tensor[Broadcast, M, N], y: Tensor[Broadcast, N, K]) -> Tensor[Broadcast, M, K]:  # type: ignore[type-arg]
    """

    :param x:
    :param y:
    :return:
    """
    return np.matmul(x, y)


a = np.random.randint(0, 6, size=(5, 6))
b = np.random.randint(0, 6, size=(5, 6))

c = add(a, b)

print(c)
