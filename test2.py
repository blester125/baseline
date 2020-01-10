from typing import TypeVar, Generic, Tuple


T = TypeVar('T', bound=Shape)


class Tensor(Generic[T]): pass

M = TypeVar('M', bound=Shape)
N = TypeVar('N', bound=Shape)
Dim = TypeVar('Dim', bound=Shape)
Shape = Tuple[Dim, ...]

print(Tensor[Shape[Dim]])

def add(x: Tensor[Shape[M, N]], y: Tensor[Shape[M, N]]) -> Tensor[Shape[M, N]]:
    return x + y
