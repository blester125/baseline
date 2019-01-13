# from seq2seq_vMF import LogCmK
from scipy.special import iv, ive
import numpy as np
import torch

m = 300
k = np.linalg.norm(np.random.rand(2, m), axis=1)

def log_cmk(m, k):
    log_cmk = np.log(k ** (m/2-1) / ((2*np.pi)**(m/2)*ive(m/2-1, k)))
    return log_cmk

r = log_cmk(m, k)

def approx_log_cmk(m, k):
    v = m / 2
    approx_log_cmk = np.sqrt(np.square(v + 1) + np.square(k)) - (v - 1) * np.log(v - 1 + np.sqrt(np.square(v + 1) + np.square(k)))
    return approx_log_cmk

a = approx_log_cmk(m, k)

c = r - a

k = np.array([np.linalg.norm(np.random.rand(m)), np.linalg.norm(np.random.rand(m))])

r2 = log_cmk(m, k)
a2 = approx_log_cmk(m, k)

c2 = r2 - a2

print(c)
print(c2)

grad = -(ive(m/2, k) / ive(m/2 - 1, k))

v = m / 2
approx_grad = -(k / ((v-1) + np.sqrt(np.square(v+1) + np.square(k))))
print(grad)
print(approx_grad)
