import time
import numpy as np
from decay import simulate, simulate_loop

# Параметры бенчмарка
N0 = 200_000
lam = 0.4

print(f"Запуск сравнения производительности для N0 = {N0:,} атомов...\n")

# 1. Измерение времени чисто питоновской версии (циклы Python)
t0 = time.perf_counter()
res_python = simulate_loop(N0, lam)
t1 = time.perf_counter()
time_python = t1 - t0

# 2. Измерение времени NumPy-версии (векторизованные операции)
t0 = time.perf_counter()
res_numpy = simulate(N0, lam)
t1 = time.perf_counter()
time_numpy = t1 - t0

# Вычисление коэффициента ускорения (speedup)
speedup = time_python / time_numpy if time_numpy > 0 else float('inf')

# Вывод результатов
print(f"Время выполнения (Python loop): {time_python:.4f} сек")
print(f"Время выполнения (NumPy):       {time_numpy:.4f} сек")
print(f"Коэффициент ускорения (Speedup): {speedup:.2f}x")