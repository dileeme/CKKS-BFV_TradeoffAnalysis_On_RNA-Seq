
import numpy as np
import time
import tenseal as ts
from ckks_utils import create_context


POLY_MOD_DEGREE = 16384
VECTOR_SIZE     = 512
REPEATS         = 20

np.random.seed(42)
context = create_context(POLY_MOD_DEGREE)


vector       = np.abs(np.random.normal(loc=50, scale=20, size=VECTOR_SIZE))
weights      = np.random.normal(loc=0, scale=1, size=VECTOR_SIZE)
vector_list  = vector.tolist()
weights_list = weights.tolist()


_ = ts.ckks_vector(context, vector_list)
enc_times = []
for _ in range(REPEATS):
    start   = time.perf_counter()
    enc_vec = ts.ckks_vector(context, vector_list)
    end     = time.perf_counter()
    enc_times.append((end - start) * 1000)

enc_latency_mean = np.mean(enc_times)
enc_latency_std  = np.std(enc_times)

# ==========================================================
# EXECUTION BENCHMARK
# Re-encrypt each iteration for consistent ciphertext state
# ==========================================================
exec_times = []
for _ in range(REPEATS):
    enc_vec    = ts.ckks_vector(context, vector_list)
    start      = time.perf_counter()
    dot_result = enc_vec.dot(weights_list)
    end        = time.perf_counter()
    exec_times.append((end - start) * 1000)

exec_latency_mean = np.mean(exec_times)
exec_latency_std  = np.std(exec_times)

# ==========================================================
# DECRYPTION BENCHMARK
# Compute dot product once, benchmark decrypt only
# ==========================================================
enc_vec    = ts.ckks_vector(context, vector_list)
dot_result = enc_vec.dot(weights_list)

dec_times = []
for _ in range(REPEATS):
    start      = time.perf_counter()
    dec_result = dot_result.decrypt()
    end        = time.perf_counter()
    dec_times.append((end - start) * 1000)

dec_latency_mean = np.mean(dec_times)
dec_latency_std  = np.std(dec_times)

# ==========================================================
# ACCURACY (MAE vs plaintext)
# ==========================================================
plain_dot = np.dot(vector, weights)
mae       = abs(dec_result[0] - plain_dot)

# ==========================================================
# CIPHERTEXT SIZE
# ==========================================================
enc_fresh = ts.ckks_vector(context, vector_list)
size_kb   = len(enc_fresh.serialize()) / 1024

# ==========================================================
# OUTPUT
# ==========================================================
print("\n========== PROGRAM 2: MEDIUM COMPLEXITY ==========")
print(f"Operation                 : encrypted dot product")
print(f"Polynomial Modulus Degree : {POLY_MOD_DEGREE}")
print(f"Vector Size               : {VECTOR_SIZE}")
print(f"Repeats per measurement   : {REPEATS}")
print("-------------------------------------------------")
print(f"Encryption latency  (ms)  : {enc_latency_mean:.4f}  ± {enc_latency_std:.4f}")
print(f"Execution latency   (ms)  : {exec_latency_mean:.4f}  ± {exec_latency_std:.4f}")
print(f"Decryption latency  (ms)  : {dec_latency_mean:.4f}  ± {dec_latency_std:.4f}")
print(f"Ciphertext size     (KB)  : {size_kb:.4f}")
print(f"MAE                       : {mae:.10f}")
print("=================================================\n")
