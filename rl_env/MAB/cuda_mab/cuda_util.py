import numba.cuda as cuda
import numba.cuda.random as cu_rand


@cuda.jit(device=True)
def rand_uniform(th_id, rng_state):
    return cu_rand.xoroshiro128p_uniform_float32(rng_state, th_id)


@cuda.jit(device=True)
def rand_normal(th_id, rng_state):
    return cu_rand.xoroshiro128p_normal_float32(rng_state, th_id)


@cuda.jit(device=True)
def rand_int(high, th_id, rng_state):
    return int(rand_uniform(th_id, rng_state) * high)


@cuda.jit(device=True)
def argmax(arr):
    max_val = arr[0]
    res = 0
    for idx in range(1, len(arr)):
        if arr[idx] > max_val:
            max_val = arr[idx]
            res = idx
            
    return res


@cuda.jit(device=True)
def copy_1d_arr(dst, src):
    for idx in range(len(src)):
        dst[idx] = src[idx]
