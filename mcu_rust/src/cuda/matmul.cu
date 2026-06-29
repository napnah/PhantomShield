#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {

__global__ void matmul_u64_kernel(
    const uint64_t *a,
    const uint64_t *b,
    uint64_t *out,
    size_t m,
    size_t k,
    size_t n
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    uint64_t acc = 0;
    for (size_t t = 0; t < k; ++t) {
        acc += a[row * k + t] * b[t * n + col];
    }
    out[row * n + col] = acc;
}

__global__ void party_matmul_finish_kernel(
    const uint64_t *a,
    const uint64_t *b,
    const uint64_t *ra,
    const uint64_t *rb,
    const uint64_t *hp_share,
    uint64_t *out,
    size_t m,
    size_t k,
    size_t n,
    int include_rarb
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    uint64_t corr = 0;
    for (size_t t = 0; t < k; ++t) {
        uint64_t av = a[row * k + t];
        uint64_t bv = b[t * n + col];
        uint64_t rav = ra[row * k + t];
        uint64_t rbv = rb[t * n + col];
        corr += av * rbv + rav * bv;
        if (include_rarb) {
            corr += rav * rbv;
        }
    }
    size_t idx = row * n + col;
    out[idx] = hp_share[idx] - corr;
}

__global__ void hp_matmul_share_kernel(
    const uint64_t *a0,
    const uint64_t *a1,
    const uint64_t *b0,
    const uint64_t *b1,
    const uint64_t *out0,
    uint64_t *out1,
    size_t m,
    size_t k,
    size_t n
) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    uint64_t product = 0;
    for (size_t t = 0; t < k; ++t) {
        uint64_t av = a0[row * k + t] + a1[row * k + t];
        uint64_t bv = b0[t * n + col] + b1[t * n + col];
        product += av * bv;
    }
    size_t idx = row * n + col;
    out1[idx] = product - out0[idx];
}

struct WorkspaceSlot {
    uint64_t *ptr;
    size_t bytes;
};

static WorkspaceSlot g_slots[7] = {
    {nullptr, 0}, {nullptr, 0}, {nullptr, 0}, {nullptr, 0},
    {nullptr, 0}, {nullptr, 0}, {nullptr, 0},
};

static int ensure_slot(int slot, size_t bytes) {
    if (bytes == 0) {
        return 0;
    }
    if (g_slots[slot].bytes >= bytes && g_slots[slot].ptr != nullptr) {
        return 0;
    }
    if (g_slots[slot].ptr != nullptr) {
        cudaFree(g_slots[slot].ptr);
        g_slots[slot].ptr = nullptr;
        g_slots[slot].bytes = 0;
    }
    cudaError_t err = cudaMalloc((void **)&g_slots[slot].ptr, bytes);
    if (err != cudaSuccess) {
        return (int)err;
    }
    g_slots[slot].bytes = bytes;
    return 0;
}

static int copy_to_slot(int slot, const uint64_t *host, size_t bytes) {
    int code = ensure_slot(slot, bytes);
    if (code) {
        return code;
    }
    if (bytes == 0) {
        return 0;
    }
    return (int)cudaMemcpy(g_slots[slot].ptr, host, bytes, cudaMemcpyHostToDevice);
}

static uint64_t *slot_ptr(int slot) {
    return g_slots[slot].ptr;
}

int mcu_cuda_matmul_u64(
    const uint64_t *a_host,
    const uint64_t *b_host,
    uint64_t *out_host,
    size_t m,
    size_t k,
    size_t n
) {
    size_t a_bytes = m * k * sizeof(uint64_t);
    size_t b_bytes = k * n * sizeof(uint64_t);
    size_t out_bytes = m * n * sizeof(uint64_t);
    dim3 block(16, 16);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x),
              (unsigned int)((m + block.y - 1) / block.y));

    int code = copy_to_slot(0, a_host, a_bytes);
    if (code) return code;
    code = copy_to_slot(1, b_host, b_bytes);
    if (code) return code;
    code = ensure_slot(6, out_bytes);
    if (code) return code;

    matmul_u64_kernel<<<grid, block>>>(slot_ptr(0), slot_ptr(1), slot_ptr(6), m, k, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;

    err = cudaMemcpy(out_host, slot_ptr(6), out_bytes, cudaMemcpyDeviceToHost);
    return (int)err;
}

int mcu_cuda_party_matmul_finish_u64(
    const uint64_t *a_host,
    const uint64_t *b_host,
    const uint64_t *ra_host,
    const uint64_t *rb_host,
    const uint64_t *hp_share_host,
    uint64_t *out_host,
    size_t m,
    size_t k,
    size_t n,
    int include_rarb
) {
    size_t a_bytes = m * k * sizeof(uint64_t);
    size_t b_bytes = k * n * sizeof(uint64_t);
    size_t out_bytes = m * n * sizeof(uint64_t);
    dim3 block(16, 16);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x),
              (unsigned int)((m + block.y - 1) / block.y));

    int code = copy_to_slot(0, a_host, a_bytes);
    if (code) return code;
    code = copy_to_slot(1, b_host, b_bytes);
    if (code) return code;
    code = copy_to_slot(2, ra_host, a_bytes);
    if (code) return code;
    code = copy_to_slot(3, rb_host, b_bytes);
    if (code) return code;
    code = copy_to_slot(4, hp_share_host, out_bytes);
    if (code) return code;
    code = ensure_slot(6, out_bytes);
    if (code) return code;

    party_matmul_finish_kernel<<<grid, block>>>(
        slot_ptr(0), slot_ptr(1), slot_ptr(2), slot_ptr(3), slot_ptr(4), slot_ptr(6), m, k, n, include_rarb
    );
    code = (int)cudaGetLastError();
    if (code) return code;
    code = (int)cudaDeviceSynchronize();
    if (code) return code;
    return (int)cudaMemcpy(out_host, slot_ptr(6), out_bytes, cudaMemcpyDeviceToHost);
}

int mcu_cuda_hp_matmul_share_u64(
    const uint64_t *a0_host,
    const uint64_t *a1_host,
    const uint64_t *b0_host,
    const uint64_t *b1_host,
    const uint64_t *out0_host,
    uint64_t *out1_host,
    size_t m,
    size_t k,
    size_t n
) {
    size_t a_bytes = m * k * sizeof(uint64_t);
    size_t b_bytes = k * n * sizeof(uint64_t);
    size_t out_bytes = m * n * sizeof(uint64_t);
    dim3 block(16, 16);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x),
              (unsigned int)((m + block.y - 1) / block.y));

    int code = copy_to_slot(0, a0_host, a_bytes);
    if (code) return code;
    code = copy_to_slot(1, a1_host, a_bytes);
    if (code) return code;
    code = copy_to_slot(2, b0_host, b_bytes);
    if (code) return code;
    code = copy_to_slot(3, b1_host, b_bytes);
    if (code) return code;
    code = copy_to_slot(4, out0_host, out_bytes);
    if (code) return code;
    code = ensure_slot(6, out_bytes);
    if (code) return code;

    hp_matmul_share_kernel<<<grid, block>>>(
        slot_ptr(0), slot_ptr(1), slot_ptr(2), slot_ptr(3), slot_ptr(4), slot_ptr(6), m, k, n
    );
    code = (int)cudaGetLastError();
    if (code) return code;
    code = (int)cudaDeviceSynchronize();
    if (code) return code;
    return (int)cudaMemcpy(out1_host, slot_ptr(6), out_bytes, cudaMemcpyDeviceToHost);
}

}
