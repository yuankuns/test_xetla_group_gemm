import torch
import numpy as np
from enum import Enum
base_atol = 1e-2
base_rtol = 2e-2

class QuantMode(Enum):
    SYM = 1
    ASYM = 2
    ASYM_FP_ZP = 3

def unpack_weight(qweight, scales, qzeros, q_config):
    group_size = q_config["group_size"]
    bits = q_config["bits"]
    s32_bits = 32

    assert bits == 4
    # Int32 can store 8 * 4bits data. This is the offset for each data.
    wf = (
        torch.tensor(list(range(0, s32_bits, bits)), dtype=torch.int32)
        .unsqueeze(0)
    )
    zeros = qzeros
    if qzeros is not None and not qzeros.dtype.is_floating_point:
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
        ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(zeros, (2**bits) - 1, out=zeros)

        # zeros = zeros + 1  # TODO(Yi): confirm dequant logic
        zeros = zeros.reshape(scales.shape)

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    torch.bitwise_and(weight, (2**bits) - 1, out=weight)

    return weight, scales, zeros

def dequantize(qweight, scales, qzeros, group_size, g_idx=None):
    q_config = {"group_size": group_size, "bits": 4}
    weight, gptq_scales, gptq_zeros = unpack_weight(
        qweight, scales, qzeros, q_config
    )
    if len(weight.shape) > 2:
        weight = weight.reshape(-1, weight.shape[-1])
    infeatures = weight.shape[0]
    if g_idx is None:
        g_idx = g_idx = (
            torch.arange(infeatures, dtype=torch.int32) // q_config["group_size"]
        )
    if gptq_zeros is None:
        return (weight - 8) * gptq_scales[g_idx]
    elif gptq_zeros.dtype.is_floating_point:
        return (weight - 8) * gptq_scales[g_idx] + gptq_zeros[g_idx]
    else:
        return (weight - gptq_zeros[g_idx]) * gptq_scales[g_idx]

def test_gemm_int4(seed, m, n, k, per_channel, act_order, qmode, dtype):
    # if (dtype == torch.bfloat16) and skip_bf16_input:
    #     pytest.skip("bf16 input is not available on mtl")
    # elif (dtype == torch.bfloat16) and (COMPILER_VERSION < 20240200):
    #     pytest.skip("bf16 input is only available on OneAPI 2024.2 and above")
    torch.manual_seed(seed)
    checking_atol = base_atol
    checking_rtol = base_rtol
    if qmode == QuantMode.ASYM:  # sym needs more tolerance
        checking_atol = 2e-1
        checking_rtol = 5e-2
    input = torch.rand([m, k], dtype=dtype)
    input_torch = input.cpu()
    weight = torch.randint(-2147483648, 2147483647, size=(k // 8, n), dtype=torch.int32)
    group_size = min(64, k)
    if per_channel:
        group_size = k
    group_num = k // group_size

    scales = torch.randn([group_num, n], dtype=dtype)
    # scales = torch.ones([group_num, n], dtype=dtype)
    if qmode == QuantMode.SYM:
        zero_points = None
    # elif qmode == QuantMode.ASYM:
    #     zero_points = self.rand_int4(group_num * n, torch.int32, "xpu").reshape(
    #         group_num, n // 8
    #     )
    # elif qmode == QuantMode.ASYM_FP_ZP:
    #     zero_points = torch.rand([group_num, n], device="xpu", dtype=dtype)

    if act_order:
        g_idx = torch.randperm(k, dtype=torch.int32) // group_size
        shuf_weight = GPTQShuffle(bits=4, blocksize=group_size)
        shuffled_weight, g_idx4kernel = shuf_weight(weight, g_idx)
    else:
        g_idx = None
        g_idx4kernel = None
        shuffled_weight = weight

    weight_fp = dequantize(
        weight, scales, zero_points, group_size, g_idx
    ).cpu()
    weight_fp_t = weight_fp.t().contiguous()
    weight = weight.t().contiguous()
    print(weight.shape, weight_fp_t.shape)
    # scales = scales.t().contiguous()
    out_torch = torch.matmul(input_torch, weight_fp)
    print('x', input_torch[0:4,0:32])
    for j,row in enumerate(weight[0:n,:].view(torch.uint32)):
        print(f"{j:02d}: ", end="")
        for val in row:
            print(f"{val:08x}", end=" ")
        print("")
    print('s', scales)
    print('w_fp_t', weight_fp_t[0:32, 0:32])
    print("o", out_torch[0:8, 0:16])
    # check gemm
    # with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.XETLA):
    #     out_xetla = torch.ops.torch_ipex.mm_int4(
    #         input,
    #         shuffled_weight.t().contiguous(),
    #         scales.t().contiguous(),
    #         zero_points,
    #         group_size,
    #         g_idx4kernel,
    #     )
    dump_dict = {}
    dump_dict['x'] = input_torch.view(torch.uint16).detach().cpu().numpy()
    dump_dict['w'] = weight.detach().cpu().numpy()
    dump_dict['wfp'] = weight_fp_t.view(torch.uint16).detach().cpu().numpy()
    dump_dict['y'] = out_torch.view(torch.uint16).detach().cpu().numpy()
    dump_dict['s'] = scales.view(torch.uint16).detach().cpu().numpy()
    dump_dict['shape'] = np.array([m, n, k, group_size])
    np.savez("int4.npz", **dump_dict)

if __name__ == '__main__':
    #test_gemm_int4(123, 128, 512, 256, False, False, QuantMode.SYM, torch.bfloat16)
    test_gemm_int4(123, 32, 128, 64, False, False, QuantMode.SYM, torch.bfloat16)
