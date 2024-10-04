import tenseal as ts
import torch

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

zero = torch.zeros(784, 69)
zeroes = zero.chunk(50)
print(zero)

one = torch.ones(784, 69)
ones = one.chunk(50)
print(one)

for i in range(0, len(ones), 1):
    enc_zero = ts.ckks_tensor(context, zeroes[i])
    enc_one = ts.ckks_tensor(context, ones[i])

    result = enc_zero + enc_one + enc_one
    result_tensor : ts.PlainTensor = result.decrypt()

    print(result_tensor.raw)
    print('I to len ' + str(i) + ' ' + str(len(ones)))