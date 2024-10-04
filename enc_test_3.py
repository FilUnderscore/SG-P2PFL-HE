import tenseal as ts
import torch

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

zero = torch.zeros(784, 69)
zeroes = zero.reshape(1, zero.shape[0] * zero.shape[1])[0]

one = torch.ones(784, 69)
ones = one.reshape(1, one.shape[0] * one.shape[1])[0]

enc_zero = ts.ckks_vector(context, zeroes)
enc_one = ts.ckks_vector(context, ones)
result = enc_zero + enc_one + enc_one

print(result.decrypt())