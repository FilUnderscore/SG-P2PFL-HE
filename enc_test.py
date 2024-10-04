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

shape = one.shape
tensors = []

for i in range(0, len(ones), 1):
    enc_zero = ts.ckks_tensor(context, zeroes[i])
    enc_one = ts.ckks_tensor(context, ones[i])

    result = enc_zero + enc_one + enc_one
    result_tensor : ts.PlainTensor = result.decrypt()

    tensors.append(torch.tensor(result_tensor.raw))
    print('i ' + str(i) + ' out of ' + str(len(ones)))

resultant_tensor = torch.cat(tensors).reshape(shape)
print(resultant_tensor)