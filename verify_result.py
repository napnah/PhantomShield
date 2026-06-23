L = 2**64
r0 = 12284714200205569230
r1 = 6162029874342084436
result   = (r0 + r1) % L
expected = 12345 * 67890
print(f'P0份额:  {r0}')
print(f'P1份额:  {r1}')
print(f'合并结果: {result}')
print(f'期望结果: {expected}')
print('验证:', '通过' if result == expected else '失败')