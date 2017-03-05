#! /usr/bin/env python3

#simple exercise to demostrate issues that can be encountered when dealing with very large numbers and very small numbers
#answer should be 1.0. outputed answer is ~0.95
n = 1000000000
for i in range(0, 1000000):
    n += 0.000001
n -= 1000000000

print(n)
