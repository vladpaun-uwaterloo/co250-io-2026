# maximize 3x1 + 2x2
#
# x1 + x2 + x3      = 4
# 2x1 + x2      + x4 = 5
# x >= 0

m 2
n 4

c 3 2 0 0

A
1 1 1 0
2 1 0 1

b 4 5

offset 0