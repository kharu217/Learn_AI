a = {
    'a' : 1,
    'b' : 2,
    'c' : 3,
    'd' : 4
}

l = ['a', 'a', 'b', 'd']
l = list(map(lambda x : a[x], l))
print(l)
