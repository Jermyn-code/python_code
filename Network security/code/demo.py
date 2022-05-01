def bar(a, b, *args, start=5):
    """

    :param a:
    :param b:
    :param args:
    :param start:
    :return:
    """
    c = a + b + start
    for arg in args:
        c += arg
    print(c)


bar(1, 2, 3, 5, 6, start=0)
bar(1, 3, 5, 6, 8, 7, start=0)
