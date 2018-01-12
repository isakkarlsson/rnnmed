def read_translator(f, code_len=None, sep="|"):
    """ Read an translator from code to translator

    :param f: input (file)
    :param code_len: max code length
    :param sep: separator
    :return: a dict: code => description
    """
    with f as lines:
        d = {}
        for code, translation in map(lambda x: x.strip().split(sep), lines):
            d[code[:code_len] if code_len is not None else code] = translation
        return d
