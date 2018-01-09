def read_visits(code_path, code_len=None, visit_sep="|", code_sep=" "):
    """Read data from `code_path` optionally truncating the codes to at max `code_len`

    Each code is separated by `code_sep` and each visit is separated by a `sep`

    :param code_path: the file path to read from
    :param code_len: the maximum length of code
    :param visit_sep: the visit separator
    :param code_sep: the code separator
    :return: a tuple with 4 arguments: (data, dictionary, reverse_dictionary, code_size)
    """
    with open(code_path) as code_file:
        lines = code_file.readlines()
        pats = (line.strip().split(visit_sep) for line in lines)
        visits = []
        dictionary = {}
        for pat in pats:
            out_visit = []
            for visit in (vis.strip().split(code_sep) for vis in pat):
                out = set()
                for code in visit:
                    code = code[:code_len] if code_len is not None else code
                    index = dictionary.get(code)
                    if index is None:
                        index = len(dictionary)
                        dictionary[code] = index
                    out.add(code)
                out_visit.append(list(out))
            visits.append(out_visit)
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return list(visits), dictionary, reversed_dictionary


def read_translate(f, code_len=None):
    with open(f) as lines:
        d = {}
        for code, translation in map(lambda x: x.strip().split("|"), lines):
            d[code[:code_len] if code_len is not None else code] = translation
        return d