class AlwaysIn:
    """Dummy type to simulate a collection that contains all items"""

    def __contains__(self, item):
        return True


# singleton instance of AlwaysIn
__always_in__ = AlwaysIn()


def read_visits(code_path, transform=None, valid_code=None, visit_sep="|",
                code_sep=" "):
    """Read data from `code_path` optionally truncating the codes to at max `code_len`

    Each code is separated by `code_sep` and each visit is separated by a `sep`

    :param code_path: the file path to read from
    :param transform: 
    :param valid_code: 
    :param visit_sep: the visit separator
    :param code_sep: the code separator

    :return: a tuple (data, dictionary, reverse_dictionary)

    """
    valid_code = valid_code or __always_in__
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
                    code = transform(code) if transform is not None else code
                    if valid_code is not None and code in valid_code:
                        index = dictionary.get(code)
                        if index is None:
                            index = len(dictionary)
                            dictionary[code] = index
                        out.add(code)

                if out:
                    out_visit.append(out)
            if out_visit:
                visits.append(out_visit)
        if not visits:
            raise AttributeError("no visits")

        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return list(visits), dictionary, reversed_dictionary


def read_dadel_visits(self, f):
    """Read a dataset consisting of DADEL visits.

    :param f: the reader
    :returns: a tuple with (data, dictionary, reverse_dictionary)
    :rtype: tuple
    """
    
    pass
