def chain_dict(*translators):
    """Create a translator chain.

    Example
    -------

    ::
    a = {0: "A", 1:"B"}
    b = {"A": "An A", "B": "A B"}

    translator = translator_chain(a, b)



    :param translators:
    :return:
    """
    if len(set(map(len, translators))) > 1:
        raise ValueError("dicts are not of same size")

    class F:
        def __len__(self):
            return len(translators[0])

        def __getitem__(self, code):
            r = code
            for translator in translators:
                r = translator.get(code)
                if r is None:
                    return code
                code = r
            return code

    return F()


if __name__ == "__main__":
    a = {0: "A", 1: "B"}
    b = {"A": "An A", "B": "A B"}

    translator = chain_dict(a, b)
    print(len(translator))
    print(translator[0])
