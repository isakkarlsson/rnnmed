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
    prev_trans = translators[0]
    for translator in translators[1:]:
        if len(prev_trans) > len(translator):
            raise ValueError("illegal sizes")
        prev_trans = translator

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
