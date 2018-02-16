__author__ = "Isak Karlsson"
import csv


def read_translator(f):
    """Read a dictionary of code to explanation from the MIMIC-III
    ICD_DIAGNOSIS-file

    :param f: the file (closed aftar call)
    :return: a dict, code => description

    """
    with f as lines:
        d = {}
        for _, code, short, _ in csv.reader(lines):
            d[code.strip()] = short.strip()
        return d
