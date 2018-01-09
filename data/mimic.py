__author__ = "Isak Karlsson"
import csv

def read_translator(f):
    with open(f) as lines:
        d = {}
        for _, code, short, _ in csv.reader(lines):
            d[code.strip()] = short.strip()
        return d



