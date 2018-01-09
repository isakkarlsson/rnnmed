visits = [["A", "a"], ("B", "b")]

print("|".join(map(lambda visit: " ".join(visit[1]), visits)))