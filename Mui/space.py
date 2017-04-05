str = raw_input("Type input: ")

charArray = list(str)

words = 1
insideWord = False

for char in charArray:
    if (char == ' ' && insideWord):
        words += 1
        insideWord = False
    elif (char != ' '):
        insideWord = True
print(words)
