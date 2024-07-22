import itertools

def is_solution(s, m, o, r, y, e, n, d):
    # Form the two numbers from the letters
    send = s * 1000 + e * 100 + n * 10 + d
    more = m * 1000 + o * 100 + r * 10 + e
    money = m * 10000 + o * 1000 + n * 100 + e * 10 + y

    # Check if the sum of SEND and MORE equals MONEY
    return send + more == money

# List of unique digits (0-9) and the unique letters in the problem
digits = range(10)
letters = 'smoreynd'

# Try all possible permutations of the digits
for perm in itertools.permutations(digits, len(letters)):
    # Map letters to the corresponding digits
    mapping = dict(zip(letters, perm))

    # Extract the digits
    s, m, o, r, e, y, n, d = [mapping[letter] for letter in letters]

    # Skip permutations where S or M are zero, as leading zeros are not allowed
    if s == 0 or m == 0:
        continue

    # Check if the current permutation solves the problem
    if is_solution(s, m, o, r, y, e, n, d):
        print(f"SEND + MORE = MONEY: {mapping}")
        break
