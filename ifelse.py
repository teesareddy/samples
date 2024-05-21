score = 0

ans = input("Who developed the language Python? \n").strip().lower()
if ans == "guido van rossum":
    print("Correct!")
    score += 1
else:
    print("Incorrect!")

ans = input("What does 'OOP' stand for? \n").strip().lower()
if ans == "object oriented programming":
    print("Correct!")
    score += 1
else:
    print("Incorrect!")

ans = input("What does 'HTML' stand for? \n").strip().lower()
if ans == "hyper text markup language":
    print("Correct!")
    score += 1
else:
    print("Incorrect!")

print(f"Your final score is: {score}/3")