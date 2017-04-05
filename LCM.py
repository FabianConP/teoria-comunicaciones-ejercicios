def pairGCD(a, b):
	return a if b == 0 else pairGCD(b, a % b)


def pairLCM(a, b):
	return (a * b) / pairGCD(a, b)


def LCM(* values):
	lcm = 1
	for v in values:
		lcm = pairLCM(lcm, v)
	return lcm


print(LCM(2,3,5))
	