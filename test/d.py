from a import A
from b import B
from c import C

class D:
	def __init__(self):
		A.a = "class D"
		B.change()
		print(A.a)
		C.change()
		print(A.a)

D()