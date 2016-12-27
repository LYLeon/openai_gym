
class test:
	def __private_test(self):
		print('i am private')

	def call_private(slef):
		slef.__private_test()

t = test()
t.call_private()