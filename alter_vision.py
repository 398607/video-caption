import time
import threading
import Queue
import itchat

class Mover(threading.Thread):
	def __init__(self, alter):
		threading.Thread.__init__(self)
		self.alter = alter
		self.running = True
		self.count = 0
	
	def run(self):
		print 'enter'

		# itchat login
		itchat.auto_login(enableCmdQR=1, hotReload=True)

		while self.running:
			if not self.alter.queue.empty():
				self.count = 0
				msg = self.alter.queue.get()
				itchat.send('[%s]\n%s'%(time.ctime(), msg), toUserName='filehelper')

			self.count += 1
			if self.count > 5 * 3600:
				print '----> [AlterVision] quit after 5h silence'
				self.running = False
			time.sleep(0.5)
	
	def die(self):
		self.running = False
			
class Alter(object):

	al = None

	def __init__(self):
		self.mover = Mover(self)
		self.state = False
		self.queue = Queue.Queue(maxsize=20)
		
	@staticmethod
	def __get_al():
		if Alter.al is None:
			Alter.al = Alter()
		if Alter.al.state is False:
			Alter.al.mover.start()
			Alter.al.state = True
		return Alter.al

	@staticmethod
	def vision(msg):
		Alter.__get_al().__vision(msg)

	@staticmethod
	def die():
		Alter.__get_al().__die()

	def __vision(self, msg):
		self.queue.put(msg)
		# print self.list

	def __die(self):
		self.mover.die()
		

if __name__ == '__main__':
	print 'a'
	Alter.vision('0')
	print 'b'
	Alter.vision('1 hello world')
	print 'b'
	Alter.vision('2 hello world')
	print 'b'
	Alter.vision('3 hello world')
	print 'b'
	Alter.vision('4 hello world')
	print 'b'
	Alter.vision('5 hello world')
	print 'b'
	time.sleep(10)
	Alter.die()