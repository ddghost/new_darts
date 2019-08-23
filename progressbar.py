import time
class progressbar(object):
	progressTime = 0
	def __init__(self, stepNum, frontStr='', backStr=''):
		self.stepNum = stepNum
		self.fullRate = 100
		self.lastOutputLen = 0
		self.frontStr = frontStr
		self.backStr = backStr
		self.lastTime =  time.perf_counter()
        
	def clear(self):
		spaceStr = ' ' * self.lastOutputLen
		print('\r%s' % (spaceStr),end='' )
		
	def output(self, nowStep):
		
		self.progressTime += time.perf_counter() - self.lastTime
		self.lastTime = time.perf_counter()
		rate = nowStep / self.stepNum 
		if(rate == 0):
			predictRestTime = -1
		else:
			predictRestTime = self.progressTime * (1-rate) / rate
		rate *= self.fullRate
		barLen = int(rate ) // 2
		barStr = '#' * barLen 
		#self.clear()
		outputStr = '\r%s%s%.2f%% [%.2fs/%.2fs]%s' % \
					(self.frontStr, barStr, rate, self.progressTime, predictRestTime, self.backStr)
		self.lastOutputLen = len(outputStr)
		print(outputStr, end='' )
	def getProgressTime():
		return self.progressTime
	
 
 
 
if __name__ == '__main__':
	bar = progressbar(100)
	for i in range(100):
		bar.output(i+1)
		time.sleep(0.1)
