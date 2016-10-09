from request import requestByZip
from datetime import date,timedelta
import numpy as np
import neural_net

f = open('saves/data_saves.txt', 'r')
#for x in range(900):
#    tempsArray = requestByZip("02459",x)
#    f.write(str(tempsArray)) #Note to self, never forget a new line again
#f.close() 

#Using just the average velocity of temperatures over the day to do a prediction
#Could be made (very slightly) more accurate using acceleration etc., but this is good for now
#because we always have to simply predict a single hour
def increase_to(array):
	derivs = [(array[i+1] - array[i])/2 for i in range(len(array)-1)]
	aveDeriv = sum(derivs)/len(derivs)
	array +=  [array[len(array)-1] + aveDeriv * (i+1) for i in range (24 - len(array))]
	return array

def decrease_to(array):
	return array[0:24]

#All the work below will no longer be necessary soon
split_up=f.readline().split("[")[1:]
for x in range(len(split_up)):
	split_up[x] = split_up[x].replace("]","")
	split_up[x] = split_up[x].split(",")
	split_up[x] = [float(to_float) for to_float in split_up[x]]
	if (len(split_up[x])<24):
		split_up[x] = increase_to(split_up[x])
	elif (len(split_up[x])>24):
		split_up[x] = decrease_to(split_up[x])
	if (len(split_up[x])!=24):
		print("ERROR" + str(len(split_up[x])))

oct_9_temp = [62.84, 62.58, 62.65, 62.79, 62.03, 61.43, 60.77, 58.87, 56.68, 56.34, 55.68, 55.29, 54.49, 54.61, 54.43, 54.05, 53.51, 52.69, 51.49, 50.71, 44.79, 43.6, 42.74, 41.58]
neural_net.main(split_up) #Do not run this line and the line below it
#neural_net.test_load(oct_9_temp) #For that will cause a graph duplication