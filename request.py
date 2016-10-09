import json
import time
from datetime import date,timedelta
import urllib.request as request
#There is no error checking on either of these functions. Not sure what will happen if it fails

def left_pad(string, length):
    if (len(string)<length):
        for x in range(length-len(string)):
            string = "0" + string
    return string

#Just take the zipcode as a string and have google fix it for us
#Zipcode is passed in as a string
#Day is how many days in the past, as an int
def requestByZip(zipcode, days = 0):
    wReq = request.urlopen(
        "https://maps.googleapis.com/maps/api/geocode/json?address="+zipcode+"&key=InsertYourKey")
    str_response = wReq.read().decode('utf-8')
    obj = json.loads(str_response)
    location = obj["results"][0]["geometry"]["location"]
    return requestByLatLong(str(location["lat"]),str(location["lng"]),days)

#Make sure we start our hourly forecast from the beginning of the day
#This ensure the data will always be organized for the network so that individual hours can be weighted differently
def requestByLatLong(latitude, longitude, days = 0):
    currentDay = date.today()
    requestedDay = currentDay - timedelta(days=days)
    timeString = str(requestedDay.year) + "-" + left_pad(str(requestedDay.month),2) + "-" + left_pad(str(requestedDay.day),2) + "T00:00:00"
    requestUrl = "https://api.darksky.net/forecast/InsertYourKey/" + latitude + "," + longitude+","+ timeString + "?&exclude=minutely,currently,alerts"
    
    wReq = request.urlopen(requestUrl)
    str_response = wReq.read().decode('utf-8')
    obj = json.loads(str_response) #Get the json object of data

    #Though we actually call up a lot more data than just the hourly temperature
    #We filter everything out
    #In the future we should use the daily data in case of error
    #And actually do error checking
    curHourlyData = obj["hourly"]["data"]
    tempSum = 0
    
    toReturn  = []
    for i in range(len(curHourlyData)):
        toReturn= toReturn + [curHourlyData[i]["apparentTemperature"]]
    return toReturn
