import pyrebase
from datetime import date
import time

config = {
  "apiKey": "AIzaSyBc6u4ZNfgmZqN8YzoQjdiKtUV7iNMliWg",
  "authDomain": "phoenix-bot-2021.firebaseapp.com",
  "projectId": "phoenix-bot-2021",
  "databaseURL": "https://phoenix-bot-2021-default-rtdb.firebaseio.com",
  "storageBucket": "phoenix-bot-2021.appspot.com"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()

stock = 'AAPL'


def generateEntry(stockName, imgPath):
  dateToday = date.today()
  timestamp = time.time()
  data = {"timestamp": timestamp}

  # generate storage object
  storage.child(f"graphs/{stockName}/{dateToday}").put(imgPath)

  # generate db object
  db.child(f"storagepath/graphs/{stockName}/{dateToday}").set(data)

generateEntry(stock, "predict_aapl.png")