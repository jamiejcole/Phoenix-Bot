import pyrebase
from datetime import date

config = {
  "apiKey": "AIzaSyBc6u4ZNfgmZqN8YzoQjdiKtUV7iNMliWg",
  "authDomain": "phoenix-bot-2021.firebaseapp.com",
   "projectId": "phoenix-bot-2021",
  "databaseURL": "https://databaseName.firebaseio.com",
  "storageBucket": "phoenix-bot-2021.appspot.com"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

stock = 'AAPL'
date = date.today()

storage.child(f"graphs/{stock}/{date}").put("predict_aapl.png")
