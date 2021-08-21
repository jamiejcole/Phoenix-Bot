// Phoenix Bot v0.2 CC 2021 do not distribute! 
// Firebase config (needed to get the firebase information/images)
var firebaseConfig = {
apiKey: "AIzaSyBc6u4ZNfgmZqN8YzoQjdiKtUV7iNMliWg",
authDomain: "phoenix-bot-2021.firebaseapp.com",
databaseURL: "https://phoenix-bot-2021-default-rtdb.firebaseio.com",
projectId: "phoenix-bot-2021",
storageBucket: "phoenix-bot-2021.appspot.com",
};

// Initialize Firebase, setting up each feature we use from the database
firebase.initializeApp(firebaseConfig);
var storage = firebase.storage();
var database = firebase.database();


/* ----------------------------------------------------------------------- 

                            Updating HTML

   -----------------------------------------------------------------------
*/

// Creating the function to access which stock is selected
function getQueryParameter() {
  var query = window.location.search.substring(1);
  var stock = query.split("&");
  console.log(stock)
  return (stock);
}

// Setting the current stock so that the stocks are all available on one HTML page
function setHTML(stock) {
    console.log(document.getElementById(stock).className);
    document.getElementById(stock).className = "stock-links " + stock + " current"; 
    console.log(document.getElementById(stock).className);

}

//TEMP - THIS IS FOR IF NO DATA EXISTS FOR THAT DAY YET DEBUGGING PURPOSES ONLY - Disregard
//today = '2021-08-13'

// Getting the document name and therefore the stock name for the firebase path
// this just removes the excess file crap from which file calls the JS so there is 1 JS file for all stocks
var stockName = getQueryParameter()
setHTML(stockName);

/* ----------------------------------------------------------------------- 

                        Getting latest entry date

   -----------------------------------------------------------------------
*/

var today = ''
const databaseTimeRef = firebase.database().ref('storagepath/graphs/' + stockName + '/');

// Function that grabs the latest date from the Firebase RTDB
function getLatestDate (_callback) {
    databaseTimeRef.once('value').then((snapshot) => {
        const data = snapshot.val();
        var lastDate = [];
        for (const i in data) {
            lastDate.push(i);
        };
        today = lastDate[lastDate.length - 1];
        console.log(today);
        _callback();
    });
};


/* ----------------------------------------------------------------------- 

                            Retreiving Data

   -----------------------------------------------------------------------
*/

getLatestDate(function() {
    // Setting both references to either database/storage so that the script can access it
    var storageRef = storage.ref('graphs/' + stockName + '/' + today);
    var databaseRef = firebase.database().ref('storagepath/graphs/' + stockName + '/' + today);

    // Setting a reference to the HTML page to set the info too
    var stockInfoData = document.getElementById("info");
    var buySellText = document.getElementById("buysell");

    // Grabbing the stock info from the firebase database, formatting it, and setting it to the info id with pretty colours
    databaseRef.once('value', (snapshot) => {
        const data = snapshot.val();
        output = "Previous Close: $" + parseFloat(data['previousCloseReal']).toFixed(2) + "USD\n\n Todays Predicted Close: $" + parseFloat(data['todayClosePredicted']).toFixed(2) + "USD\n\n Tomorrow's Predicted Close: $" + parseFloat(data['tomorrowClosePredicted']).toFixed(2) + 'USD';

        stockInfoData.innerText = output;
        if (data['tomorrowClosePredicted'] >= data['previousCloseReal']) {
            buySellText.innerHTML = '<p style="font-size: 24px;">Buy! <i style="color: #2ed573" class="fas fa-arrow-circle-up"></i></p>'
        }
        else {
            buySellText.innerHTML = '<p style="font-size: 24px;">Sell! <i style="color: #ff4757" class="fas fa-arrow-circle-down"></i></p>'
        }
    });

    // Grabbing the image from the storage firebase and setting it to the img id graph in the HTML
    storageRef.getDownloadURL().then((url) => {
        var img = document.getElementById('graph');
        img.setAttribute('src', url);
    });
});



