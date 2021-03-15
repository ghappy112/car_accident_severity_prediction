# flask application

from flask import Flask, request, redirect, url_for, request, render_template
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import joblib
import sys

# logging helper
def p(*args):
  print(args[0] % (len(args) > 1 and args[1:] or []))
  sys.stdout.flush()

app = Flask(__name__)
try:
    model = joblib.load(r"TheTreeOfSeverity.pkl")
except Exception as e:
    p(e)
    
@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == "POST":
        data = request.form
        usercity = data.get('city')
        userstate = data.get('state')
        usercountry = data.get('country')
        return redirect(url_for("/predict", city=usercity, state=userstate, country=usercountry))
    else:
        return '''
            <html>
                <body style="background-color:CornflowerBlue;">
                    <h1 style="color:white;">Car Accident Severity Classifer</h1>
                    <h1 style="color:white;">Estimate the traffic delay severity from car accidents using local weather conditions and machine learning</h1>
                    <form style="color:white;" action="/predict" method=POST>
                        <label for="city">Your city:</label><br>
                        <input type="text" id="city" name="city"><br><br>
                        <label for="state">Your state/province's abbreviations:</label><br>
                        <input type="text" id="state" name="state"><br><br>
                        <label for="country">Your country's abbreviations:</label><br>
                        <input type="text" id="country" name="country"><br><br>
                        <input type="submit" value="Submit">
                    </form> 
                </body>
            </html>
        '''
    
@app.route('/predictor', methods=['GET', 'POST'])
def predictor():

    #get user input
    data = request.form
    city = data.get('city')
    state = data.get('state')
    country = data.get('country')
    state = state.upper()
    country = country.upper()
    if country == "USA" or country == "U.S." or country == "U.S.A.":
        country = "US"

    #get weather data from weatherbit
    Key = "You can create a key for free at weatherbit.io. If you want my key, please message me on slack."
    url = "https://api.weatherbit.io/v2.0/current?city=" + city + "&state=" + state + "&country=" + country + "&key=" + Key + "&include=minutely"
    r = requests.get(url, auth=('user', 'pass'))
    weather_data = r.json()
    temp = weather_data['data'][0]['temp']
    temp = (temp * 9/5) + 32
    wind_spd = weather_data['data'][0]['wind_spd']
    wind_spd *= 2.23694
    wind_chill = temp - (wind_spd * 0.7)
    pres = weather_data['data'][0]['pres']
    pres *= 0.0295301
    vis = weather_data['data'][0]['vis']
    vis *= 0.621371
    precip = weather_data['data'][0]['precip']
    precip *= 0.0394 * 24
    pod = weather_data['data'][0]['pod']
    if pod == 'd':
        day = 1
        night = 0
    elif pod == 'n':
        day = 0
        night = 1
    X = [temp, wind_chill, pres, vis, wind_spd, precip, day, night]
    
    #ml prediction
    prediction = (model.predict([X]))[0] 
    prediction = str(prediction)
    
    return '''
        <html>
            <body style="background-color:CornflowerBlue;">
                <h1 style="color:white;">Car Accident Severity Prediction:</h1>
                <h1 style="color:white;">''' + prediction + '''</h1>
                <br>
                <h1 style="color:white;">(Traffic delay severity estimates range between 1-4. 4 is more severe than 3, 3 is more severe than 2, and 2 is more severe than 1)</h1>
                <h1 style="color:white;">(Severity is defined as how long traffic delays caused by the accident are)</h1>
            </body>
        </html>
    '''

if "__name__" == "__main__":
    app.run(debug=True)
