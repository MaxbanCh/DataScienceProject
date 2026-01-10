# Data scraper for data.ina.fr
There was some data on the web api of data.ina.fr, so there is a little bot to scrap all of it !  
Because I don't want trouble with INA I will not upload the data (even though the server had crashed sometimes because of me...)

## What you can do ?
For now, you can scrap the data presents for the :
- Women-Men proportion in time spoken for each channel
- The number of iteration a word has been pronounced on each channel
- The top50 of each personality most evoked for each channel

## How to use it ?
You should have python installed on your machine
```sh
git clone 
python3 -m venv .venv
source .venv/bin/activate
cd DataScienceProject/ina-api
```

To modify the data that you want to collect you should edit the main.py file.

You can indicate : 
- The begin and end year
- The begin and end month
- The word that you want to research if it's for the word analyzer

Then you can run with :
```sh
python3 main.py
```

Be careful for the top50, if you ask to many month at the same time (arround 4 months) you can get the achievement error 500 and wait for some minutes to redo it ^^'

The data collected are store in the data folder in csv.