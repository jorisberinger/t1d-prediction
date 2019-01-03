# How to run the app

## Get Code
Check out this repository

## Prepare input data

### Health Data
Patient health data needs to be provided in a csv file. It needs the following columns:
* date	
* time	
* bgValue	
* cgmValue	
* basalValue	
* bolusValue	
* mealValue

Go to data folder

`cd data`

add file called **`data.csv`**

Start with a file containing about 15 days of history and see how fast the computation is.

### Autotune config files


Create new folder for autotune Input Data 

(still in data folder)
`mkdir Input`

Add your **profile.json** and **profile.pump.json** files

## Use a Docker container to run the python and nodejs code

### Build Docker image
Build the docker image from the provided Dockerfile

`docker build -t t1d-pred:latest .`

### Run Docker image
You need to mount your local files and input data

`docker run --rm -v <path to local folder>/t1d-prediction:/t1d t1d-pred:latest`

Inside the docker container the python/main.py will be called.

## Run Docker image with autowatch
Nodemon will watch inside the docker container for file changes (only .py) and will restart the main.py
`docker run --rm -v <path to local folder>/t1d-prediction:/t1d t1d-pred:latest nodemon --exec python python/main.py -e py -L`

### To stop running containers with autowatch

`docker stop $(docker ps -a -q --filter ancestor=t1d-pred:latest --format="{{.ID}}")`

## Change code
To change code, open the python directory with your editor of choice. For rapid development start the docker container with autowatch. It will rerun the code everytime you change a python file. 

After the first run, autotune has created all necessary files and run_autotune in python/main.py can be set to False for a faster runtime. 