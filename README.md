# How to run the app

## Get Code
Check out this repository

## Prepare input data

### Health Data
Patient health data needs to be provided in a csv file. It needs the following columns:
* date	(eg. 15.12.17)
* time	(eg. 13:59)
* bgValue	
* cgmValue	
* basalValue	
* bolusValue	
* mealValue
* glucoseAnnotation

Go to Folder `cd data/csv`

Add file called **`data.csv`**

Start with a file containing about 5 days of history and see how fast the computation is.

### Autotune config files

Go to Folder `cd data/input` (call it from root of project)

Add your **profile.json** and **profile.pump.json** files

## Use a Docker container to run the python and nodejs code

### Build Docker image
Build the docker image from the provided Dockerfile

`docker build -t t1d-pred:latest .` (call from root of project)

### Run Docker image
You need to mount your local files and input data to the docker container

`docker run --rm -v <absoulute path to root of project (should end with /t1d-prediction)>:/t1d t1d-pred:latest`

Inside the docker container the python/main.py will be called.

## Run Docker image with autowatch
Nodemon will watch inside the docker container for file changes (only .py) and will restart the main.py
`docker run --rm -v <absoulute path to root of project>:/t1d t1d-pred:latest nodemon --exec python python/main.py -e py -L`

### To stop running containers with autowatch

`docker stop $(docker ps -a -q --filter ancestor=t1d-pred:latest --format="{{.ID}}")`

## Change code
To change code, open the python directory with your editor of choice. For rapid development start the docker container with autowatch. It will rerun the code everytime you change a python file. 

After the first run, autotune has created all necessary files and run_autotune in python/main.py can be set to False for a faster runtime. 
