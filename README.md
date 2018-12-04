# Run Autotune
### Use Autotune from the OpenAPS Project to determine ISF and CR.

## Build Docker image
`docker build --rm -f "Autotune/Dockerfile" -t autotune:latest Autotune`

## Run Docker image
`docker run --rm -v <absolute path to ./Autotune/data >:/autotune/data autotune:latest`


It depends on files in /data/Input and creates a file output file `autotune-result.json`

# Run Prediction
### 

## Build Docker image
`docker build --rm -f "Python/Dockerfile" -t t1d:latest Python`

## Run Docker image
`docker run --rm -v <path to ./Python>:/t1d -v <path to directory with input data>:/t1d/data t1d:latest`

