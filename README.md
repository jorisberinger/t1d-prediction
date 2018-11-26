# Run Autotune
### Use Autotune from the OpenAPS Project to determine ISF and CR.

## Build Docker image
`docker build --rm -f "Autotune/Dockerfile" -t autotune:latest Autotune`

## Run Docker image
`docker run --rm -v <absolute path to ./Autotune/data >:/autotune/data autotune:latest`

It depends on files in /data/Input and creates a file output file `autotune-result.json`

