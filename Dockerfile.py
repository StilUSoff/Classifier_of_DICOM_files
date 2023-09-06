# first stage
FROM python:3.9 AS builder
COPY settings.txt .

# install dependencies to the local user directory (eg. /root/.local)
RUN pip install --user -r settings.txt

# second stage
FROM python:3.9-slim
WORKDIR /code

# copy only the dependencies that are needed for our application and the source files
COPY --from=builder /root/.local /root/.local
COPY ./src .

# update PATH
ENV PATH=/root/.local:$PATH

# make sure you include the -u flag to have our stdout logged
CMD [ "python", "-u", "./bin/classifier/run_for_docker.py" ]