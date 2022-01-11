#!/bin/bash
if [[ -z "${LAST_FM_PUBLIC_KEY}" ]]; then
  echo "Environment variable is not set: LAST_FM_PUBLIC_KEY. Exiting."
  exit
fi
if [[ -z "${LAST_FM_PRIVATE_KEY}" ]]; then
  echo "Environment variable is not set: LAST_FM_PRIVATE_KEY. Exiting"
  exit
fi

app="mustalgia.me"
docker rm -f ${app}
docker rm -f my-redis-container
docker build -t ${app} -f Dockerfile-dev . && \
  docker run --name my-redis-container -d redis && \
  docker run -p 56733:80 \
    --name=${app} \
    -v $PWD/app:/app \
    -v /var/uwsgi:/var/uwsgi \
    --link my-redis-container:redis \
    -p 3031:3031 \
    -e LAST_FM_PUBLIC_KEY=$LAST_FM_PUBLIC_KEY \
    -e LAST_FM_PRIVATE_KEY=$LAST_FM_PRIVATE_KEY \
    ${app}
