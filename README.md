# mustalgia.me

This is the source code for [mustalgia.me](https://mustalgia.me).
Please raise an issue if you find any. 


## Running locally

Due to the `celery` dependency, the best way to run this locally is with `docker`, which handles running `redis` and the dev server. 

To run locally, you will first need to [acquire API access for last.fm](https://www.last.fm/api#getting-started). 

Once you've done that, export the your API keys to the following environment variables:

```
LAST_FM_PUBLIC_KEY=XXX
LAST_FM_PRIVATE_KEY=XXX
```

Then, simply execute the `./start.dev.sh` script. The app will be available on at `localhost:3031`.

