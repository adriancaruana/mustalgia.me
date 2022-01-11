import dataclasses
from pathlib import Path
import traceback
from typing import Dict
import shutil

from celery import Celery
from flask import Flask, request, render_template, redirect, url_for


from _cache import Cache
from _engine import (
    Engine,
    HISTORICAL_ARTEFACTS_TYPES,
    HISTORICAL_TITLES_AND_DESCRIPTIONS,
    TIMELY_TITLES_AND_DESCTIPTIONS,
    HISTORICAL_META
)
from _fetch import Fetch, FETCH_LIMIT_HIGH, TRACKS_IN_MEMORY_LIMIT

from _utils import Auth
from _covers import get_covers

def make_celery(app):
    celery = Celery(
        "app",
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask

    return celery


app = Flask(__name__)

# Patch it for celery
app.config.update(
    CELERY_BROKER_URL='redis://redis:6379/0',
    CELERY_RESULT_BACKEND='redis://redis:6379/0',
    # CELERY_IMPORTS=''
)
# Init Celery
celery = make_celery(app)


def get_full_class_name(obj):
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + "." + obj.__class__.__name__


@dataclasses.dataclass
class Runner:
    user: str
    cache: Cache

    def __post_init__(self):
        self._fetch = Fetch(
            user=self.user,
            pkey=Auth.get_public_key(),
            skey=Auth.get_secret_key(),
            cache=self.cache,
        )

    def run(self):
        meta, df = self._fetch.fetch_and_cache()
        self._engine = Engine(df=df, cache=self.cache)
        timely, historical = self._engine.run_engine()

        return timely, historical


@celery.task
def get_user_data(username: str):
    cache = Cache(user=username)
    try: 
        runner = Runner(username, cache)
        _ = runner.run()
        _ = runner._fetch._get_avatar()
        # Don't return data, just indicate that it completed successfully
        logging.warning(f"Processing completed successfully for user: {username}")
        return 0
    except Exception as e:
        # Some error occurred
        # just remove the cache so it can be attempted cleanly again
        shutil.rmtree(cache._cache_dir)
        return 1


def get_status(username):
    """Gets the status of the processing.
    Returns a tuple, first index is the step number (0 for fetching, 1 for analysing)"""
    cache = Cache(user=username)
    # Check status of fetching:
    # if not cache._cache_dir.exists():
    #     # Not started, so run background processing
    if not (cache._cache_dir / "meta.json").exists():
        # Processing has started, but nothing has been written yet
        print("STARTING TASK")
        get_user_data.delay(username)
        return 0, 0
    meta = cache.read_meta()
    if not meta["complete"]:
        # get progress
        total_pages = meta["total_pages"]
        if total_pages is None:
            # Processing is still starting
            return 0, 0
        # count the number of cached scrobbles
        cached_scrobbles = len(list(Path(cache.scrobbles_path).glob("*"))) * TRACKS_IN_MEMORY_LIMIT
        total_scrobbles = FETCH_LIMIT_HIGH * total_pages
        print(total_scrobbles, total_scrobbles)
        return 0, int(100 * cached_scrobbles / total_scrobbles)
    # Fetching is completed
    if not (cache._cache_dir / HISTORICAL_META).exists():
        # Processing is not complete
        return 1, 0
    return 1, 100


@app.route("/")
def home():
    return render_template(
        "index.html",
        timely_helpers=TIMELY_TITLES_AND_DESCTIPTIONS,
        historical_helpers=HISTORICAL_TITLES_AND_DESCRIPTIONS,
    )


@app.route("/", methods=["POST"])
def my_form_post():
    username = request.form["username-input"]
    print(f"Getting data for user: {username}")
    return redirect(
        url_for(
            "user",
            username=username,
        )
    )


@app.route("/user/<username>")
def user(username: str):
    step, progress = get_status(username)
    if step != 1 or progress != 100:
        return render_template(
            "holding.html",
            step=step,
            progress=progress
        )

    try:
        cache = Cache(user=username)
        runner = Runner(username, cache)
        timely, historical = runner.run()
        avatar = runner._fetch._get_avatar()
        return render_template(
            "user.html",
            user=username,
            timely=timely,
            timely_helpers=TIMELY_TITLES_AND_DESCTIPTIONS,
            historical=historical,
            historical_helpers=list(
                zip(HISTORICAL_TITLES_AND_DESCRIPTIONS, HISTORICAL_ARTEFACTS_TYPES)
            ),
            avatar=avatar,
        )
    except Exception as e:
        return (
            "<div>Oh no! An unexpected error occurred:  💥 💔 💥\n</div>"
            f"<pre>{get_full_class_name(e)}: {str(e)}</pre>\n"
            f"<pre>{traceback.format_exc()}</pre>"
        )


if __name__ == "__main__":
    app.run()
