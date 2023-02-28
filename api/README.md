# Census Model API

This folder contains the API app for the census library based on FastAPI. The file [`schemas.py`](./schemas.py) defines Pydantic models for API responses, whereas the file [`app.py`](./app.py) defines the endpoints and the FastAPI app.

Altogether, the following methods/endpoints are implemented:

- GET, index(): Welcome page with links to documentation.
- GET, health(): JSON with API and model version is returned. 
- POST, make_prediction(): we pass a JSON with features and the inferred target is returned.

To use this app *LOCALLY*, go to the folder where config.yaml is located and start in the terminal the Uvicorn ASGI web server:

```bash
$ uvicorn api.app:app --reload
```

Then, we open the browser in [http://127.0.0.1:8000](http://127.0.0.1:8000) and we can start using the API. We have the following endpoints:

- [http://127.0.0.1:8000](http://127.0.0.1:8000): welcome page (index).
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs): documentation with testing interfaces.
- [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health): JSON with API and model version is returned.
- [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) ...: we pass a JSON with features and the inferred target is returned; see documentation for examples.

To use this app *HOSTED ON A CLOUD SERVICE*, the web server must be run differently.
For instance, if we want to deploy to to HEROKU, we need a `Procfile` with this line:

    web: uvicorn api.app:app --host=0.0.0.0 --port=${PORT:-5000}

