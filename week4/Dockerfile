FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

RUN  pipenv install --deploy

COPY homework-week4.py week4.py

# do stuff here
ENTRYPOINT ["pipenv", "run", "python", "week4.py" ]