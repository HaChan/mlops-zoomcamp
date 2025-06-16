FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install --no-cache-dir pipenv==2024.2.0

COPY Pipfile Pipfile.lock ./

RUN pipenv install
RUN pipenv sync

COPY starter.py ./

CMD ["pipenv", "run", "python", "starter.py", "--year", "2023", "--month", "5"]
