FROM python:3.11

WORKDIR /app

ENV PYTHONUNBUFFERED True

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt --timeout 600

ENV PORT=8080

EXPOSE ${PORT}

COPY images /app/images

COPY . ./app

ENTRYPOINT streamlit run --server.port ${PORT} app/app.py –-server.address=0.0.0.0
