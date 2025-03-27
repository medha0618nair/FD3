FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Use environment variable for port
ENV PORT=10000

EXPOSE 10000

# Use shell form to properly interpolate the PORT variable
CMD gunicorn --bind 0.0.0.0:${PORT} app:app 