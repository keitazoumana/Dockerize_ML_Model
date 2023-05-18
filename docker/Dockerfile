FROM python:3.9-slim 

WORKDIR /app 

RUN pip install pipenv 

COPY ["Pipfile", "Pipfile.lock", "./"]

COPY app.py . 

COPY models models 
COPY vectors vectors 
COPY static static 
COPY templates templates 

RUN pipenv install --deploy --system 
EXPOSE 9696 

CMD ["python", "app.py"]