FROM continuumio/anaconda3:4.4.0
COPY . /usr/app/
EXPOSE 7005
WORKDIR /usr/app/
RUN pip install -r requirements.txt
CMD python flask_app_UI.py