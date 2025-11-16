# 1. Use an official Python runtime as a parent image

FROM python:3.10-slim



# 2. Set the working directory in the container

WORKDIR /code



# 3. Copy the requirements file and install dependencies

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt



# 4. Copy the app code and the trained model into the container

COPY ./app /code/app

COPY ./model /code/model



# 5. Expose the port the app runs on

EXPOSE 8000



# 6. Run the app using uvicorn

# We bind to 0.0.0.0 to make it accessible from outside the container

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]