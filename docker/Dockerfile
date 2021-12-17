FROM tensorflow/tensorflow

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY docker/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY math_model ./math_model
COPY templates ./templates
COPY static ./static
COPY symbol_locator.py ./symbol_locator.py
COPY math_parser.py ./math_parser.py
COPY math_solver.py ./math_solver.py
COPY photo_solver.py ./photo_solver.py
COPY app.py ./app.py

CMD [ "python", "./app.py" ]