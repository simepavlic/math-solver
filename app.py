from flask import Flask, render_template, request
from photo_solver import solve_photo
import numpy as np
import cv2

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    filestr = request.files['file'].read()
    npimg = np.fromstring(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    solution = solve_photo(image)
    return render_template('index.html', solution=solution)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
