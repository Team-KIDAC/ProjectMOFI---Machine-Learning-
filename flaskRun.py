from flask import Flask, jsonify
from MOFIconsoleTest import *
import json

app = Flask(__name__)


@app.route('/identify/<imgname>', methods=['GET'])
def index(imgname):
    identify = imagePath("https://mofiblob.blob.core.windows.net/mofiimages/"+imgname+".jpg")
    jsonFormat = "{\"id\":\""+identify+"\"}"
    jObg = json.loads(jsonFormat)
    return jObg


if __name__ == '__main__':
    app.run(debug=True)
