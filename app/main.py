from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)


@app.route('/')
def index():
  return render_template('style.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80,debug=True)
