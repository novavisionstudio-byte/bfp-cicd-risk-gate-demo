"""Minimal Flask API wrapping the calculator library."""
from __future__ import annotations
from flask import Flask, jsonify, request
from . import calculator as calc

app = Flask(__name__)


@app.post("/add")
def add():
    d = request.get_json()
    return jsonify(result=calc.add(d["a"], d["b"]))


@app.post("/subtract")
def subtract():
    d = request.get_json()
    return jsonify(result=calc.subtract(d["a"], d["b"]))


@app.post("/multiply")
def multiply():
    d = request.get_json()
    return jsonify(result=calc.multiply(d["a"], d["b"]))


@app.post("/divide")
def divide():
    d = request.get_json()
    try:
        return jsonify(result=calc.divide(d["a"], d["b"]))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.post("/factorial")
def factorial():
    d = request.get_json()
    try:
        return jsonify(result=calc.factorial(d["n"]))
    except ValueError as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(debug=True)
