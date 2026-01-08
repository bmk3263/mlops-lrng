from flask import Flask, request, jsonify

app = Flask(__name__)   # ✅ app is defined HERE

@app.route("/health", methods=["GET"])
def hello():
    return jsonify({"status" : "OK"}) 

@app.route("/predict", methods=["POST"])
def predict():
    data= request.json
    age=data.get("age")
    salary=data.get("salary")
    prediction="yes" if age>40000 else "No"
    return jsonify({"prediction": prediction})
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="6000")

