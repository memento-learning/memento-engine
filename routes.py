from flask import Flask, Blueprint, request, jsonify
from flask_cors import cross_origin
import json

routes_blueprint = Blueprint('routes', __name__)

@routes_blueprint.route('/api/sm2', methods=['POST'])
def runSM2():
    I = 1
    EF = 2.5
    n = 0
    hist_p = json.loads(request.data)
    for p in hist_p:
        if p == -1:
            break
        q = p*5.0
        if q >= 3:
            if n == 0:
                I = 1
            elif n == 1:
                I = 6
            else:
                I = I * EF

            if I > 274.0:
                I = 274.0
            if I < 1.0:
                I = 1.0
            
            EF = EF + (0.1 - (5-q) * (.08 + (5-q) * 0.02))
            if EF < 1.3:
                EF = 1.3
            n += 1
        else:
            n = 0
            I = 1
    return str(I)