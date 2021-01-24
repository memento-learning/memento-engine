import unittest
from flask import Flask, jsonify
from __init__ import create_app

app = Flask(__name__)
from routes import routes_blueprint
app.register_blueprint(routes_blueprint)

class BluePrintTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_sm2(self):
        with app.app_context():
            create_app()
            hist_p = [1,.3,.4,.7,.7,1,1]
            rv = self.app.post('/api/sm2', jsonify(hist_p))
            print(str(rv.data))


if __name__ == '__main__':
    unittest.main()
