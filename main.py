# from flask import Flask
# from app.api.endpoints import api
# import mlflow

# app = Flask(__name__)
# app.register_blueprint(api, url_prefix='/api/v1')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)

#second try 

# from app import create_app

# app = create_app()

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)