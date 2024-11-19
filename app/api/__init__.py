# This can be empty or initialize any API-specific configurations
from flask import Blueprint

api = Blueprint('api', __name__)

from . import endpoints