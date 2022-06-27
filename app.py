import logging.config
import sqlite3
import traceback

import joblib
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import sqlalchemy.exc
import yaml
from flask import Flask, render_template, request
from src.books_manager import BookManager, TopBooks, UserPicks
from datetime import datetime
from src.model import recommend_books, extract_info

# For setting up the Flask-SQLAlchemy database session
# from src.add_songs import Tracks, TrackManager

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug(
    'Web app should be viewable at %s:%s if docker run command maps local '
    'port to the same port as configured for the Docker container '
    'in config/flaskconfig.py (e.g. `-p 5000:5000`). Otherwise, go to the '
    'port defined on the left side of the port mapping '
    '(`i.e. -p THISPORT:5000`). If you are running from a Windows machine, '
    'go to 127.0.0.1 instead of 0.0.0.0.', app.config["HOST"]
    , app.config["PORT"])

with open("config/config.yaml", "r") as f:
    yaml_config = yaml.safe_load(f)

# Initialize the database session
book_manager = BookManager(app)
# source the files from the given directory in yaml config
book_matrix = pd.read_csv(yaml_config["book_matrix_directory"])
clean_df = pd.read_csv(yaml_config["final_file_directory"])
db = SQLAlchemy(app)


@app.route('/')
def starter():
    """Main view that lists songs in the database.
    Returns:
        Rendered html template
    """
    try:
        top_books = book_manager.session.query(TopBooks).limit(app.config["MAX_ROWS_SHOW"]).all()

        logger.debug("starter page accessed")

    except sqlite3.OperationalError as e:
        logger.error(
            "Error page returned. Not able to query local sqlite database: %s."
            " Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')
    except sqlalchemy.exc.OperationalError as e:
        logger.error(
            "Error page returned. Not able to query MySQL database: %s. "
            "Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')

    except:
        traceback.print_exc()
        logger.error("Not able to display tracks, error page returned")
        return render_template('error.html')

    else:
        return render_template("starter.html", book_list=top_books)


@app.route('/', methods=['POST'])
def add_entry():
    """View that process a POST with the user input of books
    Returns:
        Renders to results page if successful, to error page if not
    """
    try:

        response_list = []
        for response in request.form.keys():
            if response == "title":
                continue
            now = datetime.now()
            book_manager.add_picks(response, now)
            response_list.append(response)

        logger.info("User Response is parsed")
        trained_model = joblib.load(yaml_config['model']['train_model_path'], mmap_mode='r')
        recommended_books = recommend_books(book_matrix,
                                            response_list,
                                            trained_model)

        book_list = extract_info(recommended_books, clean_df)
        logger.info("Extra information on recommended books is extracted")
        logger.info("Rendering results.html template")

        return render_template('results.html', book_dict=book_list)

    except sqlite3.OperationalError as e:
        logger.error(
            "Error page returned. Not able to add song to local sqlite "
            "database: %s. Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')
    except sqlalchemy.exc.OperationalError as e:
        logger.error(
            "Error page returned. Not able to add song to MySQL database: %s. "
            "Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"],
            host=app.config["HOST"])
