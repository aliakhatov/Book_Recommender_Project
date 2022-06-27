import logging.config
import typing
import datetime

import flask
import sqlalchemy
import sqlalchemy.orm
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, TIMESTAMP

logger = logging.getLogger(__name__)

Base: typing.Any = declarative_base()


class TopBooks(Base):
    """Create a data model for the database to be set up for capturing top-rated Books
    """

    __tablename__ = "TopBooks"

    id = Column(Integer, primary_key=True)
    title = Column(String(100), unique=True, nullable=True)
    author = Column(String(100), unique=False, nullable=True)
    year = Column(Integer, unique=False, nullable=True)
    image_l = Column(String(100), unique=False, nullable=True)

    def __repr__(self):
        return f"<TopBooks {self.id, self.title, self.author, self.year, self.image_l}>"


class UserPicks(Base):
    """Create a data model for the database to be set up for capturing songs
    """
    __tablename__ = "UserPicks"

    id = Column(Integer, primary_key=True)
    time = Column(TIMESTAMP(timezone=False), primary_key=False)
    book_title = Column(String(100), unique=False, nullable=True)

    def __repr__(self):
        return f"<UserPicks {self.pick_num, self.book_title}>"


def create_db(engine_string: str) -> None:
    """Create database from the engine string parameter
    Args:
        engine_string (`str`): Engine string
    Returns: None
    """
    engine = sqlalchemy.create_engine(engine_string)

    try:
        Base.metadata.create_all(engine)
    except sqlalchemy.exc.OperationalError as err:
        check_msg = ("You might have a connection error. Please check 2 things: \n"
                     "1. The correctness of your SQLALCHEMY_DATABASE_URI\n"
                     "2. Whether you are connected to Northwestern VPN")

        logger.error("Other error is: %s", err)
        logger.error("%s", check_msg)
    else:
        logger.info("Database created.")


class BookManager:
    """Creates a SQLAlchemy connection to the TopBooks table.

    Args:
        app (:obj:`flask.app.Flask`): Flask app object for when connecting from
            within a Flask app. Optional.
        engine_string (`str`): SQLAlchemy engine string specifying which database
            to write to. Follows the format
    """

    def __init__(self, app: typing.Optional[flask.app.Flask] = None,
                 engine_string: typing.Optional[str] = None):
        if app:
            self.database = SQLAlchemy(app)
            self.session = self.database.session
        elif engine_string:
            engine = sqlalchemy.create_engine(engine_string)
            session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
            self.session = session_maker()
        else:
            raise ValueError(
                "Need either an engine string or a Flask app to initialize")

    def add_picks(self, title: str, time_of_pick: datetime.datetime) -> None:
        """Seeds an existing database with additional books user picked.

        Args:
            title (`str`): book title that's being persisted to the table
            time_of_pick (datetime.datetime): time the user pick has been persisted to rds

        Returns:
            None
        """
        session = self.session
        book = UserPicks(book_title=title, time=time_of_pick)
        session.add(book)
        session.commit()
        logger.info("%s added to database", title)

    def persist_to_rds(self, output_dir: str = None,
                       file_name: str = None) -> None:
        """
        Add the csv data to RDS database
        Args:
            output_dir(`str`): directory where the dataframe in csv form will be saved
            file_name(`str`): file name as specified in yaml file
        Returns: None
        """
        session = self.session

        # file that was read from the full path
        full_path = f"{output_dir}{file_name}"
        data_list = pd.read_csv(full_path).to_dict(orient="records")

        persist_list = []
        for data in data_list:
            persist_list.append(TopBooks(**data))
        num_of_books = len(persist_list)
        try:
            session.add_all(persist_list)
            session.commit()
        except sqlalchemy.exc.OperationalError:
            error_msg = ("You might have encountered connection error. "
                         "Please check following things:\n"
                         "1. SQLALCHEMY_DATABASE_URI was correctly specified in your arguments\n"
                         " 2. You are connected to Northwestern VPN")
            logger.error("%s \n The original error message is: %s",
                         error_msg, sqlalchemy.exc.OperationalError)
        except sqlalchemy.exc.IntegrityError:

            logger.warning("Warning! Please READ: If you are inserting the same table "
                           "twice please make sure\n "
                           "to delete the table by following instructions in the README."
                           "This database does not allow duplicates")
        else:
            logger.info("Top %d books were added to the table", num_of_books)

    def close(self) -> None:
        """Closes SQLAlchemy session

        Returns: None

        """
        self.session.close()
