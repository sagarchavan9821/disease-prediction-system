import pandas as pd
from sqlalchemy import create_engine
import os

# works both locally and in Docker
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:9821@localhost:5432/project_db"  
)

engine = create_engine(DATABASE_URL)

query = "SELECT * FROM database"

df = pd.read_sql(query, engine)
