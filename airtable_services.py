import os
import logging
from pyairtable import Api
from dotenv import load_dotenv

# Load .env variables to access your Airtable credentials
load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retrieve your Airtable credentials from environment variables
AIRTABLE_API_KEY = os.getenv('AIRTABLE_APPLICATION_TOKEN')
PODCAST_BASE_ID = os.getenv('AIRTABLE_PODCAST_BASE')


class PodcastService:

    def __init__(self):
        """
        Initialize the PodcastService by connecting to Airtable using 
        the environment credentials for the podcast base.
        """
        try:
            self.api_key = AIRTABLE_API_KEY
            self.base_id = PODCAST_BASE_ID

            self.api = Api(self.api_key)

            # Store references to tables in a dictionary for easy access
            self.tables = {
                'Guest Research bot': self.api.table(self.base_id, 'Guest Research bot'),
            }
            logger.info("PodcastService initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize PodcastService: {e}")
            raise

    def get_table(self, table_name):
        """
        Retrieve a table object by name from the base.

        Args:
            table_name (str): Name of the table to fetch.

        Returns:
            pyairtable.Table: The table object.

        Raises:
            ValueError: If the table does not exist.
        """
        table = self.tables.get(table_name)
        if not table:
            logger.error(f"Table '{table_name}' does not exist.")
            raise ValueError(f"Table '{table_name}' does not exist in the base.")
        return table

    def get_record(self, table_name, record_id):
        """
        Retrieve a record by ID from a specified table.

        Args:
            table_name (str): The name of the table.
            record_id (str): The ID of the record to retrieve.

        Returns:
            dict: The record data.
        """
        try:
            table = self.get_table(table_name)
            record = table.get(record_id)
            return record
        except Exception as e:
            logger.error(f"Error retrieving record {record_id} from table '{table_name}': {e}")
            return None

    def update_record(self, table_name, record_id, fields):
        """
        Update a record in the specified table with new field values.

        Args:
            table_name (str): The name of the table.
            record_id (str): The ID of the record to update.
            fields (dict): A dictionary of fields to update.

        Returns:
            dict: The updated record data or None on failure.
        """
        try:
            table = self.get_table(table_name)
            updated_record = table.update(record_id, fields)
            logger.debug(f"Updated record {record_id} in table '{table_name}' with {fields}")
            return updated_record
        except Exception as e:
            logger.error(f"Error updating record {record_id} in '{table_name}': {e}")
            return None

    def create_record(self, table_name, fields):
        """
        Create a new record in a specified table.

        Args:
            table_name (str): The name of the table.
            fields (dict): A dictionary of fields to set in the new record.

        Returns:
            dict: The newly created record data or None on failure.
        """
        try:
            table = self.get_table(table_name)
            new_record = table.create(fields)
            logger.debug(f"Created new record in table '{table_name}' with {fields}")
            return new_record
        except Exception as e:
            logger.error(f"Error creating record in '{table_name}': {e}")
            return None

    def search_records(self, table_name, formula=None, view=None):
        """
        Search for records in a specified table using an Airtable filter formula.

        Args:
            table_name (str): The name of the table.
            formula (str): The Airtable filter formula (e.g., '{Name}="John"').
            view (str): (Optional) A specific view to search in.

        Returns:
            list: A list of matching records.
        """
        try:
            table = self.get_table(table_name)
            params = {}
            if formula:
                params['formula'] = formula
            if view:
                params['view'] = view
            records = table.all(**params)
            logger.debug(f"Found {len(records)} records in table '{table_name}' with formula '{formula}'")
            return records
        except Exception as e:
            logger.error(f"Error searching records in '{table_name}' with formula '{formula}': {e}")
            return []

    def get_records_from_view(self, table_name, view):
        """
        Retrieve all records from a specific view in a table.

        Args:
            table_name (str): The name of the table.
            view (str): The name or ID of the view.

        Returns:
            list: A list of records from the specified view.
        """
        try:
            table = self.get_table(table_name)
            records = table.all(view=view)
            logger.debug(f"Retrieved {len(records)} records from view '{view}' in table '{table_name}'")
            return records
        except Exception as e:
            logger.error(f"Error retrieving records from view '{view}' in '{table_name}': {e}")
            return []

