"""
WARNING: This script creates a new database and will DELETE any existing data.
Only run this when setting up a new installation or if you want to reset all data.
BACKUP YOUR DATA before running this script if it contains important information.
"""

import os
import sqlite3
import logging
import sys

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("database_check.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_users_table():
    """Check specifically for issues with the users table schema"""
    # Find the database file path
    instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
    db_path = os.path.join(instance_dir, 'podcast_research.db')
    
    # Print full path for debugging
    abs_path = os.path.abspath(db_path)
    logger.info(f"Looking for database at: {abs_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return False
    
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get users table schema
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        
        logger.info(f"Users table has {len(columns)} columns:")
        for col in columns:
            logger.info(f"  - {col[1]} ({col[2]})")
        
        # Check if bio column exists
        has_bio = any(col[1] == 'bio' for col in columns)
        if has_bio:
            logger.warning("The 'bio' column still exists in the users table!")
        else:
            logger.info("The 'bio' column has been successfully removed from the users table.")
            
        # Get sample user data
        cursor.execute("SELECT id, username, email FROM users LIMIT 5")
        users = cursor.fetchall()
        
        if users:
            logger.info(f"Found {len(users)} users in the database:")
            for user in users:
                logger.info(f"  - User ID: {user[0]}, Username: {user[1]}, Email: {user[2]}")
        else:
            logger.warning("No users found in the database.")
            
        # Close the connection
        conn.close()
        logger.info("Database check completed")
        
        return not has_bio  # Return True if bio column does not exist
        
    except Exception as e:
        logger.error(f"Error checking database: {str(e)}")
        return False

if __name__ == "__main__":
    result = check_users_table()
    print(f"\nSchema check result: {'✓ PASSED' if result else '✗ FAILED'}")
    print(f"The user table structure is {'correct' if result else 'incorrect'}")
    print(f"Next step: {'You can run the server now' if result else 'Fix the database schema issues'}") 