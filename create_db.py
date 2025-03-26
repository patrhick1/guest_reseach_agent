"""
WARNING: This script creates a new database and will DELETE any existing data.
Only run this when setting up a new installation or if you want to reset all data.
BACKUP YOUR DATA before running this script if it contains important information.
"""

import os
import sqlite3
import logging
from werkzeug.security import generate_password_hash
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def create_database():
    # Database file path
    instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
    db_path = os.path.join(instance_dir, 'podcast_research.db')
    logger.info(f"Database path: {db_path}")
    
    # Make sure the instance directory exists
    if not os.path.exists(instance_dir):
        os.makedirs(instance_dir)
        logger.info(f"Created instance directory: {instance_dir}")
    
    # Delete the database file if it exists
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.info("Existing database file deleted")
        except PermissionError:
            logger.error("Cannot delete existing database: Permission denied")
            logger.info("Attempting to rename the database file first...")
            try:
                backup_path = f"{db_path}.bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(db_path, backup_path)
                logger.info(f"Renamed existing database to {backup_path}")
            except Exception as e:
                logger.error(f"Failed to rename database: {str(e)}")
                logger.info("Will attempt to use the existing database")
        except Exception as e:
            logger.error(f"Error removing existing database: {str(e)}")
            logger.info("Will attempt to use the existing database")
    
    try:
        # Connect to the database (will create it if it doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logger.info("Connected to database")
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        logger.info("Created users table")
        
        # Create researches table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS researches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            episode_title TEXT,
            guest_name TEXT,
            document_url TEXT,
            linkedin_url TEXT,
            twitter_url TEXT,
            introduction TEXT,
            summary TEXT,
            questions TEXT,
            appearances TEXT,
            status TEXT DEFAULT "pending",
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        logger.info("Created researches table")
        
        # Check if we need to recreate tables with updated schema
        try:
            # Check if bio field exists in users table
            cursor.execute("SELECT bio FROM users LIMIT 1")
            # If we got here, the bio column exists and we need to recreate the table
            logger.info("Detected old schema with bio column, recreating users table...")
            
            # Create a temporary table with the new schema
            cursor.execute('''
            CREATE TABLE users_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Copy data from old table to new
            cursor.execute('''
            INSERT INTO users_new(id, username, email, password_hash, created_at)
            SELECT id, username, email, password_hash, created_at FROM users
            ''')
            
            # Drop old table
            cursor.execute("DROP TABLE users")
            
            # Rename new table
            cursor.execute("ALTER TABLE users_new RENAME TO users")
            
            logger.info("Users table recreated with updated schema")
            
        except sqlite3.OperationalError as e:
            # This is expected if the bio column doesn't exist
            logger.info("Current schema appears to be up-to-date")
        
        # Check if admin user exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            # Create admin user
            admin_password_hash = generate_password_hash('admin123')
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                ('admin', 'admin@example.com', admin_password_hash)
            )
            logger.info("Created admin user")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        logger.info("Database created successfully")
        
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False
    
    return True

if __name__ == "__main__":
    logger.info("Starting database creation process")
    result = create_database()
    if result:
        logger.info("Database created successfully")
    else:
        logger.error("Database creation failed") 