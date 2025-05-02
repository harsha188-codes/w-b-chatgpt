import json
import datetime
import os
from pathlib import Path
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
USE_SQLITE = False  # Set to True to use SQLite instead of JSON file
USER_DATA_FILE = Path("user_data.json")
DB_FILE = Path("user_data.sqlite")

# Initialize database or JSON file
def initialize_storage():
    if USE_SQLITE:
        # Create SQLite database and tables if they don't exist
        if not DB_FILE.exists():
            conn = sqlite3.connect(str(DB_FILE))
            cursor = conn.cursor()
            
            # Create user_prompts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                prompt_date TEXT NOT NULL,
                prompt_count INTEGER NOT NULL,
                UNIQUE(username, prompt_date)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Created SQLite database at {DB_FILE}")
    else:
        # Create JSON file if it doesn't exist
        if not USER_DATA_FILE.exists():
            with open(USER_DATA_FILE, 'w') as f:
                json.dump({}, f)
            logger.info(f"Created user data file at {USER_DATA_FILE}")

# Get today's date as a string (YYYY-MM-DD)
def get_today():
    return datetime.datetime.now().strftime("%Y-%m-%d")

# Check if user has reached their prompt limit for today
def check_prompt_limit(username: str) -> int:
    today = get_today()
    
    if USE_SQLITE:
        try:
            conn = sqlite3.connect(str(DB_FILE))
            cursor = conn.cursor()
            
            # Query for today's prompt count
            cursor.execute(
                "SELECT prompt_count FROM user_prompts WHERE username = ? AND prompt_date = ?", 
                (username, today)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result is None:
                return 0
            else:
                return result[0]
                
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            return 0
    else:
        try:
            # Load user data from JSON file
            if not USER_DATA_FILE.exists():
                initialize_storage()
                return 0
                
            with open(USER_DATA_FILE, 'r') as f:
                user_data = json.load(f)
            
            # Check if user exists and has data for today
            if username not in user_data or today not in user_data[username]:
                return 0
                
            return user_data[username][today]
            
        except Exception as e:
            logger.error(f"Error reading user data: {e}")
            return 0

# Increment user's prompt count for today
def increment_prompt_count(username: str):
    today = get_today()
    current_count = check_prompt_limit(username)
    new_count = current_count + 1
    
    if USE_SQLITE:
        try:
            conn = sqlite3.connect(str(DB_FILE))
            cursor = conn.cursor()
            
            # Insert or update prompt count
            cursor.execute(
                """
                INSERT INTO user_prompts (username, prompt_date, prompt_count)
                VALUES (?, ?, ?)
                ON CONFLICT(username, prompt_date) 
                DO UPDATE SET prompt_count = ?
                """,
                (username, today, new_count, new_count)
            )
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
    else:
        try:
            # Load existing data
            if not USER_DATA_FILE.exists():
                initialize_storage()
                user_data = {}
            else:
                with open(USER_DATA_FILE, 'r') as f:
                    user_data = json.load(f)
            
            # Update user data
            if username not in user_data:
                user_data[username] = {}
                
            user_data[username][today] = new_count
            
            # Save updated data
            with open(USER_DATA_FILE, 'w') as f:
                json.dump(user_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating user data: {e}")

# Reset all users' prompt counts (useful for testing)
def reset_all_prompt_counts():
    if USE_SQLITE:
        try:
            conn = sqlite3.connect(str(DB_FILE))
            cursor = conn.cursor()
            
            # Delete all records for today
            cursor.execute("DELETE FROM user_prompts WHERE prompt_date = ?", (get_today(),))
            
            conn.commit()
            conn.close()
            logger.info("Reset all prompt counts for today")
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
    else:
        try:
            if USER_DATA_FILE.exists():
                with open(USER_DATA_FILE, 'r') as f:
                    user_data = json.load(f)
                
                # Remove today's data for all users
                today = get_today()
                for username in user_data:
                    if today in user_data[username]:
                        del user_data[username][today]
                
                with open(USER_DATA_FILE, 'w') as f:
                    json.dump(user_data, f, indent=2)
                    
                logger.info("Reset all prompt counts for today")
                
        except Exception as e:
            logger.error(f"Error resetting prompt counts: {e}")

# Function to test the prompt tracker
if __name__ == "__main__":
    initialize_storage()
    
    # Test user
    test_user = "test_user"
    
    # Check initial count
    initial_count = check_prompt_limit(test_user)
    print(f"Initial prompt count for {test_user}: {initial_count}")
    
    # Increment count
    increment_prompt_count(test_user)
    
    # Check updated count
    updated_count = check_prompt_limit(test_user)
    print(f"Updated prompt count for {test_user}: {updated_count}")