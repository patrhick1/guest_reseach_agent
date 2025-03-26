"""
Frontend Server for Podcast Guest Research Automation

This script serves the web interface for the Podcast Guest Research Automation System.
It handles template rendering and directly calls the research library functions.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
import asyncio
import functools
from typing import Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the research state manager
from research_state import ResearchStateManager, GuestResearchState
state_manager = ResearchStateManager()

# Import the research function from main.py
from main import research_podcast_guest

# Create Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Set a secret key for session management
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Configure SQLAlchemy - using absolute path for database
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
if not os.path.exists(instance_path):
    try:
        os.makedirs(instance_path)
        logger.info(f"Created instance directory: {instance_path}")
    except Exception as e:
        logger.error(f"Error creating instance directory: {str(e)}")

# Use absolute path for database file
db_path = os.path.join(instance_path, 'podcast_research.db')
db_uri = f"sqlite:///{db_path}"
logger.info(f"Database URI: {db_uri}")

# Check if directory is writable
if os.access(instance_path, os.W_OK):
    logger.info(f"Instance directory is writable: {instance_path}")
else:
    logger.error(f"Instance directory is NOT writable: {instance_path}")
    # Try to make it writable
    try:
        os.chmod(instance_path, 0o777)
        logger.info(f"Changed permissions on instance directory")
    except Exception as e:
        logger.error(f"Could not change permissions: {str(e)}")

app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    __tablename__ = 'users'  # Explicitly set table name
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    researches = db.relationship('Research', backref='user', lazy=True)

# Research History Model
class Research(db.Model):
    __tablename__ = 'researches'  # Explicitly set table name
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)  # Updated foreign key
    episode_title = db.Column(db.String(200))
    guest_name = db.Column(db.String(100))
    document_url = db.Column(db.String(500))
    linkedin_url = db.Column(db.String(500))
    twitter_url = db.Column(db.String(500))
    introduction = db.Column(db.Text)
    summary = db.Column(db.Text)
    questions = db.Column(db.Text)
    appearances = db.Column(db.Text)
    status = db.Column(db.String(20), default="pending")  # pending, completed, failed
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

# User Quota Model for rate limiting
class UserQuota(db.Model):
    __tablename__ = 'user_quotas'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, default=lambda: datetime.now(timezone.utc).date())
    count = db.Column(db.Integer, default=0)
    # Add unique constraint for user_id and date
    __table_args__ = (db.UniqueConstraint('user_id', 'date', name='uq_user_date'),)

# Create all database tables
with app.app_context():
    try:
        # Try to connect to database
        engine = db.engine
        connection = engine.connect()
        connection.close()
        logger.info("Database connection successful")
        
        # Create tables if they don't exist
        db.create_all()
        logger.info("Database tables created or verified")
        
        # Create admin user if it doesn't exist
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('admin123')
            )
            db.session.add(admin)
            db.session.commit()
            logger.info("Admin user created")
        else:
            logger.info("Admin user already exists")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        logger.error("Trying alternative approach with SQLite directly...")
        
        try:
            import sqlite3
            # Ensure the directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Try to connect and create tables directly with SQLite
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
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
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Check if admin user exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
            admin_count = cursor.fetchone()[0]
            
            if admin_count == 0:
                # Create admin user
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                    ('admin', 'admin@example.com', generate_password_hash('admin123'))
                )
                logger.info("Admin user created (SQLite direct)")
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully with SQLite direct approach")
            
        except Exception as e2:
            logger.error(f"Error with direct SQLite approach: {str(e2)}")
            logger.error("Please check file permissions and path accessibility")

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/')
def landing():
    # If user is already logged in, redirect to dashboard
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/home')
@login_required
def index():
    """Render the main research form page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not all([username, email, password, confirm_password]):
            return render_template('register.html', error='All required fields must be completed')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        # Add password complexity validation
        if len(password) < 8:
            return render_template('register.html', error='Password must be at least 8 characters long')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already exists')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        # Log the user in
        session['user'] = username
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/research-session', methods=['POST'])
@login_required
def create_research_session():
    """
    Create a new research session without starting the research yet
    This is useful for tracking UI state
    """
    data = request.get_json()
    episode_title = data.get('episode_title')
    request_id = data.get('request_id')  # Now optional
    rss_feed = data.get('rss_feed')
    search_guest_name = data.get('search_guest_name')
    host_podcast = data.get('host_podcast')
    linkedin_url = data.get('linkedin_url', '')
    twitter_url = data.get('twitter_url', '')
    direct_social = data.get('direct_social', False)
    
    # Get user for quota check
    user = User.query.filter_by(username=session['user']).first()
    
    # Just check quota without incrementing it
    # We'll increment only when the actual research starts
    today = datetime.now(timezone.utc).date()
    quota = UserQuota.query.filter_by(user_id=user.id, date=today).first()
    daily_limit = 10
    
    if quota and quota.count >= daily_limit:
        # User has reached their daily limit
        return jsonify({
            'error': f'Daily research limit reached ({quota.count}/{daily_limit}). Please try again tomorrow.',
            'quota_exceeded': True,
            'current_count': quota.count,
            'max_count': daily_limit
        }), 429  # 429 Too Many Requests
    
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Validate required data
    if direct_social:
        # For direct social media research, we need a guest name and at least one social URL
        if not search_guest_name:
            return jsonify({
                'error': 'Missing required parameter: search_guest_name is required for social media research'
            }), 400
            
        if not (linkedin_url or twitter_url):
            return jsonify({
                'error': 'At least one social media URL (linkedin_url or twitter_url) is required for social media research'
            }), 400
            
        # If using direct social approach, we'll use a generated episode title if none provided
        if not episode_title:
            episode_title = f"Research for {search_guest_name}"
    else:
        # For podcast-based research, episode title is required
        if not episode_title:
            return jsonify({
                'error': 'Missing required parameter: episode_title is required for podcast-based research'
            }), 400
    
    # Create a new session
    session_id = state_manager.create_session(
        episode_title=episode_title,
        request_id=request_id or f"web-{uuid.uuid4()}",  # Generate if None
        rss_feed=rss_feed,
        search_guest_name=search_guest_name,
        host_podcast=host_podcast,
        linkedin_url=linkedin_url,
        twitter_url=twitter_url,
        direct_social=direct_social
    )
    
    # Save the session ID in the user's session
    session['research_session_id'] = session_id
    
    # Save research history
    user = User.query.filter_by(username=session['user']).first()
    research = Research(
        session_id=session_id,
        user_id=user.id,
        episode_title=episode_title
    )
    db.session.add(research)
    db.session.commit()
    
    return jsonify({
        'session_id': session_id,
        'message': 'Research session created successfully'
    })

# Define a wrapper to allow async functions to work with Flask
def async_route(route_function):
    """Wrapper for Flask routes to allow async functions"""
    @functools.wraps(route_function)
    def wrapper(*args, **kwargs):
        # Create an event loop if not already initialized
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the route function in the event loop
        return loop.run_until_complete(route_function(*args, **kwargs))
    return wrapper

@app.route('/research-guest', methods=['GET'])
@login_required
@async_route
async def research_guest():
    """
    Endpoint for researching a podcast guest
    Calls the research function directly without proxying to another service
    """
    try:
        print("DEBUG: === Starting research_guest endpoint processing ===")
        logger.info("=== Starting research_guest endpoint processing ===")
        
        # Get user for quota check
        user = User.query.filter_by(username=session['user']).first()
        
        # Check if a session ID was provided - if so, we don't need to check quota as it was already checked
        session_id = request.args.get('session_id')
        
        # Always check and increment quota when starting an actual research
        # This is the point where we actually consume a quota
        allowed, current_count, max_count = check_and_increment_quota(user.id)
        if not allowed:
            logger.warning(f"User {user.username} has reached their daily quota ({current_count}/{max_count})")
            return jsonify({
                'error': f'Daily research limit reached ({current_count}/{max_count}). Please try again tomorrow.',
                'quota_exceeded': True,
                'current_count': current_count,
                'max_count': max_count
            }), 429  # 429 Too Many Requests
        
        # Get parameters from request
        episode_title = request.args.get('episode_title')
        rss_feed = request.args.get('rss_feed')
        search_guest_name = request.args.get('search_guest_name')
        host_podcast = request.args.get('host_podcast')
        linkedin_url = request.args.get('linkedin_url', '')
        twitter_url = request.args.get('twitter_url', '')
        direct_social = request.args.get('direct_social', 'false').lower() == 'true'
        
        # Generate a request_id 
        request_id = f"web-{uuid.uuid4()}"
        
        print(f"DEBUG: Received request with params: episode_title='{episode_title}'")
        logger.info(f"Received request with params: episode_title='{episode_title}', request_id='{request_id}', "
                   f"rss_feed='{rss_feed}', search_guest_name='{search_guest_name}', host_podcast='{host_podcast}', "
                   f"linkedin_url='{linkedin_url}', twitter_url='{twitter_url}', direct_social={direct_social}")
        
        logger.info(f"Generated request_id: {request_id}")
        
        # Validate required parameters
        # direct_social is already defined above, no need to reassign
        
        if direct_social:
            # For direct social media research, we need a guest name and at least one social URL
            if not search_guest_name:
                logger.warning("Missing required parameter: search_guest_name is required for social media research")
                return jsonify({
                    'error': 'Missing required parameter: search_guest_name is required for social media research'
                }), 400
                
            if not (linkedin_url or twitter_url):
                logger.warning("Missing required parameter: At least one social media URL is required")
                return jsonify({
                    'error': 'At least one social media URL (linkedin_url or twitter_url) is required for social media research'
                }), 400
                
            # If using direct social approach, we'll use a generated episode title if none provided
            if not episode_title:
                episode_title = f"Research for {search_guest_name}"
                logger.info(f"Generated episode title: {episode_title}")
        else:
            # For podcast-based research, episode title is required
            if not episode_title:
                logger.warning("Missing required parameter: episode_title")
                print("DEBUG: Missing required parameter: episode_title")
                return jsonify({
                    'error': 'Missing required parameter: episode_title is required for podcast-based research'
                }), 400
        
        # Get or create a session
        if session_id:
            # Use existing session if provided
            logger.info(f"Using existing session with ID: {session_id}")
            # Verify the session exists
            if not state_manager.get_state(session_id):
                logger.warning(f"Session {session_id} not found")
                return jsonify({
                    'error': f'Session {session_id} not found'
                }), 404
        else:
            # Create a session to track this request
            logger.info("Creating new session to track request")
            session_id = state_manager.create_session(
                episode_title=episode_title,
                request_id=request_id,
                rss_feed=rss_feed,
                search_guest_name=search_guest_name,
                host_podcast=host_podcast,
                linkedin_url=linkedin_url,
                twitter_url=twitter_url,
                direct_social=direct_social
            )
            logger.info(f"Created session with ID: {session_id}")
        
        # Get the state from the session manager
        state = state_manager.get_state(session_id)
        
        # Update session status
        state_manager.update_state(session_id, {"status": "in_progress"})
        
        # Call the research function directly
        logger.info("Calling research_podcast_guest function")
        # Determine entry point based on parameters
        entry_point = None
        if direct_social:
            logger.info("Using direct social entry point for workflow")
            # If we have direct social media URLs, we can start from retrieving social content
            entry_point = "retrieve_social_content"
        elif rss_feed and search_guest_name:
            logger.info("Using RSS feed entry point for workflow")
            entry_point = "get_episode_from_rss"
            
        # Need to run the async function in a way that Flask can handle
        results = await research_podcast_guest(state, session_id, entry_point)
        logger.info("Research completed successfully")
        
        # Update session with results
        logger.info(f"Updating session with completed status")
        state_manager.update_state(session_id, {
            "status": "completed",
            "guest_name": results.get("guest", ""),
            "linkedin_url": results.get("linkedin", ""),
            "document_url": results.get("document_url", "")
        })
        
        # Add session ID to the response
        response_data = results
        response_data['session_id'] = session_id
        
        # Verify the session status after update
        current_session = state_manager.get_state(session_id)
        logger.info(f"Current session status after update: {current_session.get('status', 'unknown')}")
        logger.info(f"Document URL in session: {current_session.get('document_url', 'none')}")
        
        # Update research database record
        research = Research.query.filter_by(session_id=session_id).first()
        if research:
            research.status = "completed"
            research.guest_name = results.get("guest", "")
            research.document_url = results.get("document_url", "")
            research.linkedin_url = results.get("linkedin", "")
            research.introduction = results.get("introduction", "")
            research.summary = results.get("summary", "")
            research.questions = results.get("question", "")
            research.appearances = results.get("appearance", "")
            db.session.commit()
            logger.info(f"Updated research record in database for session {session_id}")
        
        print(f"DEBUG: Returning API response with session_id: {session_id}")
        print(f"DEBUG: Response contains document_url: {'document_url' in response_data}")
        logger.info("Returning API response with added session_id")
        logger.info("=== Completed research_guest endpoint processing ===")
        return jsonify(response_data)
        
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
        logger.info("=== Failed research_guest endpoint processing (Unexpected Error) ===")
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500

@app.route('/research-status/<session_id>', methods=['GET'])
@login_required
def get_research_status(session_id):
    """Get the status of a research session"""
    print(f"DEBUG: Getting status for session: {session_id}")
    logger.info(f"Getting status for session: {session_id}")
    
    session_data = state_manager.get_state(session_id)
    if not session_data:
        print(f"DEBUG: Session {session_id} not found")
        logger.warning(f"Session {session_id} not found")
        return jsonify({
            'error': f'Session {session_id} not found'
        }), 404
    
    status = session_data.get('status', 'unknown')
    document_url = session_data.get('document_url', '')
    
    print(f"DEBUG: Session {session_id} status: {status}")
    print(f"DEBUG: Session {session_id} document URL: {document_url}")
    logger.info(f"Session {session_id} status: {status}")
    logger.info(f"Session {session_id} document URL: {document_url}")
    
    response_data = {
        'session_id': session_id,
        'status': status,
        'guest': session_data.get('guest_name', ''),
        'linkedin': session_data.get('linkedin_url', ''),
        'document_url': document_url,
        'introduction': session_data.get('introduction', ''),
        'summary': session_data.get('summary', ''),
        'question': session_data.get('question', ''),
        'appearance': session_data.get('appearance', '')
    }
    
    # Update database record if status is completed
    if status == 'completed':
        research = Research.query.filter_by(session_id=session_id).first()
        if research:
            research.status = status
            research.guest_name = response_data['guest']
            research.document_url = document_url
            research.linkedin_url = response_data['linkedin']
            research.introduction = response_data['introduction']
            research.summary = response_data['summary']
            research.questions = response_data['question']
            research.appearances = response_data['appearance']
            db.session.commit()
            logger.info(f"Updated research record in database for session {session_id}")
    
    print(f"DEBUG: Returning status response with document_url: {document_url != ''}")
    logger.info(f"Returning status response: {response_data}")
    return jsonify(response_data)

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.filter_by(username=session['user']).first()
    # Filter out connectivity test entries that might confuse users
    researches = Research.query.filter_by(user_id=user.id).filter(
        ~Research.episode_title.like('connectivity-test-%')  # Filter out connectivity tests
    ).order_by(Research.created_at.desc()).all()
    return render_template('dashboard.html', researches=researches)

@app.route('/research/<int:research_id>')
@login_required
def view_research(research_id):
    user = User.query.filter_by(username=session['user']).first()
    research = Research.query.filter_by(id=research_id, user_id=user.id).first_or_404()
    
    # Extract document ID for download link if it exists
    doc_id = None
    if research.document_url and '/d/' in research.document_url:
        try:
            doc_id = research.document_url.split('/d/')[1].split('/')[0]
        except:
            pass
    
    return render_template('research_detail.html', research=research, doc_id=doc_id)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.filter_by(username=session['user']).first()
    
    if request.method == 'POST':
        # Update user info
        user.email = request.form.get('email', user.email)
        
        # Check if password is being updated
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if current_password and new_password and confirm_password:
            if not check_password_hash(user.password_hash, current_password):
                return render_template('profile.html', user=user, error='Current password is incorrect')
                
            if new_password != confirm_password:
                return render_template('profile.html', user=user, error='New passwords do not match')
                
            if len(new_password) < 8:
                return render_template('profile.html', user=user, error='Password must be at least 8 characters')
                
            user.password_hash = generate_password_hash(new_password)
        
        db.session.commit()
        return render_template('profile.html', user=user, success='Profile updated successfully')
    
    return render_template('profile.html', user=user)

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

@app.route('/diagnostics')
def diagnostics():
    """Diagnostic route to check database connectivity"""
    results = {
        'instance_dir': instance_path,
        'instance_exists': os.path.exists(instance_path),
        'instance_writable': os.access(instance_path, os.W_OK),
        'db_path': db_path,
        'db_exists': os.path.exists(db_path),
    }
    
    # Check database connectivity
    try:
        # Try to connect to the database and run a simple query
        with app.app_context():
            admin_count = User.query.filter_by(username='admin').count()
            results['db_connection'] = 'success'
            results['admin_count'] = admin_count
    except Exception as e:
        results['db_connection'] = 'failed'
        results['db_error'] = str(e)
    
    return jsonify(results)

# Daily research quota management
def check_and_increment_quota(user_id, daily_limit=10):
    """
    Check if a user has reached their daily research quota and increment if not
    
    Args:
        user_id: The user's ID
        daily_limit: Maximum researches allowed per day (default: 10)
        
    Returns:
        tuple: (allowed, current_count, max_count)
            - allowed: True if the user is under quota, False otherwise
            - current_count: Current number of researches done today
            - max_count: Maximum allowed researches per day
    """
    today = datetime.now(timezone.utc).date()
    
    # Find today's quota record for this user
    quota = UserQuota.query.filter_by(user_id=user_id, date=today).first()
    
    if not quota:
        # First research of the day
        quota = UserQuota(user_id=user_id, date=today, count=1)
        db.session.add(quota)
        db.session.commit()
        return True, 1, daily_limit
        
    if quota.count >= daily_limit:
        # User has reached their daily limit
        return False, quota.count, daily_limit
        
    # User is under quota, increment count
    quota.count += 1
    db.session.commit()
    return True, quota.count, daily_limit

@app.route('/user-quota')
@login_required
def get_user_quota():
    """Get the user's current quota information"""
    user = User.query.filter_by(username=session['user']).first()
    daily_limit = 10  # Changed from 5 to 10 to match check_and_increment_quota default
    
    today = datetime.now(timezone.utc).date()
    quota = UserQuota.query.filter_by(user_id=user.id, date=today).first()
    
    current_count = quota.count if quota else 0
    remaining = daily_limit - current_count
    
    return jsonify({
        'daily_limit': daily_limit,
        'used_today': current_count,
        'remaining': remaining,
        'reset_at': (datetime.combine(today, datetime.min.time()) + timedelta(days=1)).isoformat()
    })

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Check if we're running on Replit
    if os.environ.get('REPL_ID'):
        # Running on Replit - use 0.0.0.0 to make it accessible
        print(f"Starting server on Replit at port {port}")
        app.run(host='0.0.0.0', port=port)
    else:
        # Local development
        print(f"Starting server on localhost at port {port}")
        app.run(port=port, debug=True) 