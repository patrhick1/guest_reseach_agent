"""
Frontend Server for Podcast Guest Research Automation

This script serves the web interface for the Podcast Guest Research Automation System.
It handles template rendering and directly calls the research library functions.
"""

import os
import json
import uuid
import logging
import datetime
import asyncio
import functools
from typing import Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

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

# Log startup information
logger.info("Starting Flask server")

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

@app.route('/')
def index():
    """Render the main research form page"""
    return render_template('index.html')

@app.route('/research-session', methods=['POST'])
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
    
    return jsonify({
        'session_id': session_id,
        'message': 'Research session created successfully'
    })

@app.route('/research-guest', methods=['GET'])
@async_route
async def research_guest():
    """
    Endpoint for researching a podcast guest
    Calls the research function directly without proxying to another service
    """
    try:
        print("DEBUG: === Starting research_guest endpoint processing ===")
        logger.info("=== Starting research_guest endpoint processing ===")
        
        # Get parameters from request
        episode_title = request.args.get('episode_title')
        rss_feed = request.args.get('rss_feed')
        search_guest_name = request.args.get('search_guest_name')
        host_podcast = request.args.get('host_podcast')
        linkedin_url = request.args.get('linkedin_url', '')
        twitter_url = request.args.get('twitter_url', '')
        direct_social = request.args.get('direct_social', 'false').lower() == 'true'
        session_id = request.args.get('session_id')  # Check if a session_id was provided
        
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
    
    print(f"DEBUG: Returning status response with document_url: {document_url != ''}")
    logger.info(f"Returning status response: {response_data}")
    return jsonify(response_data)

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error occurred'
    }), 500

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