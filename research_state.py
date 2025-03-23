"""
Research State Manager for Podcast Guest Research

This module defines state management classes and functions 
that can be imported by both main.py and server.py.
"""

import uuid
import logging
import datetime
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GuestIdentificationState(TypedDict):
    """Basic identification of a podcast guest"""
    guest: str
    guest_unique_element: str
    linkedin: Optional[str]
    twitter: Optional[str]

class GuestResearchState(TypedDict):
    """State container for guest research workflow"""
    host_podcast: str
    host_podcast_description: str  # New field to store podcast description
    rss_feed: str
    search_guest_name: str
    episode_title: str
    episodes: list
    episode_description: str
    guest_name: str
    guest_company: str
    transcript: str
    is_linkedin_url: bool
    is_twitter_url: bool
    linkedin_url: str
    twitter_url: str
    linkedIn_post: str
    linkedin_profile: str
    twitter_post: str
    guest_unique_element: str
    guest_reason: str
    report: str
    document_url: str
    request_id: str
    introduction: str
    summary: str
    question: str
    appearance: str
    created_at: str  # Timestamp when the session was created
    direct_social: bool
    
class ResearchStateManager:
    """
    Manages the state for podcast guest research sessions.
    This class can be imported and used by both main.py and server.py.
    """
    
    def __init__(self):
        """Initialize the state manager"""
        self.active_sessions = {}
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, episode_title: str, request_id: str, 
                      rss_feed: str = None, search_guest_name: str = None,
                      host_podcast: str = None, linkedin_url: str = None,
                      twitter_url: str = None, direct_social: bool = False) -> str:
        """
        Create a new research session and return the session ID
        
        Args:
            episode_title: The podcast episode title
            request_id: Unique request identifier
            rss_feed: Optional RSS feed URL
            search_guest_name: Optional guest name for RSS feed search
            host_podcast: Optional host podcast name
            linkedin_url: Optional LinkedIn profile URL
            twitter_url: Optional Twitter/X profile URL
            direct_social: Whether this is a direct social media research
            
        Returns:
            session_id: Unique identifier for the research session
        """
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Initialize state
        state = GuestResearchState(
            host_podcast=host_podcast or "",
            host_podcast_description="",
            rss_feed=rss_feed or "",
            search_guest_name=search_guest_name or "",
            episode_title=episode_title,
            episodes=[],
            episode_description="",
            guest_name=search_guest_name or "",
            guest_company="",
            transcript="",
            is_linkedin_url=bool(linkedin_url),
            is_twitter_url=bool(twitter_url),
            linkedin_url=linkedin_url or "",
            twitter_url=twitter_url or "",
            linkedIn_post="",
            linkedin_profile="",
            twitter_post="",
            guest_unique_element="",
            guest_reason="",
            report="",
            document_url="",
            request_id=request_id,
            introduction="",
            summary="",
            question="",
            appearance="",
            created_at=datetime.datetime.now().isoformat(),
            direct_social=direct_social
        )
        
        # Store the state
        self.active_sessions[session_id] = state
        self.logger.info(f"Created session {session_id} for episode '{episode_title}'")
        
        # Clean up old sessions (keep only the most recent 10)
        self._cleanup_old_sessions()
        
        return session_id
    
    def _cleanup_old_sessions(self, max_sessions=10):
        """
        Remove old sessions to prevent memory leaks
        
        Args:
            max_sessions: Maximum number of sessions to keep
        """
        if len(self.active_sessions) <= max_sessions:
            return
            
        # Sort sessions by creation time
        sessions_by_time = sorted(
            self.active_sessions.items(),
            key=lambda x: x[1].get('created_at', ''),
            reverse=True
        )
        
        # Keep only the most recent sessions
        to_keep = sessions_by_time[:max_sessions]
        keep_ids = [session_id for session_id, _ in to_keep]
        
        # Remove old sessions
        for session_id in list(self.active_sessions.keys()):
            if session_id not in keep_ids:
                del self.active_sessions[session_id]
                self.logger.info(f"Cleaned up old session {session_id}")
    
    def get_state(self, session_id: str) -> Optional[GuestResearchState]:
        """
        Get the current state for a session
        
        Args:
            session_id: The session identifier
            
        Returns:
            The current research state or None if not found
        """
        return self.active_sessions.get(session_id)
    
    def update_state(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update the state for a session
        
        Args:
            session_id: The session identifier
            updates: Dictionary of state updates
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.active_sessions:
            self.logger.error(f"Session {session_id} not found")
            return False
        
        # Update the state
        for key, value in updates.items():
            if key in self.active_sessions[session_id]:
                self.active_sessions[session_id][key] = value
        
        return True
    
    def format_results(self, session_id: str) -> Dict[str, Any]:
        """
        Format the research results for a session
        
        Args:
            session_id: The session identifier
            
        Returns:
            Formatted research results
        """
        state = self.get_state(session_id)
        if not state:
            return {"error": f"Session {session_id} not found"}
        
        return {
            "guest": state["guest_name"],
            "linkedin": state["linkedin_url"],
            "document_url": state["document_url"],
            "introduction": state.get("introduction", ""),
            "summary": state.get("summary", ""),
            "question": state.get("question", ""),
            "appearance": state.get("appearance", ""),
            "state": state
        }
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a research session
        
        Args:
            session_id: The session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Deleted research session {session_id}")
            return True
        
        self.logger.warning(f"Attempted to delete non-existent session {session_id}")
        return False
        
# Singleton instance
state_manager = ResearchStateManager() 