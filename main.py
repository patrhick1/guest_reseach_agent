"""
Podcast Guest Research Library

This module provides core functionality for podcast guest research:
1. Retrieving episode information from podcast platforms
2. Extracting guest details from transcripts/descriptions
3. Finding social media profiles and historical posts
4. Generating research reports
5. Saving results to Google Drive

Usage:
    from main import research_podcast_guest
    results = await research_podcast_guest(state)
"""

# ---------------------------
# Imports & Configurations
# ---------------------------
import os
import re
import logging
import asyncio
import uuid
import aiohttp
import tiktoken
import datetime
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict
import json

# Import shared state manager
try:
    from research_state import state_manager, GuestResearchState
    USE_STATE_MANAGER = True
    logging.info("Using shared ResearchStateManager for state management")
except ImportError:
    USE_STATE_MANAGER = False
    logging.warning("Shared ResearchStateManager not available")
    # Define GuestResearchState locally only if import fails
    class GuestResearchState(TypedDict):
        """State container for guest research workflow"""
        host_podcast: str
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

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain/LLM Components
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Service Integrations
from google_service import GoogleDocsService
from apify_client import ApifyClient
from rss_scraper import fetch_rss_feed
from transcriber import transcribe_endpoint

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Data Models
# ---------------------------

class SocialMediaProfile(BaseModel):
    """Represents social media profiles of a podcast guest"""
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    twitter: Optional[str] = Field(None, description="X/Twitter profile URL")
    name: str = Field(..., description="Full name of the person")

class GuestIdentification(BaseModel):
    """Identified guest information from episode analysis"""
    guest: str = Field(description="Full name of identified guest")
    guest_unique_element: str = Field(description="Unique identifier keyword for online searches")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    twitter: Optional[str] = Field(None, description="X/Twitter profile URL")

# ---------------------------
# Service Clients
# ---------------------------

class APIClients:
    """Container for external API clients"""
    
    def __init__(self):
        self.docs_service = GoogleDocsService()
        
        # Use the same LLM setup as app.py for consistency
        self.chat_llm = ChatOpenAI(
            model_name="o3-mini",
            temperature=None,
            openai_api_key=os.getenv("OPENAI_KEY"),
            reasoning_effort="medium"
        )
        
        # Optional Gemini model if needed
        if os.getenv("GEMINI_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                google_api_key=os.getenv("GEMINI_API_KEY"), 
                temperature=0
            )
            
        self.apify_client = ApifyClient(os.getenv("APIFY_CLIENT_TOKEN"))
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.tavily_search = TavilySearchAPIWrapper()

# ---------------------------
# Helper Functions
# ---------------------------

async def validate_social_url(pattern: re.Pattern, url: str) -> bool:
    """Validate social media URLs using regex patterns"""
    return bool(re.match(pattern, url))

# Define a custom exception for API errors
class ResearchException(Exception):
    """Exception raised for errors in the research process."""
    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

async def handle_api_error(response, service_name: str):
    """Generic API error handler"""
    if response.status != 200:
        error_msg = f"{service_name} API request failed with status {response.status}"
        logger.error(error_msg)
        raise ResearchException(error_msg, status_code=502)

# Initialize the TavilySearchAPIWrapper at module level
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_search = TavilySearchAPIWrapper()

async def async_tavily_search(query: str, max_results: int = 5):
    """Async wrapper for Tavily search"""
    try:
        return await asyncio.to_thread(
            tavily_search.results,
            query,
            max_results=max_results,
            search_depth="advanced"
        )
    except Exception as e:
        logger.error(f"Tavily search failed: {str(e)}")
        # Return empty results on error
        return []

# Transcription function using our local implementation
async def transcribe_from_url(audio_url: str, episode_name: str = "") -> str:
    """Transcribe audio from URL using our local implementation
    
    Args:
        audio_url: URL of the podcast audio file
        episode_name: Optional name of the episode for better identification
        
    Returns:
        str: The transcript text
    """
    # Call the transcribe_endpoint function directly
    try:
        logger.info(f"Transcribing audio from URL: {audio_url}")
        logger.info(f"Episode name: {episode_name}")
        
        result = await transcribe_endpoint(
            audio_url=audio_url,
            episode_name=episode_name,
            speakers=None  # Explicitly pass None for speakers
        )
        
        # Check for errors in the result
        if "error" in result:
            error_msg = f"Transcription error: {result['error']}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info("Transcription completed successfully")
        return result.get("transcript", "")
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise

async def download_transcript_from_drive(transcript_url: str) -> str:
    """Download transcript from Google Drive"""
    pattern = r"/d/([a-zA-Z0-9_-]+)"
    match = re.search(pattern, transcript_url)
    if not match:
        raise ValueError("Invalid Google Drive URL provided.")
    
    document_id = match.group(1)
    google_service = GoogleDocsService()
    
    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(None, google_service.get_document_content, document_id)
    return content

# ---------------------------
# Core Functionality
# ---------------------------

class PodcastResearchEngine:
    """Main engine handling guest research workflow"""
    
    def __init__(self):
        self.clients = APIClients()
        self.linkedin_pattern = re.compile(
            r'^https?://(www\.)?linkedin\.com/in/[a-z0-9-]+/?(\?.*)?$', re.IGNORECASE
        )
        self.twitter_pattern = re.compile(
            r'^https?://(www\.)?(twitter\.com|x\.com)/[a-z0-9_]+/?(\?.*)?$', re.IGNORECASE
        )

    async def get_host_podcast_info(self, state: GuestResearchState) -> GuestResearchState:
        """Retrieve host podcast information including description.
        
        First tries Podscan, then falls back to ListenNotes if not found.
        This method handles cases where host_podcast might not be provided.
        """
        host_podcast = state.get("host_podcast")
        
        # If no host podcast name is provided, return state unchanged
        if not host_podcast or host_podcast.strip() == "":
            logger.info("No host podcast name provided, skipping podcast info retrieval")
            state["host_podcast_description"] = ""
            return state
            
        logger.info(f"Retrieving podcast information for: {host_podcast}")
        
        # First try Podscan API
        try:
            podscan_info = await self._get_podcast_from_podscan(host_podcast)
            if podscan_info and podscan_info.get("podcasts") and len(podscan_info["podcasts"]) > 0:
                podcast = podscan_info["podcasts"][0]
                state["host_podcast_description"] = podcast.get("podcast_description", "")
                logger.info(f"Found podcast info on Podscan: {host_podcast}")
                return state
        except Exception as e:
            logger.warning(f"Failed to get podcast info from Podscan: {str(e)}")
        
        # Fall back to ListenNotes API
        try:
            logger.info(f"Falling back to ListenNotes for podcast: {host_podcast}")
            listennotes_info = await self._get_podcast_from_listennotes(host_podcast)
            
            if listennotes_info and listennotes_info.get("results") and len(listennotes_info["results"]) > 0:
                podcast = listennotes_info["results"][0]
                state["host_podcast_description"] = podcast.get("description_original", "")
                logger.info(f"Found podcast info on ListenNotes: {host_podcast}")
                return state
            else:
                logger.warning(f"No podcast info found for: {host_podcast}")
                state["host_podcast_description"] = ""
        except Exception as e:
            logger.warning(f"Failed to get podcast info from ListenNotes: {str(e)}")
            state["host_podcast_description"] = ""
            
        return state
    
    async def _get_podcast_from_podscan(self, podcast_name: str) -> Dict:
        """Retrieve podcast information from Podscan API"""
        async with aiohttp.ClientSession() as session:
            params = {"query": f'"{podcast_name}"', "per_page": 1}
            headers = {"Authorization": f"Bearer {os.getenv('PODSCANAPI')}"}
            
            async with session.get(
                "https://podscan.fm/api/v1/podcasts/search",
                headers=headers,
                params=params
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Podscan API returned status code {resp.status}")
                    return {}
                return await resp.json()
    
    async def _get_podcast_from_listennotes(self, podcast_name: str) -> Dict:
        """Retrieve podcast information from ListenNotes API"""
        if not os.getenv("LISTENNOTES_API_KEY"):
            logger.warning("ListenNotes API key not configured")
            return {}
            
        async with aiohttp.ClientSession() as session:
            params = {
                "q": f'"{podcast_name}"',
                "type": "podcast",
                "only_in": "title,description"
            }
            headers = {"X-ListenAPI-Key": os.getenv("LISTENNOTES_API_KEY")}
            
            async with session.get(
                "https://listen-api.listennotes.com/api/v2/search",
                headers=headers,
                params=params
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"ListenNotes API returned status code {resp.status}")
                    return {}
                return await resp.json()

    async def get_episode_from_rss(self, state: GuestResearchState) -> GuestResearchState:
        """Extract episode from RSS feed"""
        rss_feed = state.get("rss_feed")
        search_guest_name = state.get("search_guest_name")
        episode_title = state.get("episode_title")

        if not rss_feed:
            raise ValueError("No RSS feed provided in state.")

        podcast_title, episodes = fetch_rss_feed(rss_feed)
        episode = next((ep for ep in episodes if search_guest_name in ep["title"] or search_guest_name in ep["description"]), None)

        if not episode:
            raise ValueError(f"No episode found with title: {episode_title}")

        episode_url = episode.get("enclosure_url")
        episode_transcript = await transcribe_from_url(episode_url, episode_title)

        state["transcript"] = episode_transcript
        state["episode_description"] = episode.get("description", "")
        return state

    async def get_episode_from_podscan(self, state: GuestResearchState) -> GuestResearchState:
        """Retrieve episode metadata from Podscan API"""
        try:
            results = await self._aget_episode(state["episode_title"])
            for episode in results.get("episodes", []):
                if state["episode_title"].lower() in episode.get("episode_title", "").lower():
                    state["episode_description"] = episode.get("episode_description", "")
                    state["transcript"] = episode.get("episode_transcript", "")
                    break
            return state
        except Exception as e:
            logger.error(f"Episode data fetch failed: {str(e)}")
            raise ResearchException(f"Episode data retrieval failed: {str(e)}", status_code=500)

    async def _aget_episode(self, episode_title: str) -> Dict:
        """Async wrapper for Podscan API call"""
        async with aiohttp.ClientSession() as session:
            params = {"query": f'"{episode_title}"', "per_page": 1}
            headers = {"Authorization": f"Bearer {os.getenv('PODSCANAPI')}"}
            
            async with session.get(
                "https://podscan.fm/api/v1/episodes/search",
                headers=headers,
                params=params
            ) as resp:
                await handle_api_error(resp, "Podscan")
                return await resp.json()

    async def extract_guest_info(self, state: GuestResearchState) -> GuestResearchState:
        """Extract guest information from transcript/description"""
        try:
            # Concurrent analysis of both description and transcript
            description_info = await self._analyze_description(state["episode_description"])
            transcript_info = await self._analyze_transcript(state["transcript"])
            
            # Merge results with transcript priority
            merged_info = self._merge_guest_info(description_info, transcript_info)
            
            # Update state with merged information
            state.update({
                "guest_name": merged_info["guest"],
                "guest_unique_element": merged_info["unique_element"],
                "guest_reason": transcript_info["reason"],
                "guest_company": transcript_info["business"],
                "linkedin_url": merged_info["linkedin"],
                "twitter_url": merged_info["twitter"]
            })
            return state
        except Exception as e:
            logger.error(f"Guest info extraction failed: {str(e)}")
            raise ResearchException("Guest identification failed", status_code=400)

    async def _analyze_description(self, description: str) -> Dict:
        """Analyze episode description for guest info"""
        if not description:
            return {}
            
        prompt = SystemMessage(content= f"""
        Given a description of a podcast episode, identify who the guest of the Podcast is, if you can't confidently Identify who the guest is 
        return None
        Also in the guest unique element, A keyword that can be used to identify the user online, like their niche
        And the guest social media handles i.e Prioritize LinkedIn and Twitter Handles
        <expected output example>
        {{
        "guest": "Alice Johnson",
        "guest_unique_element": "data_science_expert",
        "linkedin": "https://www.linkedin.com/in/alicejohnson",
        "twitter": "https://x.com/alicejohnson"
        }}
        </expected output example>
        """)
        return await self.clients.llm.with_structured_output(GuestIdentification).ainvoke(
            [prompt, HumanMessage(content=description)]
        )

    async def _analyze_transcript(self, transcript: str) -> Dict:
        """Full transcript analysis with multiple aspects"""
        if not transcript:
            return {"guest": "", "unique_element": "", "reason": "", "business": "", "social": {"linkedin": "", "twitter": ""}}

        try:
            # Run analysis tasks in parallel
            analysis_tasks = [
                self._extract_guest_identification(transcript),
                self._analyze_guest_motivation(transcript),
                self._identify_guest_business(transcript)
            ]
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results safely
            guest_info = results[0] if not isinstance(results[0], Exception) else GuestIdentification(
                guest="Unknown Guest", 
                guest_unique_element="podcast_guest",
                linkedin=None, 
                twitter=None
            )
            
            guest_reason = results[1].content if not isinstance(results[1], Exception) and hasattr(results[1], 'content') else "Unknown motivation"
            guest_business = results[2].content if not isinstance(results[2], Exception) and hasattr(results[2], 'content') else "Unknown business"
            
            return {
                "guest": getattr(guest_info, "guest", "Unknown Guest"),
                "unique_element": getattr(guest_info, "guest_unique_element", "podcast_guest"),
                "reason": guest_reason,
                "business": guest_business,
                "social": {
                    "linkedin": getattr(guest_info, "linkedin", None),
                    "twitter": getattr(guest_info, "twitter", None)
                }
            }
        except Exception as e:
            logger.error(f"Error in transcript analysis: {str(e)}")
            # Return safe default values
            return {
                "guest": "Unknown Guest",
                "unique_element": "podcast_guest",
                "reason": "Unknown motivation",
                "business": "Unknown business",
                "social": {"linkedin": None, "twitter": None}
            }

    async def _extract_guest_identification(self, transcript: str) -> GuestIdentification:
        """Extract guest identification from podcast transcript"""
        try:
            prompt = SystemMessage(content=f"""
        Given a transcript of a podcast episode, identify who the guest is. 
        If you can't confidently identify the guest, return None.
        Also extract a unique element: a keyword that can be used to identify the guest online.
        """)
            return await self.clients.llm.with_structured_output(GuestIdentification).ainvoke(
                [prompt, HumanMessage(content=transcript)]
            )
        except Exception as e:
            logger.error(f"Error extracting guest identification: {str(e)}")
            # Return default GuestIdentification if there's an error
            return GuestIdentification(
                guest="Unknown Guest",
                guest_unique_element="podcast_guest",
                linkedin=None,
                twitter=None
            )
        
    async def _analyze_guest_motivation(self, transcript: str) -> HumanMessage:
        """Analyze guest's motivation for appearing on the podcast"""
        try:
            prompt = SystemMessage(content="""
            Given the podcast transcript, identify what the guest is trying to sell or market.
            I want to know the underlying reason for why the guest is on the show, what offer do they have.
            The idea is to find something the guest is passionate about that will allow them to come talk about it.
            """)
            return await self.clients.llm.ainvoke([prompt, HumanMessage(content=transcript)])
        except Exception as e:
            logger.error(f"Error analyzing guest motivation: {str(e)}")
            # Return a default message
            class MockMessage:
                def __init__(self):
                    self.content = "Guest appears to be promoting their expertise."
            return MockMessage()
        
    async def _identify_guest_business(self, transcript: str) -> HumanMessage:
        """Identify the guest's business or service from the transcript"""
        try:
            prompt = SystemMessage(content="""
            Given the podcast transcript, identify what business, product or service the guest owns.
            """)
            return await self.clients.llm.ainvoke([prompt, HumanMessage(content=transcript)])
        except Exception as e:
            logger.error(f"Error identifying guest business: {str(e)}")
            # Return a default message
            class MockMessage:
                def __init__(self):
                    self.content = "Unknown business or service."
            return MockMessage()

    def _merge_guest_info(self, desc_info: Dict, trans_info: Dict) -> Dict:
        """Merge information from different sources with improved social URL handling"""
        # Check if desc_info is a Pydantic model and convert to dict if needed
        if hasattr(desc_info, "__dict__") and not isinstance(desc_info, dict):
            # For Pydantic models
            if hasattr(desc_info, "model_dump"):
                desc_info_dict = desc_info.model_dump()
            # Fallback for older Pydantic versions
            elif hasattr(desc_info, "dict"):
                desc_info_dict = desc_info.dict()
            else:
                # Last resort - get all attributes that don't start with _
                desc_info_dict = {k: getattr(desc_info, k) for k in dir(desc_info) 
                               if not k.startswith('_') and not callable(getattr(desc_info, k))}
        else:
            desc_info_dict = desc_info or {}
        
        # Extract social info from transcript
        transcript_linkedin = trans_info.get("social", {}).get("linkedin", "")
        transcript_twitter = trans_info.get("social", {}).get("twitter", "")
        
        # Extract social info from description
        description_linkedin = desc_info_dict.get("linkedin", "")
        description_twitter = desc_info_dict.get("twitter", "")
        
        # Prioritize transcript info but try both sources
        linkedin_url = transcript_linkedin or description_linkedin
        twitter_url = transcript_twitter or description_twitter
        
        # Fix any URL issues
        fixed_linkedin = self._fix_social_url(linkedin_url, "linkedin")
        fixed_twitter = self._fix_social_url(twitter_url, "twitter")
        
        # Now safely merge the information
        return {
            "guest": trans_info.get("guest") or desc_info_dict.get("guest", ""),
            "unique_element": trans_info.get("unique_element") or desc_info_dict.get("guest_unique_element", ""),
            "linkedin": fixed_linkedin,
            "twitter": fixed_twitter
        }

    async def find_social_profiles(self, state: GuestResearchState) -> GuestResearchState:
        """Find social media profiles using search APIs"""
        try:
            search_query = f"{state['guest_name']} {state['guest_unique_element']}"
            
            # Parallel social media searches
            linkedin_task = self._search_linkedin(search_query)
            twitter_task = self._search_twitter(search_query)
            
            state["linkedin_url"], state["twitter_url"] = await asyncio.gather(
                linkedin_task, twitter_task
            )
            
            # Update validation flags
            state["is_linkedin_url"] = bool(state["linkedin_url"])
            state["is_twitter_url"] = bool(state["twitter_url"])
            
            return state
        except Exception as e:
            logger.error(f"Social media search failed: {str(e)}")
            raise ResearchException("Social media search failed", status_code=500)

    async def _search_linkedin(self, query: str) -> str:
        """Search for LinkedIn profile"""
        results = await async_tavily_search(f"{query} LinkedIn")
        for result in results:
            if await validate_social_url(self.linkedin_pattern, result["url"]):
                return result["url"]
        return ""

    async def _search_twitter(self, query: str) -> str:
        """Search for Twitter/X profile"""
        results = await async_tavily_search(f"{query} Twitter")
        for result in results:
            if await validate_social_url(self.twitter_pattern, result["url"]):
                return result["url"]
        return ""

    def _fix_social_url(self, url: str, platform: str) -> str:
        """
        Fix social media URL or username to ensure it's in the proper URL format.
        
        Args:
            url: The URL or username to fix
            platform: Either "linkedin" or "twitter"
            
        Returns:
            A properly formatted URL or empty string if invalid
        """
        if not url:
            return ""
        
        if platform == "twitter":
            # Define patterns
            full_url_pattern = self.twitter_pattern
            username_pattern = re.compile(r'^@?([a-z0-9_]+)$', re.IGNORECASE)
            partial_url_pattern = re.compile(
                r'^(twitter\.com|x\.com)/([a-z0-9_]+)/?(\?.*)?$', 
                re.IGNORECASE
            )
            
            # Case 1: Already a full URL
            if full_url_pattern.match(url):
                return url
            
            # Case 2: Just a username
            username_match = username_pattern.match(url)
            if username_match:
                username = username_match.group(1)  # Extract the username without @
                return f"https://twitter.com/{username}"
            
            # Case 3: Partial URL
            partial_match = partial_url_pattern.match(url)
            if partial_match:
                domain = partial_match.group(1)
                username = partial_match.group(2)
                return f"https://{domain}/{username}"
        
        elif platform == "linkedin":
            # Define patterns
            full_url_pattern = self.linkedin_pattern
            username_pattern = re.compile(r'^@?([a-z0-9-]+)$', re.IGNORECASE)
            partial_url_pattern = re.compile(
                r'^linkedin\.com/in/([a-z0-9-]+)/?(\?.*)?$', 
                re.IGNORECASE
            )
            
            # Case 1: Already a full URL
            if full_url_pattern.match(url):
                return url
            
            # Case 2: Just a username
            username_match = username_pattern.match(url)
            if username_match:
                username = username_match.group(1)  # Extract the username without @
                return f"https://linkedin.com/in/{username}"
            
            # Case 3: Partial URL
            partial_match = partial_url_pattern.match(url)
            if partial_match:
                username = partial_match.group(1)
                return f"https://linkedin.com/in/{username}"
        
        return ""  # Invalid format

    async def retrieve_social_content(self, state: GuestResearchState) -> GuestResearchState:
        """Retrieve historical social media posts"""
        try:
            logger.info(f"Starting social content retrieval for guest: {state['guest_name']}")
            logger.info(f"LinkedIn URL: {state['linkedin_url']}, Twitter URL: {state['twitter_url']}")
            
            tasks = []
            if state["linkedin_url"]:
                logger.info(f"Retrieving LinkedIn posts from {state['linkedin_url']}")
                tasks.append(self._get_linkedin_posts(state["linkedin_url"]))
                logger.info(f"Retrieving LinkedIn profile from {state['linkedin_url']}")
                tasks.append(self._get_linkedin_profile(state["linkedin_url"]))
            else:
                logger.info("No LinkedIn URL provided, skipping LinkedIn retrieval")
                tasks.append(asyncio.sleep(0, result=""))
                tasks.append(asyncio.sleep(0, result=""))
                
            if state["twitter_url"]:
                logger.info(f"Retrieving Twitter posts from {state['twitter_url']}")
                tasks.append(self._get_twitter_posts(state["twitter_url"]))
            else:
                logger.info("No Twitter URL provided, skipping Twitter retrieval")
                tasks.append(asyncio.sleep(0, result=""))
            
            logger.info("Waiting for all social media tasks to complete")
            results = await asyncio.gather(*tasks)
            logger.info("All social media tasks completed")
            
            # Map results to state
            if state["linkedin_url"]:
                logger.info("Storing LinkedIn posts result")
                state["linkedIn_post"] = results[0]
                logger.info(f"LinkedIn posts result length: {len(results[0]) if results[0] else 0}")
                
                logger.info("Storing LinkedIn profile result")
                state["linkedin_profile"] = results[1]
                logger.info(f"LinkedIn profile result length: {len(results[1]) if results[1] else 0}")
            
            if state["twitter_url"]:
                logger.info("Storing Twitter posts result")
                state["twitter_post"] = results[-1]
                logger.info(f"Twitter posts result length: {len(results[-1]) if results[-1] else 0}")
            
            logger.info("Social content retrieval completed successfully")
            return state
        except Exception as e:
            logger.error(f"Social content retrieval failed: {str(e)}")
            raise ResearchException("Social content retrieval failed", status_code=500)

    async def _get_linkedin_posts(self, profile_url: str) -> str:
        """Retrieve LinkedIn posts using Apify"""
        # Handle empty or None profile URL
        if not profile_url:
            logger.info("LinkedIn posts URL not provided")
            return "No LinkedIn profile URL provided."
            
        try:
            logger.info(f"Starting LinkedIn posts retrieval for URL: {profile_url}")
            run_input = {
                "username": profile_url,
                "maxPosts": 20,
                "timeout": 60
            }
            
            # Call the Apify actor with proper error handling
            try:
                logger.info("Calling LinkedIn posts scraper actor")
                run = await asyncio.to_thread(
                    self.clients.apify_client.actor('apimaestro/linkedin-profile-posts').call,
                    run_input=run_input
                )
                logger.info("LinkedIn posts scraper actor call completed")
            except Exception as actor_error:
                logger.warning(f"Error calling LinkedIn scraper actor: {str(actor_error)}")
                return f"Error accessing LinkedIn data: {str(actor_error)}"
                
            # Verify run contains the expected data
            if not run or not isinstance(run, dict) or "defaultDatasetId" not in run:
                logger.warning(f"Invalid response from LinkedIn scraper for {profile_url}")
                return "Unable to retrieve LinkedIn posts: Invalid API response"
            
            # Get the dataset with proper error handling
            try:
                logger.info(f"Retrieving LinkedIn posts dataset with ID: {run['defaultDatasetId']}")
                dataset = await asyncio.to_thread(
                    self.clients.apify_client.dataset(run["defaultDatasetId"]).list_items
                )
                logger.info("LinkedIn posts dataset retrieved successfully")
            except Exception as dataset_error:
                logger.warning(f"Error retrieving LinkedIn dataset: {str(dataset_error)}")
                return f"Error retrieving LinkedIn posts: {str(dataset_error)}"
            
            # Check if dataset and items exist
            if not dataset or not hasattr(dataset, 'items'):
                logger.warning(f"No LinkedIn posts data structure for {profile_url}")
                return "No LinkedIn posts found (invalid data structure)."
                
            if not dataset.items:
                logger.info(f"No LinkedIn posts found for {profile_url}")
                return "No LinkedIn posts found for this profile."
            
            logger.info(f"Found {len(dataset.items)} LinkedIn posts")
            formatted_posts = self._format_social_posts(dataset.items, "linkedin")
            logger.info(f"Formatted {len(dataset.items)} LinkedIn posts")
            return formatted_posts
        except Exception as e:
            logger.warning(f"LinkedIn posts retrieval failed: {str(e)}")
            return f"Error retrieving LinkedIn posts: {str(e)}"

    async def _get_twitter_posts(self, profile_url: str) -> str:
        """Retrieve Twitter posts using Apify"""
        # Handle empty or None profile URL
        if not profile_url:
            logger.info("Twitter profile URL not provided")
            return "No Twitter profile URL provided."
        
        try:
            # Extract username from URL (x.com or twitter.com)
            username = profile_url.rstrip('/').split('/')[-1].split('?')[0]
            
            run_input = {
                "max_posts": 50,
                "username": username
            }
            
            logger.info(f"Getting Twitter posts for username: {username}")
            
            # Call the Apify actor with proper error handling
            try:
                logger.info(f"Calling Twitter scraper actor for @{username}")
                run = await asyncio.to_thread(
                    self.clients.apify_client.actor("danek/twitter-scraper-ppr").call,
                    run_input=run_input
                )
                logger.info("Twitter scraper actor call completed")
            except Exception as actor_error:
                logger.warning(f"Error calling Twitter scraper actor: {str(actor_error)}")
                return f"Error accessing Twitter data for @{username}: {str(actor_error)}"
            
            # Verify run contains the expected data
            if not run or not isinstance(run, dict) or "defaultDatasetId" not in run:
                logger.warning(f"Invalid response from Twitter scraper for {username}")
                return f"Unable to retrieve tweets for @{username}: Invalid API response"
            
            # Get the dataset with proper error handling
            try:
                logger.info(f"Retrieving Twitter dataset with ID: {run['defaultDatasetId']}")
                dataset = await asyncio.to_thread(
                    self.clients.apify_client.dataset(run["defaultDatasetId"]).list_items
                )
                logger.info(f"Twitter dataset retrieved successfully for @{username}")
            except Exception as dataset_error:
                logger.warning(f"Error retrieving dataset for {username}: {str(dataset_error)}")
                return f"Error retrieving tweets for @{username}: {str(dataset_error)}"
            
            # Check if dataset exists
            if not dataset:
                logger.warning(f"No dataset object returned for {username}")
                return f"No tweets could be retrieved for @{username}: Empty dataset"
            
            # Check if dataset.items attribute exists
            if not hasattr(dataset, 'items'):
                logger.warning(f"Dataset has no 'items' attribute for {username}")
                return f"No tweets could be retrieved for @{username}: Invalid dataset structure"
            
            # Check if dataset.items is None
            if dataset.items is None:
                logger.warning(f"Dataset items is None for {username}")
                return f"No tweets could be retrieved for @{username}: Dataset items is None"
            
            # Check if dataset.items is empty
            if len(dataset.items) == 0:
                logger.warning(f"No tweets found in dataset for {username}")
                return f"No tweets found for this profile (@{username}). The account may be protected, inactive, or not exist."
            
            # Log the first tweet structure for debugging
            if len(dataset.items) > 0:
                first_tweet = dataset.items[0]
                tweet_keys = list(first_tweet.keys()) if isinstance(first_tweet, dict) else "Not a dictionary"
                logger.info(f"First tweet structure for @{username} has keys: {tweet_keys}")
            
            # Format the tweets
            logger.info(f"Found {len(dataset.items)} tweets for @{username}")
            formatted_posts = self._format_social_posts(dataset.items, "twitter")
            logger.info(f"Formatted {len(dataset.items)} tweets for @{username}")
            return formatted_posts
        except Exception as e:
            logger.warning(f"Twitter posts retrieval failed: {str(e)}")
            # Return a formatted error message
            return f"Error retrieving tweets for @{username}: {str(e)}"

    def _format_social_posts(self, posts: List[Dict], platform: str) -> str:
        """Format social posts for analysis"""
        if not posts:
            return f"No {platform} posts found."
        
        logger.info(f"Formatting {len(posts)} {platform} posts")
        
        formatted = []
        for idx, post in enumerate(posts[:10]):  # Limit to 10 posts
            try:
                if platform == "linkedin":
                    # Handle LinkedIn posts
                    text = post.get("text", "")
                    if text and isinstance(text, str):
                        text = text[:500]  # Truncate long posts
                    else:
                        text = "No content"
                    
                    # Safely get date with fallbacks
                    posted_at = post.get("posted_at")
                    if posted_at and isinstance(posted_at, dict):
                        date = posted_at.get("date", "Unknown date")
                    else:
                        date = "Unknown date"
                    
                elif platform == "twitter":
                    # Handle Twitter posts
                    # Safely get text
                    text = post.get("text", "")
                    if text and isinstance(text, str):
                        text = text[:500]  # Truncate long posts
                    else:
                        text = "No content"
                    
                    # Safely get date
                    date = post.get("created_at", "Unknown date")
                    
                    # Safely get engagement metrics with default values
                    favorites = post.get("favorites", "")
                    retweets = post.get("retweets", "")
                    
                    # Add engagement metrics if they exist
                    engagement = ""
                    if favorites:
                        engagement += f" | ❤️ {favorites}"
                    if retweets:
                        engagement += f" | 🔄 {retweets}"
                    
                    if engagement:
                        text = f"{text}\n{engagement}"
                    
                    # Handle quoted tweets if present
                    quoted = post.get("quoted")
                    if quoted and isinstance(quoted, dict):
                        quoted_text = quoted.get("text", "")
                        if quoted_text and isinstance(quoted_text, str):
                            quoted_text = quoted_text[:200]  # Truncate quoted tweets
                            quoted_author = "Unknown"
                            
                            # Safely get author info
                            quoted_author_info = quoted.get("author")
                            if quoted_author_info and isinstance(quoted_author_info, dict):
                                quoted_author = quoted_author_info.get("screen_name", "Unknown")
                            
                            text += f"\n\n↪️ Quoting @{quoted_author}: {quoted_text}"
                
                # Add formatted post to results
                formatted.append(f"Post {idx+1} ({date}):\n{text}\n")
                
            except Exception as e:
                # Log any errors during formatting but continue with other posts
                logger.warning(f"Error formatting {platform} post {idx}: {str(e)}")
                formatted.append(f"Post {idx+1}: [Error formatting post]\n")
        
        if not formatted:
            return f"No {platform} posts could be extracted."
        
        logger.info(f"Successfully formatted {len(formatted)} {platform} posts")    
        return "\n".join(formatted)

    async def get_other_appearances(self, state: GuestResearchState) -> GuestResearchState:
        """Search for other podcast appearances"""
        if not os.getenv("LISTENNOTES_API_KEY"):
            raise ResearchException("ListenNotes API key not configured.", status_code=500)

        guest_name = state["guest_name"]
        url = "https://listen-api.listennotes.com/api/v2/search"

        async with aiohttp.ClientSession() as session:
            params = {
                "q": f"\"{guest_name}\" with guest",
                "type": "episode",
                "only_in": "title,description",
                "interviews_only": 1 
            }
            headers = {"X-ListenAPI-Key": os.getenv("LISTENNOTES_API_KEY")}

            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    raise ResearchException("ListenNotes search failed", status_code=502)

                results = await resp.json()
                state["episodes"] = [
                    {
                        "title": ep.get("title_original", ""),
                        "link": ep.get("link", ""),
                        "audio": ep.get("audio", ""),
                        "image": ep.get("image", "")
                    } 
                    for ep in results.get("results", [])
                ]
                return state

    async def generate_research_report(self, state: GuestResearchState) -> GuestResearchState:
        """Generate comprehensive research report"""
        try:
            state["report"] = await self._generate_detailed_report(state)
            return state
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise ResearchException("Report generation failed", status_code=500)
            
    async def _generate_detailed_report(self, state: GuestResearchState) -> str:
        """
        Generate a detailed profile report by dividing the work among multiple LLM calls:
        - An introduction with social media handles, company, and offers.
        - A summary of topics and themes from the posts.
        - A set of engaging podcast questions for the guest.
        - A structured summary of the guest's previous podcast appearances.
        """
        # Extracting required state data
        host_podcast_name = state.get("host_podcast", "")
        host_podcast_description = state.get("host_podcast_description", "")
        linkedin_profile = state.get("linkedin_profile", "")
        linkedin_post = state.get("linkedIn_post", "")
        twitter_post = state.get("twitter_post", "")
        guest_name = state.get("guest_name", "")
        twitter_url = state.get("twitter_url", "")
        linkedin_url = state.get("linkedin_url", "")
        other_appearances = state.get("episodes", [])
        company = state.get("guest_company", "")
        offers = state.get("guest_reason", "")

        # LLM call 1: Create Introduction
        introduction_prompt = f"""
        Write an introduction for the guest. Include their social media handles:
        - LinkedIn: {linkedin_url} and \nLinkedIn Summary: {linkedin_profile}
        - Twitter: {twitter_url}
        
        Also mention their company, {company}, and any offers if applicable: {offers}.
        Provide a warm, engaging introduction.
        """
        introduction_response = await self.clients.llm.ainvoke([
            SystemMessage(content=introduction_prompt),
            HumanMessage(content=f"Guest name: {guest_name}")
        ])
        introduction = introduction_response.content.strip()
        state["introduction"] = introduction

        # LLM call 2: Generate Summary of Topics & Themes
        summary_prompt = f"""
        Analyze the following posts and extract the main topics and recurring themes:
        
        LinkedIn Post: {linkedin_post} and \nLinkedIn Summary: {linkedin_profile}
        Twitter Post: {twitter_post}
        
        Summarize your findings in a clear, concise manner using listicles.
        """
        summary_response = await self.clients.llm.ainvoke([
            SystemMessage(content=summary_prompt),
            HumanMessage(content=f"Guest name: {guest_name}")
        ])
        summary = summary_response.content.strip()
        state["summary"] = summary

        # LLM call 3: Create Podcast Questions
        podcast_context = ""
        if host_podcast_name and host_podcast_description:
            podcast_context = f"""
            The podcast is called '{host_podcast_name}' and here's its description:
            {host_podcast_description}
            
            Make sure the questions align with the theme and style of this podcast.
            """
        
        questions_prompt = f"""
        Based on the content of the posts below, generate a list of thoughtful and engaging questions to ask the guest on a podcast:
        
        LinkedIn Post: {linkedin_post} and \nLinkedIn Summary: {linkedin_profile}
        Twitter Post: {twitter_post}
        
        Podcast Context: {podcast_context}
        
        The questions should prompt discussion and reveal interesting insights.
        """
        questions_response = await self.clients.llm.ainvoke([
            SystemMessage(content=questions_prompt),
            HumanMessage(content=f"Guest name: {guest_name}")
        ])
        questions = questions_response.content.strip()
        state["question"] = questions

        # LLM call 4: Structure Previous Podcast Appearances
        appearances_prompt = f"""
        Based on the podcast appearances data below, create a formatted list of the guest's previous appearances.

        For EACH podcast appearance:
        1. Extract the image URL (which looks like https://cdn-images-*.listennotes.com/...)
        2. Extract the podcast URL (which will be the URL to the podcast platform)
        3. Extract the podcast title
        
        Format each entry as follows (one per line):
        PODCAST_TITLE||IMAGE_URL||PODCAST_URL
        
        For example:
        The Growth Show||https://cdn-images-1.listennotes.com/podcasts/example.jpg||https://example.com/podcast
        
        Don't include any additional text, commentary, or formatting - just the formatted entries.
        
        Here's the data:
        {other_appearances}
        """
        appearances_response = await self.clients.llm.ainvoke([
            SystemMessage(content=appearances_prompt),
            HumanMessage(content=f"Guest name: {guest_name}")
        ])
        appearances = appearances_response.content.strip()
        
        # Transform the LLM response into a more display-friendly format
        try:
            # Process the response into HTML for displaying images
            formatted_appearances = []
            for line in appearances.split('\n'):
                if '||' in line:
                    parts = line.split('||')
                    if len(parts) == 3:
                        title, image_url, podcast_url = parts
                        formatted_appearances.append(f"""
                        <div class="podcast-item">
                            <a href="{podcast_url}" target="_blank">
                                <img src="{image_url}" alt="{title}" title="{title}">
                                <p>{title}</p>
                            </a>
                        </div>
                        """)
            
            if formatted_appearances:
                # Use proper HTML structure with grid container for research_detail.html compatibility
                appearances = f"""<div class="podcast-grid">{''.join(formatted_appearances)}</div>"""
                logger.info(f"Successfully formatted {len(formatted_appearances)} podcast appearances with HTML")
            else:
                # Fallback: if parsing failed, revert to the direct LLM output but add a special marker
                # to indicate it should be treated as markdown
                logger.warning("No appearances could be parsed from the response, using original format")
                appearances = f"<!-- MARKDOWN_FORMAT -->\n{appearances}"
        except Exception as e:
            logger.warning(f"Failed to format appearances: {e}")
            # Keep the original response if formatting fails, but mark it
            appearances = f"<!-- MARKDOWN_FORMAT -->\n{appearances}"
        
        state["appearance"] = appearances

        # Combine all parts into a final Markdown formatted report
        final_report = f"""
# Guest Profile Report for {guest_name}

## Introduction
{introduction}

## Summary of Topics & Themes
{summary}

## Podcast Questions
{questions}

## Previous Podcast Appearances
{appearances}
        """
        return final_report.strip()

    async def save_results(self, state: GuestResearchState) -> GuestResearchState:
        """Save results to external services"""
        try:
            # Save to Google Drive
            if state["report"]:
                state["document_url"] = await self._save_to_drive(state)

            
            return state
        except Exception as e:
            logger.error(f"Results saving failed: {str(e)}")
            raise ResearchException("Results saving failed", status_code=500)

    async def _save_to_drive(self, state: GuestResearchState) -> str:
        """Save report to Google Drive"""
        doc_title = f"{state['guest_name']} Research Report"
        return self.clients.docs_service.create_document(
            title=doc_title,
            content=state["report"],
            folder_id=os.getenv("GOOGLE_REPORT_FOLDER_ID")
        )


    async def _get_linkedin_profile(self, profile_url: str) -> str:
        """Extract LinkedIn profile information using Apify"""
        # Handle empty or None profile URL
        if not profile_url:
            logger.info("LinkedIn profile URL not provided")
            return "No LinkedIn profile URL provided."
            
        try:
            logger.info(f"Starting LinkedIn profile extraction for URL: {profile_url}")
            
            # Call the LinkedIn profile scraper actor with proper error handling
            try:
                logger.info("Calling LinkedIn profile scraper actor")
                run = await asyncio.to_thread(
                    self.clients.apify_client.actor('supreme_coder/linkedin-profile-scraper').call,
                    run_input={
                        "findContacts": False,
                        "scrapeCompany": False,
                        "urls": [
                            {
                                "url": profile_url,
                                "method": "GET"
                            }
                        ]
                    }
                )
                logger.info("LinkedIn scraper actor call completed")
            except Exception as actor_error:
                logger.warning(f"Error calling LinkedIn profile scraper: {str(actor_error)}")
                return f"Error accessing LinkedIn profile data: {str(actor_error)}"
                
            # Verify run contains the expected data
            if not run or not isinstance(run, dict) or "defaultDatasetId" not in run:
                logger.warning(f"Invalid response from LinkedIn profile scraper for {profile_url}")
                return "Unable to retrieve LinkedIn profile: Invalid API response"

            # Get the dataset with proper error handling
            try:
                logger.info(f"Retrieving dataset with ID: {run['defaultDatasetId']}")
                dataset = await asyncio.to_thread(
                    self.clients.apify_client.dataset(run["defaultDatasetId"]).list_items
                )
                logger.info("LinkedIn profile dataset retrieved successfully")
            except Exception as dataset_error:
                logger.warning(f"Error retrieving LinkedIn profile dataset: {str(dataset_error)}")
                return f"Error retrieving LinkedIn profile: {str(dataset_error)}"

            # Check if we got any data
            if not dataset or not hasattr(dataset, 'items') or not dataset.items:
                logger.warning(f"No LinkedIn profile data found for {profile_url}")
                return "No LinkedIn profile data found."

            logger.info(f"Processing LinkedIn profile data for {profile_url}")
            
            # For this example, we assume the first item in the dataset is our profile.
            profile = dataset.items[0]
            
            # Ensure profile is a dictionary
            if not isinstance(profile, dict):
                logger.warning(f"LinkedIn profile data is not in expected format for {profile_url}")
                return "LinkedIn profile data is in an unexpected format."

            # Extracting basic profile details (with safe gets)
            full_name = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
            full_name = full_name if full_name else "N/A"
            headline = profile.get("headline", "N/A")
            summary = profile.get("summary", "N/A")
            occupation = profile.get("occupation", "N/A")
            
            # Add the missing variables with safe defaults
            current_job = profile.get("jobTitle", "N/A")
            current_company = profile.get("companyName", "N/A")
            location = profile.get("geoLocationName", "N/A")
            connections = profile.get("connectionsCount", "N/A")
            followers = profile.get("followersCount", "N/A")
            
            logger.info(f"Extracted basic details for {full_name}")
            
            # Extract education details
            educations = profile.get("educations", [])
            education_details = "\n".join(
                f"• {edu.get('schoolName', '')} - {edu.get('degreeName', 'No Degree')} in {edu.get('fieldOfStudy', 'Unknown Field')}"
                for edu in educations
            ) or "N/A"
            logger.info(f"Extracted {len(educations)} education entries")

            # Extract positions (experience)
            positions = profile.get("positions", [])
            positions_details = ""
            for pos in positions:
                title = pos.get("title", "N/A")
                company = pos.get("companyName", "N/A")
                location_name = pos.get("locationName", "N/A")
                time_period = pos.get("timePeriod", {})
                start_date = time_period.get("startDate", {})
                start = f"{start_date.get('month', '')}/{start_date.get('year', '')}" if start_date else "Unknown"
                end_date = time_period.get("endDate", {})
                end = f"{end_date.get('month', '')}/{end_date.get('year', '')}" if end_date else "Present"
                positions_details += f"• {title} at {company} ({location_name}) from {start} to {end}\n"
            logger.info(f"Extracted {len(positions)} position entries")

            # Extract certifications
            certifications = profile.get("certifications", [])
            certs_details = "\n".join(
                f"• {cert.get('name', 'N/A')} by {cert.get('authority', 'Unknown Authority')}"
                for cert in certifications
            ) or "N/A"
            logger.info(f"Extracted {len(certifications)} certification entries")

            # Extract skills (list joined by commas)
            skills = profile.get("skills", [])
            skills_details = ", ".join(skills) if skills else "N/A"
            logger.info(f"Extracted {len(skills)} skills")

            # Extract volunteer experiences
            volunteer_experiences = profile.get("volunteerExperiences", [])
            volunteer_details = "\n".join(
                f"• {vol.get('role', 'N/A')} at {vol.get('companyName', 'N/A')}"
                for vol in volunteer_experiences
            ) or "N/A"
            logger.info(f"Extracted {len(volunteer_experiences)} volunteer experiences")

            # Build the final summary string
            result = (
                f"Name: {full_name}\n"
                f"Headline: {headline}\n"
                f"Summary: {summary}\n\n"
                f"Occupation: {occupation}\n"
                f"Current Position: {current_job} at {current_company}\n"
                f"Location: {location}\n"
                f"Connections: {connections}\n"
                f"Followers: {followers}\n\n"
                "Education:\n"
                f"{education_details}\n\n"
                "Positions:\n"
                f"{positions_details}\n"
                "Certifications:\n"
                f"{certs_details}\n\n"
                "Skills:\n"
                f"{skills_details}\n\n"
                "Volunteer Experiences:\n"
                f"{volunteer_details}"
            )

            logger.info(f"LinkedIn profile data formatting complete for {full_name}")
            return result
        except Exception as e:
            logger.warning(f"LinkedIn profile extraction failed: {str(e)}")
            return f"Error retrieving LinkedIn profile: {str(e)}"

# ---------------------------
# Workflow Configuration
# ---------------------------

def determine_social_media_search(state: GuestResearchState) -> str:
    """Conditional routing for social media search"""
    has_social = state["is_linkedin_url"] or state["is_twitter_url"]
    return "skip_search" if has_social else "search_social"

def create_research_workflow(entry_point: str = None) -> StateGraph:
    """Configure and return the LangGraph state machine"""
    builder = StateGraph(GuestResearchState)
    engine = PodcastResearchEngine()

    # Define node functions
    nodes = [
        ("get_episode_from_rss", engine.get_episode_from_rss),
        ("get_episode_from_podscan", engine.get_episode_from_podscan),
        ("extract_guest_info", engine.extract_guest_info),
        ("find_social_profiles", engine.find_social_profiles),
        ("get_other_appearances", engine.get_other_appearances),
        ("retrieve_social_content", engine.retrieve_social_content),
        ("get_host_podcast_info", engine.get_host_podcast_info),
        ("generate_research_report", engine.generate_research_report),
        ("save_results", engine.save_results)
    ]

    # Add nodes to workflow
    for node_name, node_func in nodes:
        builder.add_node(node_name, node_func)

    # Configure workflow edges based on entry point
    if entry_point == "get_episode_from_rss":
        builder.set_entry_point("get_episode_from_rss")
        builder.add_edge("get_episode_from_rss", "extract_guest_info")
    elif entry_point == "retrieve_social_content":
        # For direct social media profiles, we start from retrieving content
        # Then check other appearances before generating the report
        builder.set_entry_point("retrieve_social_content")
        builder.add_edge("retrieve_social_content", "get_other_appearances")
        builder.add_edge("get_other_appearances", "get_host_podcast_info")
        builder.add_edge("get_host_podcast_info", "generate_research_report")
    elif entry_point == "get_other_appearances":
        # For cases where we already have social media handles but want to find appearances
        builder.set_entry_point("get_other_appearances")
        builder.add_edge("get_other_appearances", "retrieve_social_content")
    else:
        # Default flow starting with Podscan
        builder.set_entry_point("get_episode_from_podscan")
        builder.add_edge("get_episode_from_podscan", "extract_guest_info")

    # Configure conditional and standard edges if they're not already handled
    if entry_point not in ["retrieve_social_content"]:
        builder.add_conditional_edges(
            "extract_guest_info",
            determine_social_media_search,
            {
                "search_social": "find_social_profiles",
                "skip_search": "get_other_appearances"
            }
        )
        builder.add_edge("find_social_profiles", "get_other_appearances")
    
    # Add the edge from get_other_appearances to retrieve_social_content
    # only if we're not starting from retrieve_social_content
    if entry_point not in ["retrieve_social_content"]:
        builder.add_edge("get_other_appearances", "retrieve_social_content")
    
    # Add the edge from retrieve_social_content to host podcast info
    # then to generate_research_report, for all workflows except when starting from retrieve_social_content
    if entry_point not in ["retrieve_social_content"]:
        builder.add_edge("retrieve_social_content", "get_host_podcast_info")
        builder.add_edge("get_host_podcast_info", "generate_research_report")
        
    builder.add_edge("generate_research_report", "save_results")
    builder.add_edge("save_results", END)

    return builder.compile(checkpointer=MemorySaver())

# ---------------------------
# Main Execution
# ---------------------------

async def research_podcast_guest(state: GuestResearchState, session_id: str = None, custom_entry_point: str = None) -> Dict[str, Any]:
    """
    Main function to research a podcast guest
    
    Parameters:
        state (GuestResearchState): The initial state containing episode info
        session_id (str): Optional session ID for tracking (will generate if None)
        custom_entry_point (str): Optional specific entry point for the workflow
        
    Returns:
        Dict[str, Any]: Research results including guest info and document URL
    """
    try:
        logger.info("=== Starting podcast guest research ===")
        logger.info(f"Episode title: {state['episode_title']}")
        logger.info(f"Request ID: {state['request_id']}")
        logger.info(f"RSS feed: {state.get('rss_feed', 'Not provided')}")
        logger.info(f"Search guest name: {state.get('search_guest_name', 'Not provided')}")
        logger.info(f"Host podcast: {state.get('host_podcast', 'Not provided')}")
        logger.info(f"LinkedIn URL: {state.get('linkedin_url', 'Not provided')}")
        logger.info(f"Twitter URL: {state.get('twitter_url', 'Not provided')}")
        logger.info(f"Direct social: {state.get('direct_social', False)}")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        logger.info(f"Using session ID: {session_id}")
        
        # Choose entry point based on provided parameters
        entry_point = None
        if custom_entry_point:
            # If a specific entry point is provided, use it
            entry_point = custom_entry_point
            logger.info(f"Using custom workflow entry point: {entry_point}")
        elif state.get('direct_social') and (state.get('linkedin_url') or state.get('twitter_url')):
            # If direct social is enabled and we have social media URLs
            entry_point = "retrieve_social_content"
            logger.info(f"Using direct social workflow entry point: {entry_point}")
        elif state.get('rss_feed') and state.get('search_guest_name'):
            # If RSS feed and guest name are provided
            entry_point = "get_episode_from_rss"
            logger.info(f"Using RSS feed workflow entry point: {entry_point}")
        else:
            # Default to Podscan
            logger.info(f"Using default workflow entry point (Podscan)")
        
        # Initialize workflow
        logger.info("Creating research workflow")
        workflow = create_research_workflow(entry_point)
        
        # Run workflow
        logger.info("Starting workflow execution")
        results = await workflow.ainvoke(
            state,
            config=RunnableConfig(
                configurable={
                    "thread_id": session_id,
                    "checkpoint_ns": "research_session"
                }
            )
        )
        
        logger.info("Workflow execution completed")
        
        # Reinitialize the checkpointer by replacing it with a new MemorySaver instance
        workflow.checkpointer = MemorySaver()
        logger.info("Reinitialized workflow checkpointer")
        
        # Format results for return
        logger.info(f"Formatting research results for guest: {results.get('guest_name', 'Unknown')}")
        logger.info(f"LinkedIn URL: {results.get('linkedin_url', 'None')}")
        logger.info(f"Document URL: {results.get('document_url', 'None')}")
        
        formatted_results = {
            "guest": results.get("guest_name", ""),
            "linkedin": results.get("linkedin_url", ""),
            "document_url": results.get("document_url", ""),
            "introduction": results.get("introduction", ""),
            "summary": results.get("summary", ""),
            "question": results.get("question", ""),
            "appearance": results.get("appearance", "")
        }
        
        logger.info("=== Podcast guest research completed ===")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Research failed: {str(e)}")
        logger.error("=== Podcast guest research failed ===")
        raise e

if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Usage example:")
    print("    from main import research_podcast_guest")
    print("    results = await research_podcast_guest(state)")