"""
Podcast Guest Research Automation System

This FastAPI application automates the process of researching podcast guests by:
1. Retrieving episode information from podcast platforms
2. Extracting guest details from transcripts/descriptions
3. Finding social media profiles and historical posts
4. Generating research reports
5. Saving results to Google Drive and Airtable

Key Components:
- Async API integrations (Podscan, ListenNotes, Twitter, LinkedIn)
- AI-powered analysis using LangChain/LangGraph
- Google Drive integration for report storage
- Airtable integration for data management
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
import uvicorn
import tiktoken
import datetime
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from airtable_services import PodcastService
from rss_scraper import fetch_rss_feed
from apify_client import ApifyClient

# Configure environment
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------
# Logging Configuration
# ---------------------------
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
    record_id: str

# ---------------------------
# Service Clients
# ---------------------------

class APIClients:
    """Container for external API clients"""
    
    def __init__(self):
        self.podcast_service = PodcastService()
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

async def handle_api_error(response, service_name: str):
    """Generic API error handler"""
    if response.status != 200:
        error_msg = f"{service_name} API request failed with status {response.status}"
        logger.error(error_msg)
        raise HTTPException(status_code=502, detail=error_msg)

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

def transcribe_from_url(audio_url: str, payload: dict) -> str:
    """Transcribe audio from URL using Replit endpoint"""
    endpoint = "http://localhost:8000/transcribe"
    form_data = {
        "audio_url": audio_url,
        "episode_name": payload.get("episode_name", "")
    }
    if payload.get("speakers"):
        form_data["speakers"] = payload["speakers"]

    response = requests.post(endpoint, data=form_data)
    if response.status_code != 200:
        raise Exception(f"Transcription endpoint error: {response.status_code} {response.text}")
    
    result = response.json()
    return result.get("transcript", "")

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

    async def initialize_state(self, episode_title: str, record_id: str, rss_feed: str = None, search_guest_name: str = None) -> GuestResearchState:
        """Create initial research state container"""
        return GuestResearchState(
            host_podcast="",
            rss_feed=rss_feed or "",
            search_guest_name=search_guest_name or "",
            episode_title=episode_title,
            episodes=[],
            episode_description="",
            guest_name="",
            guest_company="",
            transcript="",
            is_linkedin_url=False,
            is_twitter_url=False,
            linkedin_url="",
            twitter_url="",
            linkedIn_post="",
            linkedin_profile="",
            twitter_post="",
            guest_unique_element="",
            guest_reason="",
            report="",
            document_url="",
            record_id=record_id
        )

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
        payload = {"episode_name": episode_title}
        episode_transcript = transcribe_from_url(episode_url, payload)

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
            raise HTTPException(status_code=500, detail="Episode data retrieval failed")

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
            raise HTTPException(status_code=400, detail="Guest identification failed")

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
            raise HTTPException(status_code=500, detail="Social media search failed")

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
            tasks = []
            if state["linkedin_url"]:
                tasks.append(self._get_linkedin_posts(state["linkedin_url"]))
                tasks.append(self._get_linkedin_profile(state["linkedin_url"]))
            else:
                tasks.append(asyncio.sleep(0, result=""))
                tasks.append(asyncio.sleep(0, result=""))
                
            if state["twitter_url"]:
                tasks.append(self._get_twitter_posts(state["twitter_url"]))
            else:
                tasks.append(asyncio.sleep(0, result=""))
            
            results = await asyncio.gather(*tasks)
            
            # Map results to state
            if state["linkedin_url"]:
                state["linkedIn_post"] = results[0]
                state["linkedin_profile"] = results[1]
            
            if state["twitter_url"]:
                state["twitter_post"] = results[-1]
            
            return state
        except Exception as e:
            logger.error(f"Social content retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Social content retrieval failed")

    async def _get_linkedin_posts(self, profile_url: str) -> str:
        """Retrieve LinkedIn posts using Apify"""
        run_input = {
            "username": profile_url,
            "maxPosts": 20,
            "timeout": 60
        }
        
        try:
            run = await asyncio.to_thread(
                self.clients.apify_client.actor('apimaestro/linkedin-profile-posts').call,
                run_input=run_input
            )
            dataset = await asyncio.to_thread(
                self.clients.apify_client.dataset(run["defaultDatasetId"]).list_items
            )
            return self._format_social_posts(dataset.items, "linkedin")
        except Exception as e:
            logger.warning(f"LinkedIn posts retrieval failed: {str(e)}")
            return ""

    async def _get_twitter_posts(self, profile_url: str) -> str:
        """Retrieve Twitter posts using Apify"""
        # Extract username from URL (x.com or twitter.com)
        username = profile_url.rstrip('/').split('/')[-1].split('?')[0]
        
        run_input = {
            "max_posts": 50,
            "username": username
        }
        
        try:
            logger.info(f"Getting Twitter posts for username: {username}")
            run = await asyncio.to_thread(
                self.clients.apify_client.actor("danek/twitter-scraper-ppr").call,
                run_input=run_input
            )
            
            dataset = await asyncio.to_thread(
                self.clients.apify_client.dataset(run["defaultDatasetId"]).list_items
            )
            
            # Check if we received any data
            if not dataset.items:
                logger.warning(f"No Twitter posts found for {username}")
                return f"No tweets found for this profile (@{username}). The account may be protected, inactive, or not exist."
                
            return self._format_social_posts(dataset.items, "twitter")
        except Exception as e:
            logger.warning(f"Twitter posts retrieval failed: {str(e)}")
            return f"Error retrieving tweets for @{username}: {str(e)}"

    def _format_social_posts(self, posts: List[Dict], platform: str) -> str:
        """Format social posts for analysis"""
        if not posts:
            return f"No {platform} posts found."
            
        formatted = []
        for idx, post in enumerate(posts[:10]):  # Limit to 10 posts
            if platform == "linkedin":
                text = post.get("text", "")[:500]  # Truncate long posts
                date = post.get("posted_at", {}).get("date", "Unknown date")
            elif platform == "twitter":
                text = post.get("text", "")[:500]  # Both actors use 'text' field
                date = post.get("created_at", "Unknown date")
                # Add some additional Twitter metadata if available
                favorites = post.get("favorites", "")
                retweets = post.get("retweets", "")
                if favorites or retweets:
                    engagement = f" | ❤️ {favorites}" if favorites else ""
                    engagement += f" | 🔄 {retweets}" if retweets else ""
                    text = f"{text}\n{engagement}"
            
            formatted.append(f"Post {idx+1} ({date}):\n{text}\n")
        
        if not formatted:
            return f"No {platform} posts could be extracted."
            
        return "\n".join(formatted)

    async def get_other_appearances(self, state: GuestResearchState) -> GuestResearchState:
        """Search for other podcast appearances"""
        if not os.getenv("LISTENNOTES_API_KEY"):
            raise HTTPException(status_code=500, detail="ListenNotes API key not configured.")

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
                    raise HTTPException(status_code=502, detail="ListenNotes search failed")

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
            # Use the more detailed report generation approach from app.py
            state["report"] = await self._generate_detailed_report(state)
            return state
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Report generation failed")
            
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

        # LLM call 3: Create Podcast Questions
        questions_prompt = f"""
        Based on the content of the posts below, generate a list of thoughtful and engaging questions to ask the guest on a podcast:
        
        LinkedIn Post: {linkedin_post} and \nLinkedIn Summary: {linkedin_profile}
        Twitter Post: {twitter_post}
        
        The questions should prompt discussion and reveal interesting insights.
        """
        questions_response = await self.clients.llm.ainvoke([
            SystemMessage(content=questions_prompt),
            HumanMessage(content=f"Guest name: {guest_name}")
        ])
        questions = questions_response.content.strip()

        # LLM call 4: Structure Previous Podcast Appearances
        appearances_prompt = f"""
        Using the following information, create a well-structured Markdown section listing the guest's previous podcast appearances:
        
        {other_appearances}
        
        Format it with bullet points or a table if needed.
        """
        appearances_response = await self.clients.llm.ainvoke([
            SystemMessage(content=appearances_prompt),
            HumanMessage(content=f"Guest name: {guest_name}")
        ])
        appearances = appearances_response.content.strip()

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
            
            # Update Airtable
            await self._update_airtable(state)
            
            return state
        except Exception as e:
            logger.error(f"Results saving failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Results saving failed")

    async def _save_to_drive(self, state: GuestResearchState) -> str:
        """Save report to Google Drive"""
        doc_title = f"{state['guest_name']} Research Report"
        return self.clients.docs_service.create_document(
            title=doc_title,
            content=state["report"],
            folder_id=os.getenv("GOOGLE_REPORT_FOLDER_ID")
        )

    async def _update_airtable(self, state: GuestResearchState):
        """Update Airtable record"""
        # Only update with the document url field, which is confirmed to exist
        update_data = {
            "document url": state["document_url"]
        }
        self.clients.podcast_service.update_record(
            "Guest Research bot",
            state["record_id"],
            update_data
        )

    async def _get_linkedin_profile(self, profile_url: str) -> str:
        """Extract LinkedIn profile information using Apify"""
        try:
            # Call the LinkedIn profile scraper actor in a blocking way inside a thread
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

            # Get the dataset items (the scraped data)
            dataset = await asyncio.to_thread(
                self.clients.apify_client.dataset(run["defaultDatasetId"]).list_items
            )

            # Check if we got any data
            if not dataset or not dataset.items:
                return "No profile data found."

            # For this example, we assume the first item in the dataset is our profile.
            profile = dataset.items[0]

            # Extracting basic profile details
            full_name = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
            headline = profile.get("headline", "N/A")
            summary = profile.get("summary", "N/A")
            occupation = profile.get("occupation", "N/A")
            current_job = profile.get("jobTitle", "N/A")
            current_company = profile.get("companyName", "N/A")
            location = profile.get("geoLocationName", "N/A")
            connections = profile.get("connectionsCount", "N/A")
            followers = profile.get("followersCount", "N/A")

            # Extract education details
            educations = profile.get("educations", [])
            education_details = "\n".join(
                f"• {edu.get('schoolName', '')} - {edu.get('degreeName', 'No Degree')} in {edu.get('fieldOfStudy', 'Unknown Field')}"
                for edu in educations
            ) or "N/A"

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

            # Extract certifications
            certifications = profile.get("certifications", [])
            certs_details = "\n".join(
                f"• {cert.get('name', 'N/A')} by {cert.get('authority', 'Unknown Authority')}"
                for cert in certifications
            ) or "N/A"

            # Extract skills (list joined by commas)
            skills = profile.get("skills", [])
            skills_details = ", ".join(skills) if skills else "N/A"

            # Extract volunteer experiences
            volunteer_experiences = profile.get("volunteerExperiences", [])
            volunteer_details = "\n".join(
                f"• {vol.get('role', 'N/A')} at {vol.get('companyName', 'N/A')}"
                for vol in volunteer_experiences
            ) or "N/A"

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

            return result
        except Exception as e:
            logger.warning(f"LinkedIn profile extraction failed: {str(e)}")
            return ""

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
        ("get_other_appearances", engine.get_other_appearances),
        ("find_social_profiles", engine.find_social_profiles),
        ("retrieve_social_content", engine.retrieve_social_content),
        ("generate_research_report", engine.generate_research_report),
        ("save_results", engine.save_results)
    ]

    # Add nodes to workflow
    for node_name, node_func in nodes:
        builder.add_node(node_name, node_func)

    # Configure workflow edges
    if entry_point == "get_episode_from_rss":
        builder.set_entry_point("get_episode_from_rss")
        builder.add_edge("get_episode_from_rss", "extract_guest_info")
    else:
        builder.set_entry_point("get_episode_from_podscan")
        builder.add_edge("get_episode_from_podscan", "extract_guest_info")

    builder.add_edge("extract_guest_info", "get_other_appearances")
    builder.add_conditional_edges(
        "get_other_appearances",
        determine_social_media_search,
        {
            "search_social": "find_social_profiles",
            "skip_search": "retrieve_social_content"
        }
    )
    builder.add_edge("find_social_profiles", "retrieve_social_content")
    builder.add_edge("retrieve_social_content", "generate_research_report")
    builder.add_edge("generate_research_report", "save_results")
    builder.add_edge("save_results", END)

    return builder.compile(checkpointer=MemorySaver())

# ---------------------------
# FastAPI Endpoint Implementation
# ---------------------------

def format_research_results(state: GuestResearchState) -> Dict[str, str]:
    """Format final research results for API response"""
    return {
        "guest": state["guest_name"],
        "linkedin": state["linkedin_url"],
        "document_url": state["document_url"],
        "state": state
    }

app = FastAPI(title="Podcast Guest Research Automator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
async def service_status():
    """Service health check endpoint"""
    return {"status": "active", "version": "1.0.0"}

@app.get("/research-guest", tags=["Research"])
async def research_guest_endpoint(
    episode_title: str,
    record_id: str,
    rss_feed: Optional[str] = None,
    search_guest_name: Optional[str] = None,
    host_podcast: Optional[str] = None
) -> Dict[str, str]:
    """
    Initiate guest research workflow for a podcast episode
    
    Parameters:
    - episode_title: Exact title of the podcast episode
    - record_id: Airtable record ID for saving results
    - rss_feed: Optional RSS feed URL for podcast episodes
    - search_guest_name: Optional guest name for RSS feed search
    - host_podcast: Optional name of the host podcast
    
    Returns:
    - Research results with document URL and guest details
    """
    try:
        # Choose entry point based on provided parameters
        entry_point = "get_episode_from_rss" if rss_feed and search_guest_name else None
        
        # Initialize workflow and state
        workflow = create_research_workflow(entry_point)
        session_id = str(uuid.uuid4())
        engine = PodcastResearchEngine()
        initial_state = await engine.initialize_state(
            episode_title=episode_title,
            record_id=record_id,
            rss_feed=rss_feed,
            search_guest_name=search_guest_name
        )
        
        # Add host_podcast to state if provided
        if host_podcast:
            initial_state["host_podcast"] = host_podcast
        
        # Run workflow
        results = await workflow.ainvoke(
            initial_state,
            config=RunnableConfig(
                configurable={
                    "thread_id": session_id,
                    "checkpoint_ns": "research_session"
                }
            )
        )
        
        # Update Airtable record
        table_name = "Guest Research bot"
        podcast_client = PodcastService()
        podcast_client.update_record(table_name, record_id, {"document url": results["document_url"]})
        
        # Reinitialize the checkpointer by replacing it with a new MemorySaver instance
        workflow.checkpointer = MemorySaver()
        
        # Format and return results
        return format_research_results(results)
    
    except Exception as e:
        logger.error(f"Research failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)