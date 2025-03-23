import os
import re
import logging
import asyncio
import uuid
import tiktoken
import datetime
from dataclasses import asdict, dataclass
from dotenv import load_dotenv
from typing import Annotated, List, Optional, Literal, Union, Dict, Any, get_type_hints, get_origin, get_args
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
import uvicorn
from rss_scraper import fetch_rss_feed
from google_service import GoogleDocsService
from airtable_services import PodcastService

# LangChain imports
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from apify_client import ApifyClient
from langchain_core.runnables import RunnableConfig
import json


import requests
#from langgraph.checkpoint.base import AsyncMemorySaver
from langgraph.checkpoint.memory import MemorySaver
import aiohttp  

from fastapi import FastAPI, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# -------------------------------------------------------------------

app = FastAPI()

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LISTENNOTES_API_KEY = os.getenv("LISTENNOTES_API_KEY")
PODSCANAPI = os.getenv("PODSCANAPI")

client = ApifyClient(os.getenv("APIFY_CLIENT_TOKEN"))
tavily_search = TavilySearchAPIWrapper()
chat_llm = ChatOpenAI(model_name="o3-mini", temperature=None, openai_api_key=OPENAI_KEY, reasoning_effort="medium")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0)

#Data Model
class SocialMediaProfile(BaseModel):
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    twitter: Optional[str] = Field(None, description="X profile URL (x.com/{username})")
    name: str = Field(..., description="Full name of the person")

class guestIdentification(BaseModel):
    guest: str = Field(description="Identify who the guest is and extract their full names as a string")
    guest_unique_element: str =Field(description="A keyword that can be used to identify the user online, like their niche")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    twitter: Optional[str] = Field(None, description="X profile URL (x.com/{username})")

class ResearchRequest(BaseModel):
    episode_title: str

class GuestResearchState(TypedDict):
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



from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing only!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def aget_episode(episode_title: str) -> str:
    if not PODSCANAPI:
        raise HTTPException(status_code=500, detail="Podscan API key not configured.")

    async with aiohttp.ClientSession() as session:
        params = {
            "query": f'"{episode_title}"',
            'podcast_language': "en",
            "per_page": 10
        }
        headers = {"Authorization": f"Bearer {PODSCANAPI}"}

        async with session.get(
            "https://podscan.fm/api/v1/episodes/search",
            headers=headers,
            params=params
        ) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=502, detail="Podscan search failed")

            results = await resp.json()
            return results


async def aextract_past_twitter_post(twitter_url: str) -> str:

    # Wrap blocking calls in thread executor
    run = await asyncio.to_thread(
        client.actor("apidojo/tweet-scraper").call,
        run_input = {
            "startUrls": [twitter_url],
            "maxItems": 25,
            "excludeReplies": True,
            "excludeRetweets": True,
            "tweetLanguage": "en",
            "searchMode": "live" 
        }
    )

    # Get dataset items
    dataset = await asyncio.to_thread(
        client.dataset(run["defaultDatasetId"]).list_items
    )

    return "\n".join(
        f"Post {idx}: {item.get('text', '')}\nDate: {item.get('createdAt', '')}"
        for idx, item in enumerate(dataset.items)
    )


async def aextract_past_linkedin_post(linkedin_url: str) -> str:

    # Wrap blocking calls in thread executor
    run = await asyncio.to_thread(
        client.actor('apimaestro/linkedin-profile-posts').call,
        run_input={"username": linkedin_url}
    )

    # Get dataset items
    dataset = await asyncio.to_thread(
        client.dataset(run["defaultDatasetId"]).list_items
    )

    return "\n".join(
        f"Post {idx}: {item.get('text', '')}\nDate: {item.get('posted_at', {}).get('date', '')}"
        for idx, item in enumerate(dataset.items)
    )

async def aextract_linkedin_profile(linkedin_url: str) -> str:
    # Call the LinkedIn profile scraper actor in a blocking way inside a thread
    run = await asyncio.to_thread(
        client.actor('supreme_coder/linkedin-profile-scraper').call,
        run_input={
            "findContacts": False,
            "scrapeCompany": False,
            "urls": [
                {
                    "url": linkedin_url,
                    "method": "GET"
                }
            ]
        }
    )

    # Get the dataset items (the scraped data)
    dataset = await asyncio.to_thread(
        client.dataset(run["defaultDatasetId"]).list_items
    )

    # Check if we got any data
    if not dataset:
        return "No profile data found."

    # For this example, we assume the first item in the dataset is our profile.
    profile = dataset[0]

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


# 2. Async version of Tavily search
async def async_tavily_search(query: str, max_results: int = 5):
    return await asyncio.to_thread(
        tavily_search.results,
        query,
        max_results=max_results,
        search_depth="advanced"
    )

async def get_guest_info(description: str):
    prompt = (
        f"""
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
        """
    )
    structured_llm = llm.with_structured_output(guestIdentification)
    return await structured_llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=description)
    ])


async def get_guest_info_from_transcript(transcript: str) -> dict:
    # Create the prompt for guest identification (guest name, unique element, and social handles)
    guest_prompt = (
        f"""
        Given a transcript of a podcast episode, identify who the guest is. 
        If you can't confidently identify the guest, return None.
        Also extract a unique element: a keyword that can be used to identify the guest online.
        """
    )
    structured_llm = llm.with_structured_output(guestIdentification)
    guest_info_task = structured_llm.ainvoke([
        SystemMessage(content=guest_prompt),
        HumanMessage(content=transcript)
    ])

    # Create the prompt for guest reason (what the guest is trying to sell/market or their underlying motive)
    reason_prompt = (
        f"""
        Given the same podcast transcript, identify what the guest is trying to sell or market.
        I want to know the underlying reason for why the guest is on the show, what offer do they have.
        The idea is to find something the guest is passionate about that will allow them to come talk about it on our show
        """
    )
    guest_reason_task = llm.ainvoke([
        SystemMessage(content=reason_prompt),
        HumanMessage(content=transcript)
    ])

    guest_value_prompt = (
        f"""
        Given the same podcast transcript, identify what business, product or service the guest owns.
        """
    )
    guest_value_task = llm.ainvoke([
        SystemMessage(content=guest_value_prompt),
        HumanMessage(content=transcript)
    ])

    # Run both LLM calls concurrently
    guest_info_result, guest_reason_result, guest_value_result = await asyncio.gather(guest_info_task, guest_reason_task, guest_value_task)

    # You might want to further process guest_reason_result.content if necessary.
    return {
        "guest_info": guest_info_result,
        "guest_reason": guest_reason_result.content,  # storing as a string
        "guest_value": guest_value_result.content
    }


async def agenerate_profile_topic(state: GuestResearchState) -> str:
    """
    Generate a detailed profile report by dividing the work among multiple LLM calls:
    - An introduction with social media handles, company, and offers.
    - A summary of topics and themes from the posts.
    - A set of engaging podcast questions for the guest.
    - A structured summary of the guest's previous podcast appearances.
    """
    # Extracting required state data
    host_podcast_name = state["host_podcast"]
    linkedin_profile = state["linkedin_profile"]
    linkedin_post = state["linkedIn_post"]
    twitter_post = state["twitter_post"]
    guest_name = state["guest_name"]
    twitter_url = state["twitter_url"]
    linkedin_url = state["linkedin_url"]
    other_appearances = state["episodes"]
    company = state["guest_company"]
    offers = state["guest_reason"]

    # LLM call 1: Create Introduction
    introduction_prompt = f"""
    Write an introduction for the guest. Include their social media handles:
    - LinkedIn: {linkedin_url} and \nLinkedIn Summary: {linkedin_profile}
    - Twitter: {twitter_url}
    
    Also mention their company, {company}, and any offers if applicable: {offers}.
    Provide a warm, engaging introduction.
    """
    introduction_response = await llm.ainvoke([
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
    summary_response = await llm.ainvoke([
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
    questions_response = await llm.ainvoke([
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
    appearances_response = await llm.ainvoke([
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




def transcribe_from_url(audio_url: str, payload: dict) -> str:
    """
    Calls the Replit transcription endpoint with the given audio URL and payload.
    
    Parameters:
      audio_url (str): The URL of the audio file to be transcribed.
      payload (dict): A dictionary that may include keys like 'episode_name' and 'speakers'.
                      For example: {"episode_name": "My Podcast", "speakers": "Host,Guest"}
    
    Returns:
      str: The transcript returned by the transcription service.
    """
    # The endpoint of your Replit transcription service
    endpoint = "http://localhost:8000/transcribe"
    
    # Prepare the form data payload
    form_data = {
        "audio_url": audio_url,
        "episode_name": payload.get("episode_name", "")
    }
    if payload.get("speakers"):
        form_data["speakers"] = payload["speakers"]

    # Make the POST request
    response = requests.post(endpoint, data=form_data)
    
    if response.status_code != 200:
        raise Exception(f"Transcription endpoint error: {response.status_code} {response.text}")
    
    # Parse and return the transcript from the JSON response
    result = response.json()
    transcript = result.get("transcript", "")
    return transcript

async def downlaod_transcript_from_drive(transcript_url: str) -> str:
    """
    Download the transcript from the given URL.
    """
    # Extract the document ID from the URL using a regular expression.
    pattern = r"/d/([a-zA-Z0-9_-]+)"
    match = re.search(pattern, transcript_url)
    if not match:
        raise ValueError("Invalid Google Drive URL provided.")
    
    document_id = match.group(1)
    
    # Instantiate the GoogleDocsService.
    google_service = GoogleDocsService()
    
    # Since get_document_content is a synchronous method, we run it in an executor.
    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(None, google_service.get_document_content, document_id)
    
    return content



async def check_airtable_for_transcript(state: GuestResearchState):
    airtable_client = PodcastService()
    table_name = "Podcast"
    episode_title = state.get("episode_title")
    guest_name = state.get("search_guest_name")
    formula = (
        "OR("
        "IFERROR(FIND('" + episode_title + "', {Podcast Title}), 0) > 0, "
        "IFERROR(FIND('" + guest_name + "', {Podcast Title}), 0) > 0"
        ")"
    )

    records = airtable_client.search_records(table_name, formula)
    
    # Iterate through records and process the first one with a transcript URL.
    for record in records:
        fields = record.get("fields", {})
        transcript_url = fields.get("Podcast Transcription")
        if transcript_url:
            transcript = await downlaod_transcript_from_drive(transcript_url)
            state["transcript"] = transcript
            break  # Stop after processing the first valid record.


async def get_episode_transcript_from_rss_feed(state: GuestResearchState) -> str:
    """
    Extract the episode transcript from the RSS feed.
    """
    rss_feed = state.get("rss_feed")
    search_guest_name = state.get("search_guest_name")
    episode_title = state.get("episode_title")

    if not rss_feed:
        raise ValueError("No RSS feed provided in state.")

    # Fetch the RSS feed and extract the episode transcript
    podcast_title, episodes = fetch_rss_feed(rss_feed)

    # Find the episode with the matching title
    episode = next((ep for ep in episodes if search_guest_name in ep["title"] or search_guest_name in ep["description"]), None)

    if not episode:
        raise ValueError(f"No episode found with title: {episode_title}")

    episode_url = episode.get("enclosure_url")
    payload = {"episode_name": episode_title}
    episode_transcript = transcribe_from_url(episode_url, payload)

    state["transcript"] = episode_transcript
    state["episode_description"] = episode.get("description", "")
    return state

async def save_report_to_google_drive(state: GuestResearchState) -> str:
    """
    Save the generated report (state["report"]) to a Google Drive folder
    using the GoogleDocsService. Returns the URL of the created document.
    """
    # Ensure there is a report to save.
    report = state.get("report")
    if not report:
        raise ValueError("No report available in state to save.")

    # Create a title for the document using the guest's name.
    guest_name = state.get("guest_name", "Unknown Guest")
    title = f"Report for {guest_name}"

    # Instantiate the GoogleDocsService.
    google_docs_service = GoogleDocsService()

    # Retrieve the folder ID from the environment.
    folder_id = os.getenv('GOOGLE_REPORT_FOLDER_ID')

    # Use the create_document method to create a new Google Doc with the report content.
    document_url = google_docs_service.create_document(title=title, content=report, folder_id=folder_id)

    print(f"Report successfully saved. Document URL: {document_url}")
    state["document_url"] = document_url

    return state


async def get_other_guest_appearances(state: GuestResearchState):
    """Search ListenNotes for episodes featuring the given guest, return other podcast names."""

    if not LISTENNOTES_API_KEY:
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

        headers = {"X-ListenAPI-Key": LISTENNOTES_API_KEY}

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

async def retrieve_transcript_and_description(state: GuestResearchState):

    results = await aget_episode(state["episode_title"])
    for episode in results.get("episodes", []):
        if state["episode_title"] in episode.get("episode_title"):
            print(episode.get("episode_title"))
            print(episode.get("episode_transcript", "")[:100])
            if episode.get("episode_description"):
                state["episode_description"] = episode.get("episode_description", "")

            if episode.get("episode_transcript"):
                state["transcript"] = episode.get("episode_transcript", "")


    return state


async def extract_guest_info(state: GuestResearchState):
    try:
        tasks = []
        # Description extraction: use if available but as a fallback for social handles
        if state.get("episode_description"):
            tasks.append(get_guest_info(state["episode_description"]))
        else:
            tasks.append(asyncio.sleep(0, result=None))  # placeholder if description is missing

        # Always try transcript extraction as primary (with its richer context)
        tasks.append(get_guest_info_from_transcript(state["transcript"]))

        # Unpack results; note: transcript info is second in the list
        desc_info, transcript_result = await asyncio.gather(*tasks)

        # Extract the guest info from the transcript result
        transcript_info = transcript_result.get("guest_info")
        transcript_reason = transcript_result.get("guest_reason")
        guest_value = transcript_result.get("guest_value")

        # Merge the two results, prioritizing transcript info for guest details,
        # but falling back to description info for social handles if transcript data is missing them.
        merged_info = {
            "guest": transcript_info.guest if transcript_info and transcript_info.guest else (desc_info.guest if desc_info and desc_info.guest else None),
            "guest_unique_element": transcript_info.guest_unique_element if transcript_info and transcript_info.guest_unique_element else (desc_info.guest_unique_element if desc_info and desc_info.guest_unique_element else None),
            "linkedin": transcript_info.linkedin if transcript_info and transcript_info.linkedin else (desc_info.linkedin if desc_info and desc_info.linkedin else None),
            "twitter": transcript_info.twitter if transcript_info and transcript_info.twitter else (desc_info.twitter if desc_info and desc_info.twitter else None),
            "guest_reason": transcript_reason  # using transcript's reason directly
        }

        # Ensure a guest name was identified from either source
        if not merged_info["guest"]:
            raise ValueError("No guest identified from transcript or description")

        # Update state with merged guest info
        state["guest_name"] = merged_info["guest"]
        state["guest_unique_element"] = merged_info["guest_unique_element"]
        state["guest_reason"] = merged_info["guest_reason"]
        state["guest_company"] = guest_value

        if merged_info["linkedin"]:
            state["is_linkedin_url"] = True
            state["linkedin_url"] = merged_info["linkedin"]

        if merged_info["twitter"]:
            state["is_twitter_url"] = True
            state["twitter_url"] = merged_info["twitter"]

        print(f"Guest unique element is {state['guest_unique_element']}")
        return state
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Guest identification failed: {str(e)}"
        )


async def find_social_media(state: GuestResearchState):

    # Compile regex patterns for validation
    linkedin_pattern = re.compile(
        r'^https?://(www\.)?linkedin\.com/in/[a-z0-9-]+/?(\?.*)?$', 
        re.IGNORECASE
    )
    twitter_pattern = re.compile(
        r'^https?://(www\.)?(twitter\.com|x\.com)/[a-z0-9_]+/?(\?.*)?$', 
        re.IGNORECASE
    )

    # Use async pattern matching
    linkedin_results, twitter_results = await asyncio.gather(
        async_tavily_search(f"{state['guest_name']} {state['guest_unique_element']} LinkedIn"),
        async_tavily_search(f"{state['guest_name']} {state['guest_unique_element']} Twitter")
    )

    # Parallel URL validation
    async def validate_url(pattern, results):
        for result in results:
            if re.match(pattern, result.get("url", "")):
                return result["url"]
        return ""

    state["linkedin_url"], state["twitter_url"] = await asyncio.gather(
        validate_url(linkedin_pattern, linkedin_results),
        validate_url(twitter_pattern, twitter_results)
    )
    return state

async def retrieve_social_media_content(state: GuestResearchState):
    state["linkedIn_post"] = await aextract_past_linkedin_post(state["linkedin_url"]) if state["linkedin_url"] else ""
    state["twitter_post"] = await aextract_past_twitter_post(state["twitter_url"]) if state["twitter_url"] else ""
    state["linkedin_profile"] = await aextract_linkedin_profile(state["linkedin_url"]) if state["linkedin_url"] else ""
    return state

async def generate_guest_report(state: GuestResearchState):

    state["report"] = await agenerate_profile_topic(state)
    return state

def guest_routing(state: GuestResearchState):
    if not state.get("is_linkedin_url", False) and not state.get("is_twitter_url", False):
        return "find_social_media"
    else:
        return "retrieve_social_media_content"


def initialize_graph(entry_point: str = None) -> StateGraph:
    builder = StateGraph(GuestResearchState)

    async_nodes = [
        ("get_episode_transcript_from_rss_feed", get_episode_transcript_from_rss_feed),
        ("retrieve_transcript_and_description", retrieve_transcript_and_description),
        ("extract_guest_info", extract_guest_info),
        ("find_social_media", find_social_media),
        ("retrieve_social_media_content", retrieve_social_media_content),
        ("generate_guest_report", generate_guest_report),
        ("get_other_guest_appearances", get_other_guest_appearances),
        ("save_report_to_google_drive", save_report_to_google_drive)
    ]

    for node_name, node_func in async_nodes:
        print(f"Adding node: {node_name} -> Type: {type(node_func)}")
        if not callable(node_func):
            raise TypeError(f"Invalid node: {node_name} is not callable")
        builder.add_node(node_name, node_func)
    
    # Use provided entry point if given, otherwise default to "retrieve_transcript_and_description"
    if entry_point:
        builder.set_entry_point(entry_point)
    else:
        builder.set_entry_point("retrieve_transcript_and_description")
    
    # Conditionally add the edge from the chosen entry point to "extract_guest_info"
    if entry_point == "get_episode_transcript_from_rss_feed":
        builder.add_edge("get_episode_transcript_from_rss_feed", "extract_guest_info")
    else:
        builder.add_edge("retrieve_transcript_and_description", "extract_guest_info")
    
    builder.add_edge("extract_guest_info", "get_other_guest_appearances")
    builder.add_conditional_edges(
        "get_other_guest_appearances",
        guest_routing,
        {
            "find_social_media": "find_social_media",
            "retrieve_social_media_content": "retrieve_social_media_content"
        }
    )
    builder.add_edge("find_social_media", "retrieve_social_media_content")
    builder.add_edge("retrieve_social_media_content", "generate_guest_report")
    builder.add_edge("generate_guest_report", "save_report_to_google_drive")
    builder.add_edge("save_report_to_google_drive", END)

    return builder.compile(checkpointer=MemorySaver())



@app.get("/")
def index():
    """
    Root endpoint that returns a simple JSON message.
    """
    return {"message": "awaiting podcast episode title (FastAPI version)!"}


@app.get("/research-guest")
async def research_guest_endpoint(episode_title: str, 
                                  record_id: str, 
                                  rss_feed: Optional[str] = None, 
                                  search_guest_name: Optional[str] = None,
                                  host_podcast: Optional[str] = None
                                  ) -> dict:
    
    try:
        # Choose entry point based on provided parameters
        if rss_feed and search_guest_name:
            entry_point = "get_episode_transcript_from_rss_feed"
        else:
            entry_point = "retrieve_transcript_and_description"
        
        workflow = initialize_graph(entry_point=entry_point)
        session_id = str(uuid.uuid4())

        # Fresh state for each request
        initial_state = GuestResearchState(
            host_podcast = host_podcast or "",
            rss_feed = rss_feed or "",
            search_guest_name = search_guest_name or "",
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
            twitter_post="",
            guest_unique_element="",
            guest_reason="",
            report="",
            document_url="",
            record_id = record_id,
        )

        # Run the workflow
        result = await workflow.ainvoke(
            initial_state,
            config=RunnableConfig(
                configurable={
                    "thread_id": session_id,
                    "checkpoint_ns": "research_session"
                }
            )
        )

        table_name = "Guest Research bot"
        
        podcast_client = PodcastService()
        podcast_client.update_record(table_name, record_id, {"document url": result["document_url"]})
        
        
        # Reinitialize the checkpointer by replacing it with a new MemorySaver instance
        workflow.checkpointer = MemorySaver()

        return {
            "guest": result["guest_name"],
            "linkedin": result["linkedin_url"],
            "document_url": result["document_url"]
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)