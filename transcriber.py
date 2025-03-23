import os
from urllib.parse import urlparse
import time
import base64
from pathlib import Path
import pydub
from pydub import AudioSegment
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
import requests  
import logging
import concurrent.futures
from functools import partial
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transcriber.log')
    ]
)
logger = logging.getLogger('transcriber')

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
MAX_WORKERS = 4  # Number of parallel workers for chunk processing

# Set the full path to the ffmpeg and ffprobe executables
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\ffmpeg\bin\ffprobe.exe"

# Only set the paths if the files exist
if os.path.exists(FFMPEG_PATH):
    AudioSegment.converter = FFMPEG_PATH
    logger.info(f"Using ffmpeg at: {FFMPEG_PATH}")
else:
    logger.warning(f"ffmpeg not found at {FFMPEG_PATH}. Using system default if available.")

if os.path.exists(FFPROBE_PATH):
    pydub.utils.get_prober_name = lambda: FFPROBE_PATH
    logger.info(f"Using ffprobe at: {FFPROBE_PATH}")
else:
    logger.warning(f"ffprobe not found at {FFPROBE_PATH}. Using system default if available.")


def setup_gemini_api():
    """Set up the Gemini API with authentication"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured")
    return genai.GenerativeModel('gemini-1.5-flash-001')


def get_safe_filename(url: str) -> str:
    parsed = urlparse(url)
    # Extract the base file name from the URL path (ignores query parameters)
    return os.path.basename(parsed.path)


def process_audio_file(file_path):
    """Process an audio file and return content for the API"""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Audio file not found: {file_path}")
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    extension = path.suffix.lower()
    mime_map = {
        '.mp3': 'audio/mp3',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
    }
    mime_type = mime_map.get(extension)
    if not mime_type:
        logger.error(f"Unsupported audio format: {extension}")
        raise ValueError(f"Unsupported audio format: {extension}")
    
    logger.info(f"Loading audio file: {file_path} ({mime_type})")
    with open(file_path, 'rb') as f:
        audio_data = f.read()
    
    return {
        "mime_type": mime_type,
        "data": base64.b64encode(audio_data).decode('utf-8')
    }


def transcribe_audio(model, audio_content, episode_name, speakers=None, chunk_id=None):
    """Transcribe audio using Gemini API with retries"""
    chunk_info = f" (chunk {chunk_id})" if chunk_id is not None else ""
    prompt = "Transcribe this podcast with speaker labels and timestamps in [HH:MM:SS] format. \
        Listen for speaker names mentioned in the conversation and use those real names as speaker labels. \
            If a speaker's name isn't mentioned, label them as 'Host', 'Guest','Narrator' etc. or 'Speaker 1', 'Speaker 2', etc."
    if speakers:
        speakers_str = ", ".join(speakers)
        prompt += f" Identify speakers as: {speakers_str}."
    if episode_name:
        prompt += f" This is an episode titled '{episode_name}'."
    prompt += " Format as [HH:MM:SS] Speaker: Text"
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Sending transcription request{chunk_info} (attempt {attempt+1}/{MAX_RETRIES})...")
            response = model.generate_content(
                [audio_content, prompt],
                generation_config={"temperature": 0.1})
            logger.info(f"Transcription complete{chunk_info}")
            return response.text
        except (ResourceExhausted, ServiceUnavailable) as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"API error{chunk_info}: {e}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to transcribe{chunk_info} after {MAX_RETRIES} attempts: {e}")
                raise


def process_audio_chunk(chunk_index, audio, total_chunks, chunk_length_ms, overlap_ms, file_suffix, episode_name, speakers=None):
    """Process a single audio chunk and return its transcript"""
    try:
        logger.info(f"Processing chunk {chunk_index+1} of {total_chunks}")
        
        if chunk_index == 0:
            start_ms = 0
        else:
            start_ms = max(chunk_index * chunk_length_ms - overlap_ms, 0)
        
        end_ms = min((chunk_index + 1) * chunk_length_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        
        # Use a temporary file that will be automatically cleaned up
        with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            chunk.export(temp_path, format="mp3")
            model = setup_gemini_api()
            audio_content = process_audio_file(temp_path)
            chunk_transcript = transcribe_audio(
                model, 
                audio_content,
                f"{episode_name} - Part {chunk_index+1}" if episode_name else f"Chunk {chunk_index+1}", 
                speakers,
                chunk_id=chunk_index+1
            )
            return chunk_index, chunk_transcript
        finally:
            # Ensure temporary file is removed
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Removed temporary file: {temp_path}")
    
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index+1}: {e}")
        return chunk_index, f"ERROR in chunk {chunk_index+1}: {str(e)}"


def process_long_audio(file_path, chunk_minutes=45, overlap_seconds=30, episode_name=None, speakers=None):
    """Process a long audio file by splitting it into manageable chunks with parallelization."""
    logger.info(f"Processing long audio file: {file_path} (chunk size: {chunk_minutes} minutes, overlap: {overlap_seconds} seconds)")
    
    audio = AudioSegment.from_file(file_path)
    chunk_length_ms = chunk_minutes * 60 * 1000
    overlap_ms = overlap_seconds * 1000
    total_chunks = len(audio) // chunk_length_ms + (1 if len(audio) % chunk_length_ms > 0 else 0)
    
    logger.info(f"Audio length: {len(audio)/1000/60:.2f} minutes, will be split into {total_chunks} chunks")
    
    file_suffix = Path(file_path).suffix
    
    # Create a partial function with fixed parameters to simplify the executor mapping
    process_chunk = partial(
        process_audio_chunk,
        audio=audio,
        total_chunks=total_chunks,
        chunk_length_ms=chunk_length_ms,
        overlap_ms=overlap_ms,
        file_suffix=file_suffix,
        episode_name=episode_name,
        speakers=speakers
    )
    
    # Use ThreadPoolExecutor to process chunks in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(process_chunk, i): i for i in range(total_chunks)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                chunk_index, transcript = future.result()
                results.append((chunk_index, transcript))
                logger.info(f"Completed chunk {chunk_index+1}/{total_chunks}")
            except Exception as e:
                logger.error(f"Chunk {chunk_index+1} processing failed: {e}")
                results.append((chunk_index, f"ERROR in chunk {chunk_index+1}: {str(e)}"))
    
    # Sort the results by chunk index to maintain the correct order
    results.sort(key=lambda x: x[0])
    transcripts = [transcript for _, transcript in results]
    
    logger.info(f"Completed processing all {total_chunks} chunks")
    return "\n\n".join(transcripts)


async def transcribe_endpoint(
    audio_url: str,
    episode_name: str = None,
    speakers: str = None
):
    """
    Function to transcribe an audio file provided by its URL.
    
    Args:
        audio_url: URL of the audio file to transcribe
        episode_name: Optional name of the episode for better identification
        speakers: Optional comma-separated list of speaker names
        
    Returns:
        Dict containing either the transcript or an error message
    """
    # Download the audio file from the provided URL
    try:
        logger.info(f"Downloading audio from URL: {audio_url}")
        
        # Add browser-like headers to prevent 403 Forbidden errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'audio/webm,audio/ogg,audio/mp3,audio/wav,audio/*;q=0.9,application/ogg;q=0.7,video/*;q=0.6,*/*;q=0.5',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.buzzsprout.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(audio_url, headers=headers)
        if response.status_code != 200:
            error_msg = f"Failed to download audio. Status code: {response.status_code}"
            logger.error(error_msg)
            return {"error": error_msg}
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return {"error": str(e)}

    # Create a temporary file for the downloaded audio
    # Parse the URL to remove query parameters when getting the file extension
    from urllib.parse import urlparse
    
    # Extract the clean file extension
    parsed_url = urlparse(audio_url)
    url_path = parsed_url.path
    file_extension = os.path.splitext(url_path)[1] or ".mp3"  # Default to .mp3 if no extension found
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(response.content)
    
    logger.info(f"Audio downloaded to temporary file: {temp_file_path}")
    
    try:
        # Estimate audio duration
        audio = AudioSegment.from_file(temp_file_path)
        duration_minutes = len(audio) / (60 * 1000)
        logger.info(f"Audio duration: {duration_minutes:.2f} minutes")
        
        max_duration = 60  # minutes; adjust as needed
        
        # Improved handling of speakers parameter
        speaker_list = None
        if speakers and speakers.strip():  # Check if speakers is not None and not empty
            speaker_list = [s.strip() for s in speakers.split(",")]
            logger.info(f"Using provided speaker list: {speaker_list}")
        else:
            logger.info("No speakers provided, AI will identify speakers automatically")
        
        if duration_minutes > max_duration:
            logger.info(f"Long audio detected ({duration_minutes:.2f} min > {max_duration} min), processing in parallel chunks")
            transcript = process_long_audio(
                temp_file_path,
                chunk_minutes=45,
                overlap_seconds=30,
                episode_name=episode_name,
                speakers=speaker_list
            )
        else:
            logger.info(f"Processing audio as a single file ({duration_minutes:.2f} min)")
            model = setup_gemini_api()
            audio_content = process_audio_file(temp_file_path)
            transcript = transcribe_audio(model, audio_content, episode_name, speaker_list)
        
        logger.info("Transcription completed successfully")
        return {"transcript": transcript}
    
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return {"error": str(e)}
    
    finally:
        # Always clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Removed temporary file: {temp_file_path}")



