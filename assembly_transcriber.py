import os
from dotenv import load_dotenv
import requests
import time
import re
from urllib.parse import urlparse
import gc  # Import garbage collection
import uuid

load_dotenv()
# Directory where audio files will be saved temporarily
DOWNLOAD_DIR = "audio_files"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# AssemblyAI API key
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')

# Header for downloads, including a User-Agent to mimic a browser
download_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}


def extract_gdrive_file_id(url):
    """Extracts Google Drive file ID from various URL formats"""
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",  # Standard URL format
        r"id=([a-zA-Z0-9_-]+)",  # Direct download format
        r"folders/([a-zA-Z0-9_-]+)"  # Folder format (though this is for completeness)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def upload_to_assemblyai(audio_file_path):
    """Uploads audio file to AssemblyAI and returns the file URL."""
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    with open(audio_file_path, 'rb') as f:
        response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, files={'file': f})
    response.raise_for_status()
    print("The file has been uploaded to Assemblyai")
    return response.json()['upload_url']

def transcribe_with_assemblyai(audio_url, max_wait_minutes=15):
    """Initiates transcription with AssemblyAI and returns the completed transcription with speaker labels."""
    endpoint = "https://api.assemblyai.com/v2/transcript"
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    json_data = {
        'audio_url': audio_url,
        'speaker_labels': True
    }

    response = requests.post(endpoint, headers=headers, json=json_data)
    response.raise_for_status()
    transcript_id = response.json()['id']

    start_time = time.time()

    # Polling for the completed transcription
    while True:
        response = requests.get(f"{endpoint}/{transcript_id}", headers=headers)
        response.raise_for_status()
        transcript_data = response.json()
        if transcript_data['status'] == 'completed':
            return transcript_data
        elif transcript_data['status'] == 'failed':
            raise Exception(f"Transcription failed: {transcript_data['error']}")
        
        # Check how long we’ve been polling
        elapsed_time = time.time() - start_time
        if elapsed_time > max_wait_minutes * 60:
            # If we want to skip silently instead of raising an error,
            # replace this with a return None or similar handling.
            raise TimeoutError(
                f"Transcription is taking too long (more than {max_wait_minutes} minutes)."
            )
        print("Polling AssemblyAI for transcription status...")
        time.sleep(10)  # Waits before polling again

def format_transcription(transcript_data):
    """Formats the transcription output with speaker labels in the specified format."""
    formatted_transcript = ""
    current_speaker = None  # Tracks the current speaker to avoid redundant speaker labels

    for word in transcript_data['words']:
        speaker = word.get('speaker', 'Speaker')  # Default to 'Speaker' if no label is found
        start_time = word['start'] / 1000  # Convert ms to seconds
        word_text = word['text']

        # Add new speaker and timestamp if speaker changes
        if speaker != current_speaker:
            formatted_transcript += f"\n\n[{speaker}] {start_time:.2f}s:\n"  # New speaker line with timestamp
            current_speaker = speaker

        formatted_transcript += f"{word_text} "  # Append word to the line

    return formatted_transcript.strip()  # Remove extra whitespace

def delete_audio_file(audio_file_path):
    try:
        os.remove(audio_file_path)
        print(f"Deleted audio file: {audio_file_path}")
    except Exception as e:
        print(f"Error deleting audio file: {e}")


def process_audio_file(audio_url: str) -> str:
    """Process an audio URL and return path to transcript"""
    try:
        # Generate unique filename
        file_uuid = str(uuid.uuid4())
        parsed_url = urlparse(audio_url)
        original_filename = parsed_url.path.split('/')[-1]
        
        # Preserve extension if available
        if '.' in original_filename:
            ext = original_filename.split('.')[-1]
            audio_filename = f"{file_uuid}.{ext}"
        else:
            audio_filename = f"{file_uuid}.mp3"

        audio_file_path = os.path.join(DOWNLOAD_DIR, audio_filename)
        
        # Download audio file
        print(f"Downloading audio file from {audio_url}")
        response = requests.get(audio_url, headers=download_headers)
        
        if response.status_code == 404:
            raise ValueError(f"URL not found - {audio_url}")
            
        response.raise_for_status()

        with open(audio_file_path, 'wb') as f:
            f.write(response.content)

        print(f"Audio file saved to: {audio_file_path}")

        # Transcribe the audio file
        try:
            audio_url = upload_to_assemblyai(audio_file_path)
            transcript_data = transcribe_with_assemblyai(audio_url, max_wait_minutes=30)
        except TimeoutError as e:
            raise Exception("Transcription timeout") from e

        transcription_text = format_transcription(transcript_data)


        return transcription_text

    except Exception as e:
        # Clean up audio file if it exists
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        raise

    finally:
        # Always clean up audio file
        if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            print(f"Cleaned up audio file: {audio_file_path}")


transcription_text = process_audio_file("https://dts.podtrac.com/redirect.mp3/pdst.fm/e/mgln.ai/e/35/p.podderapp.com/1353134996/media.blubrry.com/3808867/arttrk.com/p/LBS4N/api.spreaker.com/download/episode/64502124/ricochet_729.mp3")

print(transcription_text)