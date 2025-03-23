import requests
from bs4 import BeautifulSoup

# Custom headers to mimic a browser
custom_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
url = "https://feeds.simplecast.com/dFXndvsL"
# Fetch the RSS feed
def fetch_rss_feed(url):
    """
    Fetches an RSS feed and extracts podcast metadata and episodes. 
    """
    response = requests.get(url, headers=custom_headers)

    if response.status_code != 200:
        print("Failed to retrieve the RSS feed.")
        exit()

    # Parse the XML content
    soup = BeautifulSoup(response.content, 'xml')

    # Extract podcast metadata
    channel = soup.channel
    podcast_title = channel.title.text if channel.title else "No title"
    podcast_description = channel.description.text if channel.description else "No description"

    # Extract episodes
    episodes = []
    for item in soup.find_all('item'):
        title = item.title.text if item.title else "No title"
        pub_date = item.pubDate.text if item.pubDate else "No publication date"
        description = item.description.text if item.description else "No description"
        
        # iTunes-specific elements
        duration = item.find('itunes:duration')
        duration = duration.text if duration else "No duration"
        
        # Enclosure (audio URL)
        enclosure = item.enclosure
        enclosure_url = enclosure['url'] if enclosure and 'url' in enclosure.attrs else "No URL"
        enclosure_type = enclosure['type'] if enclosure and 'type' in enclosure.attrs else "No type"
        enclosure_length = enclosure['length'] if enclosure and 'length' in enclosure.attrs else "No length"

        episodes.append({
            'title': title,
            'pub_date': pub_date,
            'description': description,
            'duration': duration,
            'enclosure_url': enclosure_url,
            'enclosure_type': enclosure_type,
            'enclosure_length': enclosure_length
        })
    return podcast_title, episodes


if __name__ == "__main__":
    podcast_title, episodes = fetch_rss_feed(url)
    print(episodes)