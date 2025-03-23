// DOM Elements
const researchForm = document.getElementById('researchForm');
const submitBtn = document.getElementById('submitBtn');
const resultsSection = document.getElementById('resultsSection');
const loader = document.getElementById('loader');
const resultsContainer = document.getElementById('results');
const podcastFields = document.getElementById('podcastFields');
const socialFields = document.getElementById('socialFields');
const researchMethodRadios = document.querySelectorAll('input[name="researchMethod"]');

// API URL (change this if your API is hosted elsewhere)
const API_BASE_URL = window.location.origin;

// Log backend connection settings for debugging
console.log('Frontend API base URL:', API_BASE_URL);

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    // Hide results section initially
    resultsSection.style.display = 'none';
    loader.style.display = 'none';
    resultsContainer.style.display = 'none';
    
    // Set up research method toggle
    researchMethodRadios.forEach(radio => {
        radio.addEventListener('change', toggleResearchMethod);
    });
    
    // Set up form submission
    researchForm.addEventListener('submit', handleFormSubmit);
    
    // Check backend connectivity on page load
    checkBackendConnectivity();
});

// Toggle between research methods
function toggleResearchMethod() {
    const method = document.querySelector('input[name="researchMethod"]:checked').value;
    
    if (method === 'podcast') {
        podcastFields.style.display = 'block';
        socialFields.style.display = 'none';
        
        // Update required fields
        document.getElementById('episodeTitle').required = true;
        document.getElementById('directGuestName').required = false;
    } else {
        podcastFields.style.display = 'none';
        socialFields.style.display = 'block';
        
        // Update required fields
        document.getElementById('episodeTitle').required = false;
        document.getElementById('directGuestName').required = true;
    }
}

// Check if backend is reachable
async function checkBackendConnectivity() {
    try {
        // Make a simple request to the session creation endpoint to test connectivity
        const response = await fetch(`${API_BASE_URL}/research-session`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                episode_title: 'connectivity-test-' + Date.now()
            })
        });
        
        if (!response.ok) {
            console.warn('Backend connectivity issue detected. Status:', response.status);
        } else {
            console.log('Backend connectivity test passed');
        }
    } catch (error) {
        console.error('Backend connectivity test failed:', error);
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    // Get form data
    const formData = new FormData(researchForm);
    const researchMethod = formData.get('researchMethod');
    
    // Get common fields
    const hostPodcast = formData.get('hostPodcast');
    
    // Method-specific fields
    let episodeTitle, rssFeed, guestName, linkedinUrl, twitterUrl;
    
    if (researchMethod === 'podcast') {
        episodeTitle = formData.get('episodeTitle');
        rssFeed = formData.get('rssFeed');
        guestName = formData.get('guestName');
        linkedinUrl = '';
        twitterUrl = '';
        
        // Validate required fields for podcast method
        if (!episodeTitle) {
            showError('Episode title is required for podcast-based research.');
            return;
        }
    } else {
        // For social media method
        episodeTitle = 'Research for ' + formData.get('directGuestName');
        rssFeed = '';
        guestName = formData.get('directGuestName');
        linkedinUrl = formData.get('linkedinUrl');
        twitterUrl = formData.get('twitterUrl');
        
        // Validate required fields for social media method
        if (!guestName) {
            showError('Guest name is required for social media-based research.');
            return;
        }
        
        if (!linkedinUrl && !twitterUrl) {
            showError('At least one social media profile (LinkedIn or Twitter) is required.');
            return;
        }
    }
    
    // Log the data being sent for debugging
    console.log('Submitting research request with data:', {
        researchMethod,
        episodeTitle,
        rssFeed,
        guestName,
        hostPodcast,
        linkedinUrl,
        twitterUrl
    });
    
    // Show results section with loader
    resultsSection.style.display = 'block';
    showLoader(true);
    
    // Disable submit button
    toggleSubmitButton(false);
    
    try {
        // Create session for tracking research
        console.log('Creating research session...');
        const sessionResponse = await fetch(`${API_BASE_URL}/research-session`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                episode_title: episodeTitle,
                rss_feed: rssFeed || '',
                search_guest_name: guestName || '',
                host_podcast: hostPodcast || '',
                linkedin_url: linkedinUrl || '',
                twitter_url: twitterUrl || '',
                direct_social: researchMethod === 'social'
            })
        });
        
        if (!sessionResponse.ok) {
            const errorText = await sessionResponse.text();
            console.error('Session creation failed:', sessionResponse.status, errorText);
            throw new Error(`Failed to create research session: ${sessionResponse.status} ${errorText}`);
        }
        
        const sessionData = await sessionResponse.json();
        const sessionId = sessionData.session_id;
        console.log('Session created with ID:', sessionId);
        
        // Build the URL with query parameters
        let apiUrl = `${API_BASE_URL}/research-guest?episode_title=${encodeURIComponent(episodeTitle)}`;
        
        // Add optional parameters if provided
        if (rssFeed) apiUrl += `&rss_feed=${encodeURIComponent(rssFeed)}`;
        if (guestName) apiUrl += `&search_guest_name=${encodeURIComponent(guestName)}`;
        if (hostPodcast) apiUrl += `&host_podcast=${encodeURIComponent(hostPodcast)}`;
        if (linkedinUrl) apiUrl += `&linkedin_url=${encodeURIComponent(linkedinUrl)}`;
        if (twitterUrl) apiUrl += `&twitter_url=${encodeURIComponent(twitterUrl)}`;
        if (researchMethod === 'social') apiUrl += `&direct_social=true`;
        
        // Add the session_id to the URL
        apiUrl += `&session_id=${encodeURIComponent(sessionId)}`;
        
        console.log('Making research request to:', apiUrl);
        
        // Make API request with a longer timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5-minute timeout
        
        try {
            const response = await fetch(apiUrl, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            // Check if response is successful
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Research failed with status:', response.status, errorText);
                
                // Show more helpful error message based on status code
                if (response.status === 502) {
                    throw new Error('Could not connect to the research backend. Please ensure main.py is running on port 8080.');
                } else if (response.status === 504) {
                    throw new Error('The research request timed out. Please try again.');
                } else {
                    throw new Error(`Server responded with status: ${response.status} - ${errorText}`);
                }
            }
            
            // Parse JSON response
            const data = await response.json();
            console.log('Research completed successfully, data:', data);
            
            // If we have state data in string format, try to parse it
            if (data.state && typeof data.state === 'string') {
                try {
                    data.parsedState = JSON.parse(data.state);
                    console.log('Parsed state data:', data.parsedState);
                } catch (e) {
                    console.error('Failed to parse state data:', e);
                }
            }
            
            // If we have a document URL in the response, research completed successfully
            if (data.document_url) {
                console.log('Document URL found, research is complete:', data.document_url);
                // Display research results
                displayResults(data, sessionId);
            } else if (data.session_id) {
                // If we have a session ID but no document URL, poll for status
                console.log('No document URL yet, polling for status with session ID:', data.session_id);
                pollResearchStatus(data.session_id || sessionId);
            } else {
                // Something went wrong but we got a response
                console.error('Invalid API response format:', data);
                showError('Research completed but no results were returned. Please try again.');
            }
        } catch (fetchError) {
            if (fetchError.name === 'AbortError') {
                console.error('Request timed out after 5 minutes');
                throw new Error('Research request timed out after 5 minutes. Please try again.');
            }
            throw fetchError;
        }
    } catch (error) {
        console.error('Research request failed:', error);
        showError(`Research failed: ${error.message}`);
    } finally {
        // Hide loader and re-enable submit button (if displayResults hasn't done this already)
        toggleSubmitButton(true);
    }
}

// Show results in a tabbed interface
function displayResults(data) {
    console.log('Displaying results:', data);
    
    // Hide loader and show results container
    showLoader(false);
    toggleSubmitButton(true);
    resultsContainer.style.display = 'block';
    
    // Clear previous results if any
    while (resultsContainer.firstChild) {
        resultsContainer.removeChild(resultsContainer.firstChild);
    }
    
    // Set the guest name and LinkedIn link if available
    const headerDiv = document.createElement('div');
    headerDiv.className = 'results-header';
    
    const guestName = document.createElement('h2');
    guestName.textContent = data.guest || 'Unknown Guest';
    headerDiv.appendChild(guestName);
    
    if (data.linkedin) {
        const linkedinLink = document.createElement('a');
        linkedinLink.href = data.linkedin;
        linkedinLink.textContent = 'LinkedIn Profile';
        linkedinLink.target = '_blank';
        linkedinLink.rel = 'noopener noreferrer';
        headerDiv.appendChild(linkedinLink);
    }
    
    resultsContainer.appendChild(headerDiv);
    
    // Create tabs container
    const tabsContainer = document.createElement('div');
    tabsContainer.className = 'tabs-container';
    
    // Create tabs navigation
    const tabsNav = document.createElement('div');
    tabsNav.className = 'tabs-nav';
    
    const tabContent = document.createElement('div');
    tabContent.className = 'tab-content';
    
    // Define tabs and their content
    const tabs = [
        { id: 'intro', label: 'Intro', content: data.introduction || 'No introduction available.' },
        { id: 'summary', label: 'Summary', content: data.summary || 'No summary available.' },
        { id: 'questions', label: 'Questions', content: data.question || 'No questions available.' },
        { id: 'appearances', label: 'Previous Appearances', content: data.appearance || 'No previous appearances found.' }
    ];
    
    // Check if we have structured data or should fall back to iframe
    const hasStructuredData = data.introduction || data.summary || data.question || data.appearance;
    
    if (!hasStructuredData && data.document_url) {
        // Fallback to iframe if no structured data but we have a document URL
        console.log('No structured data available, falling back to iframe');
        
        const messageDiv = document.createElement('div');
        messageDiv.textContent = 'Displaying research results in Google Doc:';
        resultsContainer.appendChild(messageDiv);
        
        const iframe = document.createElement('iframe');
        iframe.src = data.document_url;
        iframe.width = '100%';
        iframe.height = '600px';
        resultsContainer.appendChild(iframe);
        return;
    }
    
    // Create tabs and tab panels
    tabs.forEach((tab, index) => {
        // Create tab button
        const tabButton = document.createElement('button');
        tabButton.textContent = tab.label;
        tabButton.className = 'tab-button';
        tabButton.dataset.tabId = tab.id;
        
        // First tab is active by default
        if (index === 0) {
            tabButton.classList.add('active');
        }
        
        tabsNav.appendChild(tabButton);
        
        // Create tab panel
        const tabPanel = document.createElement('div');
        tabPanel.className = 'tab-panel';
        tabPanel.id = `${tab.id}-panel`;
        tabPanel.style.display = index === 0 ? 'block' : 'none';
        
        // Convert Markdown to HTML and set content
        const htmlContent = convertMarkdownToHtml(tab.content);
        tabPanel.innerHTML = htmlContent;
        
        tabContent.appendChild(tabPanel);
    });
    
    // Add tabs navigation and content to container
    tabsContainer.appendChild(tabsNav);
    tabsContainer.appendChild(tabContent);
    resultsContainer.appendChild(tabsContainer);
    
    // Add event listeners to tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and hide all panels
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.style.display = 'none');
            
            // Add active class to clicked button and show corresponding panel
            button.classList.add('active');
            document.getElementById(`${button.dataset.tabId}-panel`).style.display = 'block';
        });
    });
    
    // If document URL is available, add a link to the Google Doc
    if (data.document_url) {
        const docLink = document.createElement('div');
        docLink.className = 'doc-link';
        
        const link = document.createElement('a');
        link.href = data.document_url;
        link.textContent = 'View complete research in Google Doc';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        
        docLink.appendChild(link);
        resultsContainer.appendChild(docLink);
    }
}

// Poll for research status
async function pollResearchStatus(sessionId) {
    console.log(`Polling for status of session: ${sessionId}`);
    
    // Poll every 5 seconds for up to 5 minutes (60 polls)
    const maxPolls = 60;
    let pollCount = 0;
    
    const pollInterval = setInterval(async () => {
        try {
            // Check if we've reached the maximum number of polls
            if (pollCount >= maxPolls) {
                clearInterval(pollInterval);
                showError('Research is taking longer than expected. Please check back later.');
                console.error('Maximum poll count reached');
                toggleSubmitButton(true);
                return;
            }
            
            pollCount++;
            console.log(`Poll attempt ${pollCount}/${maxPolls}`);
            
            // Make the status request
            const response = await fetch(`${API_BASE_URL}/research-status/${sessionId}`);
            
            if (!response.ok) {
                console.error(`Status check failed: ${response.status}`);
                return; // Continue polling
            }
            
            const data = await response.json();
            console.log('Status check response:', data);
            
            // Check if the research is complete
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                console.log('Research completed, displaying results');
                
                // Add the structured data to the results display
                const resultData = {
                    guest: data.guest,
                    linkedin: data.linkedin,
                    document_url: data.document_url,
                    introduction: data.introduction,
                    summary: data.summary,
                    question: data.question,
                    appearance: data.appearance
                };
                
                displayResults(resultData);
                return;
            }
            
            // If there's an error status, show it
            if (data.status === 'error') {
                clearInterval(pollInterval);
                showError(`Research failed: ${data.error || 'Unknown error'}`);
                toggleSubmitButton(true);
                return;
            }
            
            // Continue polling if not complete
            console.log(`Research status: ${data.status}, continuing to poll`);
            
        } catch (error) {
            console.error('Error checking status:', error);
            // Don't clear the interval, try again next time
        }
    }, 5000); // Poll every 5 seconds
}

// Show error message
function showError(message) {
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = '';
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
    
    resultsContainer.appendChild(errorDiv);
    resultsContainer.classList.add('visible');
}

// Show/hide loader
function showLoader(show) {
    console.log(`${show ? 'Showing' : 'Hiding'} loader`);
    
    // Always make the results section visible
    resultsSection.style.display = 'block';
    
    // Toggle loader and results container visibility
    loader.style.display = show ? 'flex' : 'none';
    resultsContainer.style.display = show ? 'none' : 'block';
    
    if (show) {
        // Clear any previous results when showing loader
        resultsContainer.innerHTML = '';
        resultsContainer.classList.remove('visible');
        
        // Scroll to the results section
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    } else {
        // Make sure results are visible when hiding the loader
        resultsContainer.classList.add('visible');
    }
}

// Toggle submit button state
function toggleSubmitButton(enabled) {
    submitBtn.disabled = !enabled;
    submitBtn.innerHTML = enabled 
        ? '<i class="fas fa-search"></i> Start Research' 
        : '<i class="fas fa-spinner fa-spin"></i> Processing...';
}

// Convert Markdown to HTML
function convertMarkdownToHtml(markdown) {
    if (!markdown) return '';
    
    let html = markdown;
    
    // Replace headers
    html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    // Replace bold and italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
    html = html.replace(/_(.*?)_/g, '<em>$1</em>');
    
    // Replace links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Replace bullet lists
    html = html.replace(/^\s*\* (.*$)/gm, '<li>$1</li>');
    html = html.replace(/^\s*- (.*$)/gm, '<li>$1</li>');
    html = html.replace(/^\s*\+ (.*$)/gm, '<li>$1</li>');
    
    // Wrap lists in <ul> tags
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    
    // Fix any nested lists (multiple <ul> tags)
    html = html.replace(/<\/ul>\s*<ul>/g, '');
    
    // Replace numbered lists
    html = html.replace(/^\s*(\d+)\. (.*$)/gm, '<li>$2</li>');
    
    // Wrap numbered lists in <ol> tags
    html = html.replace(/(<li>\d+\. .*<\/li>)/gs, '<ol>$1</ol>');
    html = html.replace(/<\/ol>\s*<ol>/g, '');
    
    // Replace code blocks
    html = html.replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>');
    
    // Replace inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Replace horizontal rules
    html = html.replace(/^---$/gm, '<hr>');
    
    // Replace paragraphs (double line breaks)
    html = html.replace(/\n\s*\n/g, '</p><p>');
    
    // Wrap with paragraph tags if not already wrapped
    if (!html.startsWith('<')) {
        html = '<p>' + html + '</p>';
    }
    
    return html;
} 