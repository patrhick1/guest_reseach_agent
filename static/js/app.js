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
    
    // Check quota on page load
    checkQuota();
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

// Check quota before submitting research
function checkQuota() {
    return fetch('/user-quota')
        .then(response => response.json())
        .then(data => {
            if (data.remaining <= 0) {
                // User has reached their quota limit
                document.getElementById('quota-limit-message').style.display = 'block';
                document.querySelector('#researchForm button').disabled = true;
                return false;
            }
            return true;
        })
        .catch(error => {
            console.error('Error checking quota:', error);
            return true; // Allow submission on error to let server-side check handle it
        });
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    // Check quota before submitting
    const hasQuota = await checkQuota();
    if (!hasQuota) {
        showToast('Daily research limit reached. Please try again tomorrow.');
        return false;
    }
    
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

// Display research results
function displayResults(data, sessionId) {
    // Hide loader
    showLoader(false);
    
    // Enable submit button
    toggleSubmitButton(true);
    
    // Clear previous results
    resultsContainer.innerHTML = '';
    
    // Create results HTML
    let resultsHTML = `
        <div class="results-header">
            <h3>Research Completed</h3>
        </div>
    `;
    
    // Guest information
    if (data.guest) {
        resultsHTML += `
            <div class="result-section">
                <h4>Guest Information</h4>
                <p><strong>Name:</strong> ${data.guest}</p>
            </div>
        `;
    }
    
    // LinkedIn profile
    if (data.linkedin) {
        resultsHTML += `
            <div class="result-section">
                <h4>Social Profiles</h4>
                <p>
                    <a href="${data.linkedin}" target="_blank" class="social-link linkedin">
                        <i class="fab fa-linkedin"></i> LinkedIn Profile
                    </a>
                </p>
            </div>
        `;
    }
    
    // Document URL
    if (data.document_url) {
        // Extract document ID from Google Docs URL for direct download as PDF
        let docId = '';
        try {
            const urlMatch = data.document_url.match(/\/d\/([^\/]+)/);
            if (urlMatch && urlMatch[1]) {
                docId = urlMatch[1];
            }
        } catch (e) {
            console.error('Failed to extract document ID:', e);
        }
        
        let downloadUrl = '';
        if (docId) {
            downloadUrl = `https://docs.google.com/document/d/${docId}/export?format=pdf`;
        }
        
        resultsHTML += `
            <div class="result-section">
                <h4>Research Document</h4>
                <div class="document-links">
                    <a href="${data.document_url}" target="_blank" class="button-link document-button">
                        <i class="fas fa-file-alt"></i> View Document Online
                    </a>
                    ${downloadUrl ? `
                    <a href="${downloadUrl}" target="_blank" class="button-link download-button">
                        <i class="fas fa-download"></i> Download as PDF
                    </a>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    // Show any introduction or summary if available
    if (data.introduction) {
        resultsHTML += `
            <div class="result-section">
                <h4>Introduction</h4>
                <div class="markdown-content">${convertMarkdownToHtml(data.introduction)}</div>
            </div>
        `;
    }
    
    if (data.summary) {
        resultsHTML += `
            <div class="result-section">
                <h4>Summary</h4>
                <div class="markdown-content">${convertMarkdownToHtml(data.summary)}</div>
            </div>
        `;
    }
    
    if (data.question) {
        resultsHTML += `
            <div class="result-section">
                <h4>Question Suggestions</h4>
                <div class="markdown-content">${convertMarkdownToHtml(data.question)}</div>
            </div>
        `;
    }
    
    if (data.appearance) {
        // Check if the appearance data is already HTML (containing podcast-grid)
        if (data.appearance.includes('podcast-grid')) {
            resultsHTML += `
                <div class="result-section">
                    <h4>Previous Appearances</h4>
                    <div>${data.appearance}</div>
                </div>
            `;
        } else {
            // Check if it contains our special markdown marker
            let appearanceContent = data.appearance;
            if (appearanceContent.includes('<!-- MARKDOWN_FORMAT -->')) {
                appearanceContent = appearanceContent.replace('<!-- MARKDOWN_FORMAT -->', '');
            }
            
            resultsHTML += `
                <div class="result-section">
                    <h4>Previous Appearances</h4>
                    <div class="markdown-content">${convertMarkdownToHtml(appearanceContent)}</div>
                </div>
            `;
        }
    }
    
    // Add dashboard link
    resultsHTML += `
        <div class="result-section actions">
            <a href="${API_BASE_URL}/dashboard" class="button-link dashboard-button">
                <i class="fas fa-tachometer-alt"></i> Go to Dashboard
            </a>
        </div>
    `;
    
    // Insert results into page
    resultsContainer.innerHTML = resultsHTML;
    resultsContainer.style.display = 'block';
    
    // Add custom styling for the results
    const style = document.createElement('style');
    style.textContent = `
        .results-header {
            margin-bottom: 2rem;
            text-align: center;
        }
        .result-section {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #eee;
        }
        .result-section:last-child {
            border-bottom: none;
        }
        .social-link {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            text-decoration: none;
            color: #0077b5;
            font-weight: 500;
        }
        .button-link {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            text-decoration: none;
            color: white;
            font-weight: 500;
            margin: 0.5rem 0.5rem 0.5rem 0;
        }
        .document-button {
            background-color: #4285F4;
        }
        .download-button {
            background-color: #34A853;
        }
        .dashboard-button {
            background-color: var(--primary-color);
        }
        .document-links {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .actions {
            margin-top: 2rem;
            text-align: center;
        }
        
        /* Podcast appearances styling */
        .podcast-grid {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-start;
            gap: 20px;
            margin-top: 20px;
        }
        
        .podcast-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 150px;
            margin-bottom: 20px;
            text-align: center;
            transition: transform 0.2s ease;
            text-decoration: none;
        }
        
        .podcast-item:hover {
            transform: scale(1.05);
        }
        
        .podcast-item img {
            width: 130px;
            height: 130px;
            border-radius: 12px;
            object-fit: cover;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            margin-bottom: 10px;
        }
        
        .podcast-item a {
            text-decoration: none;
            color: var(--primary-color);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .podcast-item p {
            font-size: 0.85rem;
            margin: 5px 0;
            line-height: 1.3;
            color: var(--text-color);
            font-weight: 500;
            max-width: 140px;
        }
        
        /* Regular markdown image styling */
        .markdown-content img {
            max-width: 130px;
            max-height: 130px;
            border-radius: 10px;
            object-fit: cover;
            margin: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        
        .markdown-content a img {
            width: 130px;
            height: 130px;
            border-radius: 10px;
            object-fit: cover;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            margin: 10px;
            transition: transform 0.2s ease;
        }
        
        .markdown-content a img:hover {
            transform: scale(1.05);
        }
    `;
    document.head.appendChild(style);
}

// Poll for research status
async function pollResearchStatus(sessionId) {
    console.log('Polling for research status with session ID:', sessionId);
    
    if (!sessionId) {
        console.error('No session ID provided for polling');
        showError('Unable to track research progress. Please try again.');
        return;
    }
    
    // Keep track of polling attempts
    let attempts = 0;
    const maxAttempts = 30; // 5 minutes (10 seconds * 30)
    
    // Update polling message in loader
    const loaderMessage = document.querySelector('#loader p');
    if (loaderMessage) {
        loaderMessage.innerHTML = 'Researching guest information...<br><small>This may take a few minutes</small>';
    }
    
    // Start polling
    const pollInterval = setInterval(async () => {
        attempts++;
        
        try {
            console.log(`Polling attempt ${attempts}/${maxAttempts}`);
            
            // Make API request to check status
            const response = await fetch(`${API_BASE_URL}/research-status/${sessionId}`);
            
            if (!response.ok) {
                // Handle server errors
                const errorText = await response.text();
                console.error('Status polling failed:', response.status, errorText);
                
                // If we've reached max attempts or got a 404, stop polling
                if (attempts >= maxAttempts || response.status === 404) {
                    clearInterval(pollInterval);
                    showError('Research is taking longer than expected. Please check the dashboard later for results.');
                }
                return;
            }
            
            // Parse JSON response
            const data = await response.json();
            console.log('Poll response:', data);
            
            // Check if research is complete
            if (data.status === 'completed' && data.document_url) {
                console.log('Research completed:', data);
                clearInterval(pollInterval);
                displayResults(data, sessionId);
            } else if (data.status === 'failed') {
                console.error('Research failed:', data.error || 'Unknown error');
                clearInterval(pollInterval);
                showError(data.error || 'Research failed. Please try again.');
            } else {
                // Update progress message if provided
                if (data.progress_message && loaderMessage) {
                    loaderMessage.innerHTML = `${data.progress_message}<br><small>Please wait...</small>`;
                }
                
                // Stop polling if we've reached max attempts
                if (attempts >= maxAttempts) {
                    clearInterval(pollInterval);
                    console.warn('Reached maximum polling attempts');
                    
                    // Show a message that we'll continue in the background
                    showError('Research is taking longer than expected. It will continue in the background. Please check the dashboard later for results.');
                }
            }
        } catch (error) {
            console.error('Error polling for status:', error);
            
            // Only show error and stop polling if we've reached max attempts
            if (attempts >= maxAttempts) {
                clearInterval(pollInterval);
                showError('Lost connection to server. Research will continue in the background. Please check the dashboard later.');
            }
        }
    }, 10000); // Poll every 10 seconds
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
    
    // Preserve line breaks by replacing them with <br> tags temporarily 
    // (before other processing, we'll handle lists specially)
    html = html.replace(/\n/g, '{{linebreak}}');
    
    // Replace headers
    html = html.replace(/^### (.*?){{linebreak}}/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*?){{linebreak}}/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*?){{linebreak}}/gm, '<h1>$1</h1>');
    
    // Replace bold and italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');
    html = html.replace(/_(.*?)_/g, '<em>$1</em>');
    
    // Replace links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Handle bullet lists more carefully (using temporary markers to preserve line breaks)
    html = html.replace(/{{linebreak}}(\s*)\* (.*?)(?={{linebreak}}|$)/gm, '{{listbreak}}<li>$2</li>');
    html = html.replace(/{{linebreak}}(\s*)- (.*?)(?={{linebreak}}|$)/gm, '{{listbreak}}<li>$2</li>');
    html = html.replace(/{{linebreak}}(\s*)\+ (.*?)(?={{linebreak}}|$)/gm, '{{listbreak}}<li>$2</li>');
    
    // Wrap lists in <ul> tags and clean up list markers
    if (html.includes('{{listbreak}}<li>')) {
        html = html.replace(/{{listbreak}}/g, '');
        // Find sequences of <li> elements and wrap them in <ul> tags
        html = html.replace(/(<li>.*?<\/li>)+/gs, '<ul>$&</ul>');
    }
    
    // Fix any nested lists (multiple <ul> tags)
    html = html.replace(/<\/ul>\s*<ul>/g, '');
    
    // Replace numbered lists (similar careful approach)
    html = html.replace(/{{linebreak}}(\s*)\d+\. (.*?)(?={{linebreak}}|$)/gm, '{{numlistbreak}}<li>$2</li>');
    
    // Wrap numbered lists in <ol> tags
    if (html.includes('{{numlistbreak}}<li>')) {
        html = html.replace(/{{numlistbreak}}/g, '');
        // Find sequences of these list items and wrap them in <ol> tags
        html = html.replace(/(<li>.*?<\/li>)+/gs, '<ol>$&</ol>');
        // Make sure we don't double-wrap things that are already in <ul> tags
        html = html.replace(/<ol><ul>(.*?)<\/ul><\/ol>/g, '<ul>$1</ul>');
    }
    
    // Replace code blocks
    html = html.replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>');
    
    // Replace inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Replace horizontal rules
    html = html.replace(/{{linebreak}}---{{linebreak}}/g, '<hr>');
    
    // Handle paragraphs - groups of text separated by empty lines
    html = html.replace(/{{linebreak}}{{linebreak}}+/g, '</p><p>');
    
    // Put back normal line breaks (if they're not already part of a list or heading)
    html = html.replace(/{{linebreak}}/g, '<br>');
    
    // Wrap with paragraph tags if not already wrapped
    if (!html.startsWith('<')) {
        html = '<p>' + html + '</p>';
    }
    
    // Fix any potential issues with paragraph tags inside list items
    html = html.replace(/<li><p>(.*?)<\/p><\/li>/g, '<li>$1</li>');
    
    return html;
}

// Handle API errors related to quota
function handleApiError(error) {
    if (error.quota_exceeded) {
        document.getElementById('quota-limit-message').style.display = 'block';
        document.querySelector('#researchForm button').disabled = true;
        showToast(`Daily research limit reached (${error.current_count}/${error.max_count}). Please try again tomorrow.`);
    } else {
        showToast(error.message || 'An error occurred. Please try again.');
    }
}

// Start the research process
async function startResearch(event) {
    event.preventDefault();
    
    // Check if user has reached their quota limit
    const quotaOk = await checkQuota();
    if (!quotaOk) {
        return; // Don't proceed if quota limit reached
    }
    
    toggleSubmitButton(false);
    
    // Get selected research method
    const isDirectSocial = document.querySelector('input[name="researchMethod"]:checked').value === 'social';
    
    // Collect data based on the selected research method
    let data = {
        host_podcast: document.getElementById('hostPodcast').value.trim(),
        direct_social: isDirectSocial
    };
    
    // Add method-specific fields
    if (isDirectSocial) {
        // For social media research
        data.episode_title = `Research for ${document.getElementById('directGuestName').value.trim()}`;
        data.search_guest_name = document.getElementById('directGuestName').value.trim();
        data.linkedin_url = document.getElementById('linkedinUrl').value.trim();
        data.twitter_url = document.getElementById('twitterUrl').value.trim();
    } else {
        // For podcast-based research
        data.episode_title = document.getElementById('episodeTitle').value.trim();
        data.rss_feed = document.getElementById('rssFeed').value.trim();
        data.search_guest_name = document.getElementById('guestName').value.trim();
    }
    
    console.log('Starting research with data:', data);
    
    try {
        // First, create a research session to track the UI state
        // This will check quota but NOT decrement it - quota is only decremented
        // when the actual research starts in the research-guest endpoint
        const sessionResponse = await fetch(`${API_BASE_URL}/research-session`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
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
        let apiUrl = `${API_BASE_URL}/research-guest?episode_title=${encodeURIComponent(data.episode_title)}`;
        
        // Add optional parameters if provided
        if (data.rss_feed) apiUrl += `&rss_feed=${encodeURIComponent(data.rss_feed)}`;
        if (data.search_guest_name) apiUrl += `&search_guest_name=${encodeURIComponent(data.search_guest_name)}`;
        if (data.host_podcast) apiUrl += `&host_podcast=${encodeURIComponent(data.host_podcast)}`;
        if (data.linkedin_url) apiUrl += `&linkedin_url=${encodeURIComponent(data.linkedin_url)}`;
        if (data.twitter_url) apiUrl += `&twitter_url=${encodeURIComponent(data.twitter_url)}`;
        if (data.direct_social) apiUrl += `&direct_social=true`;
        
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