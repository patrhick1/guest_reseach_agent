document.addEventListener('DOMContentLoaded', function() {
    // Fetch user quota information
    fetchQuotaInfo();
    
    // Set up polling to update quota info every minute
    setInterval(fetchQuotaInfo, 60000);
});

function fetchQuotaInfo() {
    fetch('/user-quota')
        .then(response => response.json())
        .then(data => {
            updateQuotaDisplay(data);
        })
        .catch(error => {
            console.error('Error fetching quota information:', error);
            document.getElementById('quota-display').innerHTML = 
                '<p><i class="fas fa-exclamation-circle"></i> Unable to load quota information.</p>';
        });
}

function updateQuotaDisplay(quotaData) {
    const quotaDisplay = document.getElementById('quota-display');
    
    // Format the reset time
    const resetTime = new Date(quotaData.reset_at);
    const formattedReset = resetTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    // Set color based on remaining quota
    let statusColor = '#4CAF50'; // Green
    let statusIcon = 'fa-check-circle';
    
    if (quotaData.remaining === 0) {
        statusColor = '#F44336'; // Red
        statusIcon = 'fa-times-circle';
    } else if (quotaData.remaining <= 2) {
        statusColor = '#FF9800'; // Orange
        statusIcon = 'fa-exclamation-circle';
    }
    
    quotaDisplay.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
            <div>
                <span style="font-size: 1.2rem; font-weight: bold; color: ${statusColor};">
                    <i class="fas ${statusIcon}"></i> 
                    ${quotaData.remaining} remaining today
                </span>
            </div>
            <div>
                <span style="color: #666;">Resets at ${formattedReset}</span>
            </div>
        </div>
        <div class="quota-progress" style="height: 10px; background-color: #e0e0e0; border-radius: 5px; overflow: hidden;">
            <div style="height: 100%; width: ${(quotaData.used_today / quotaData.daily_limit) * 100}%; 
                      background-color: ${statusColor}; transition: width 0.5s ease-in-out;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.9rem; color: #666;">
            <span>0</span>
            <span>${quotaData.daily_limit} research limit</span>
        </div>
    `;
} 