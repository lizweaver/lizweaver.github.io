// ASCII Art Loader
// This script loads the ASCII art from ascii-art.txt and inserts it into any element with class 'ascii-art'

document.addEventListener('DOMContentLoaded', function() {
    const asciiElements = document.querySelectorAll('.ascii-art');
    
    if (asciiElements.length > 0) {
        // Determine the correct path to ascii-art.txt based on current page location
        const currentPath = window.location.pathname;
        const isInSubdirectory = currentPath.includes('/cs180/');
        const asciiPath = isInSubdirectory ? '../ascii-art.txt' : 'ascii-art.txt';
        
        fetch(asciiPath)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load ASCII art');
                }
                return response.text();
            })
            .then(asciiContent => {
                // Insert the ASCII content into all elements with class 'ascii-art'
                asciiElements.forEach(element => {
                    element.textContent = asciiContent;
                });
            })
            .catch(error => {
                console.warn('Could not load ASCII art:', error);
                // Optionally, you could provide fallback content here
            });
    }
});
