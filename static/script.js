//initialize the carousel
const carousel = new bootstrap.Carousel(document.getElementById('myCarousel'), {
    interval: false,
    ride: 'carousel',
});

async function submitResponse() {
    const userResponse = document.getElementById('userResponse').value;

    //send the user's response to the Flask backend
    const response = await fetch('/generate_feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ response: userResponse }),
    });

    const data = await response.json();

    //display the feedback
    document.getElementById('feedback').innerHTML = `
        <h3>Feedback:</h3>
        <p><strong>What AdvocAID AI Recommends To Add To Your Request:</strong> ${data.feedback}</p>
        <p><strong>Advice To Enhance Your Conversation:</strong> ${data.additional_feedback}</p>
        <p><strong>Your Confidence! </strong> ${data.confidence}</p>
        <p><strong>Your Self-Advocacy Level! </strong> ${data.self_advocacy}</p>
    `;
}

function retryResponse() {
    document.getElementById('userResponse').value = '';
    document.getElementById('feedback').innerHTML = '';
}
