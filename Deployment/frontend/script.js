// UPDATE THIS URL AFTER DEPLOYING STEP 4
const API_URL = "http://127.0.0.1:8000/predict"; 

const fileInput = document.getElementById('audioInput');
const fileNameDisplay = document.getElementById('fileName');

fileInput.addEventListener('change', () => {
    if(fileInput.files.length > 0){
        fileNameDisplay.textContent = fileInput.files[0].name;
    }
});

async function analyzeAudio() {
    const file = fileInput.files[0];
    if (!file) { alert("Please select a file first!"); return; }

    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('resultCard').classList.add('hidden');

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        displayResult(data);
    } catch (error) {
        alert("Error connecting to server: " + error);
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
}

function displayResult(data) {
    const card = document.getElementById('resultCard');
    const title = document.getElementById('resultTitle');
    const conf = document.getElementById('confidenceScore');
    const action = document.getElementById('actionItem');

    card.classList.remove('hidden', 'safe', 'danger');

    if (data.prediction === "ANOMALY") {
        card.classList.add('danger');
        title.innerText = "🚨 THREAT DETECTED";
        action.innerText = "Action: Notify Security";
    } else {
        card.classList.add('safe');
        title.innerText = "✅ SECURE";
        action.innerText = "Action: Monitor Only";
    }
    conf.innerText = data.confidence + "%";
}