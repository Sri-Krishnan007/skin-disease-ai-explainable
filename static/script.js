window.addEventListener("DOMContentLoaded", function () {
    const imageInput = document.getElementById('image-input');
    const previewSection = document.getElementById('preview-section');
    const previewImg = document.getElementById('preview-img');
    const confirmBtn = document.getElementById('confirm-btn');
    const changeBtn = document.getElementById('change-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const form = document.getElementById('upload-form');
    const loadingOverlay = document.getElementById('loading-overlay');

    if (imageInput) {
        imageInput.addEventListener('change', function () {

    const file = this.files[0];

    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("image/")) {
        alert("Please upload a valid image file.");
        this.value = "";
        return;
    }

    // Validate size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
        alert("Image must be smaller than 5MB.");
        this.value = "";
        return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
        previewImg.src = e.target.result;
        previewSection.classList.remove('hidden');
        analyzeBtn.classList.add('hidden');
    };

    reader.readAsDataURL(file);
});
    }

    if (confirmBtn) {
        confirmBtn.addEventListener('click', function () {

        if (loadingOverlay) {
        loadingOverlay.style.display = "flex";
    }

    confirmBtn.disabled = true;
    changeBtn.disabled = true;

    form.submit();
});
    }

    if (changeBtn) {
        changeBtn.addEventListener('click', function () {
            imageInput.value = "";
            previewSection.classList.add('hidden');
            analyzeBtn.classList.add('hidden');
            confirmBtn.disabled = false;
            changeBtn.disabled = false;
        });
    }

    if (form) {
        form.addEventListener('submit', function (e) {
            if (loadingOverlay) {
                 document.getElementById('loading-overlay').style.display = 'flex';
            }
        });
    }
});

function renderChart(labels, data) {

    const canvas = document.getElementById("chart");

    if (!canvas) {
        console.error("Canvas not found");
        return;
    }

    const ctx = canvas.getContext("2d");

    if (!labels || !data) {
        console.error("Data missing");
        return;
    }

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Probability (%)",
                data: data,
                backgroundColor: [
                    "#38bdf8","#22c55e","#facc15",
                    "#ef4444","#a855f7","#fb7185","#f97316"
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

