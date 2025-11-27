// ============ GLOBAL INITIALIZATION ============
document.addEventListener('DOMContentLoaded', function () {
    // Auto-dismiss alerts after 5 sec
    document.querySelectorAll('.alert').forEach(alert => {
        setTimeout(() => {
            new bootstrap.Alert(alert).close();
        }, 5000);
    });

    // Button loading animation on form submit
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function () {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<span class="loading"></span> Processing...';
                submitBtn.disabled = true;
            }
        });
    });
});


// ============ CAMERA UTILS ============
class CameraUtils {
    static async requestCameraPermission() {
        try {
            return await navigator.mediaDevices.getUserMedia({ video: true });
        } catch (err) {
            alert("❌ Camera access denied. Please allow permission.");
            throw err;
        }
    }

    static stopCamera(stream) {
        if (stream) stream.getTracks().forEach(t => t.stop());
    }

    static captureFrame(videoElement) {
        const canvas = document.createElement("canvas");
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        canvas.getContext("2d").drawImage(videoElement, 0, 0);
        return canvas.toDataURL("image/jpeg");
    }
}


// ============ API REQUEST UTILS ============
class ApiUtils {
    static async post(url, data) {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        return await res.json();
    }
    static error(e) {
        console.error(e);
        alert("⚠ Something went wrong.");
    }
}


// ============ FACE REGISTRATION ============
class FaceRegistration {
    constructor() {
        this.capturedCount = 0;
        this.totalRequired = 20;
        this.videoStream = null;
        this.status = document.getElementById("status-text");
        this.captureSound = new Audio("https://assets.mixkit.co/sfx/download/mixkit-unlock-new-item-game-notification-254.wav");
    }

    async startCamera() {
        try {
            this.videoStream = await CameraUtils.requestCameraPermission();
            document.getElementById("video").srcObject = this.videoStream;
        } catch (e) {}
    }

    async captureFace() {
        if (this.capturedCount >= this.totalRequired) return;

        document.body.classList.add("capturing");
        this.status.innerHTML = "📸 Capturing...";

        const video = document.getElementById("video");
        const imageData = CameraUtils.captureFrame(video);

        try {
            const result = await ApiUtils.post("/capture_face", { image: imageData });

            if (result.success) {
                this.captureSound.play();
                this.capturedCount++;

                if (this.capturedCount > this.totalRequired) this.capturedCount = this.totalRequired;
                this.updateUI();

                if (this.capturedCount === this.totalRequired) this.enableTraining();
            } else {
                alert(result.message);
            }
        } catch (e) {
            ApiUtils.error(e);
        } finally {
            document.body.classList.remove("capturing");
        }
    }

    updateUI() {
        const percent = (this.capturedCount / this.totalRequired) * 100;
        document.getElementById("progress").style.width = percent + "%";
        document.getElementById("capture-count").textContent = `${this.capturedCount}/${this.totalRequired}`;
        this.status.innerHTML = `📌 ${this.totalRequired - this.capturedCount} more captures required`;
    }

    enableTraining() {
        this.status.innerHTML = "🎉 All images captured — ready to train!";
        const btn = document.getElementById("train-btn");
        btn.disabled = false;
        btn.classList.add("btn-success");
        btn.classList.remove("btn-secondary");
    }

    async trainModel() {
        const btn = document.getElementById("train-btn");
        btn.innerHTML = "⏳ Training AI Model...";
        btn.disabled = true;
        this.status.innerHTML = "🤖 Training AI model — please wait...";

        try {
            const result = await ApiUtils.post("/train_model", {});
            if (result.success) {
                this.status.innerHTML = "✔ Model trained successfully!";
                setTimeout(() => {
                    CameraUtils.stopCamera(this.videoStream);
                    window.location.href = "/student/login";
                }, 1200);
            } else {
                alert(result.message);
            }
        } catch (e) {
            ApiUtils.error(e);
        } finally {
            btn.innerHTML = "Train Model";
        }
    }
}


// ============ INITIALIZE ON FACE REGISTRATION PAGE ============
if (window.location.pathname.includes("face_registration")) {
    const faceReg = new FaceRegistration();
    window.faceRegistration = faceReg;

    faceReg.startCamera();

    document.getElementById("capture-btn")?.addEventListener("click", () => faceReg.captureFace());
    document.getElementById("train-btn")?.addEventListener("click", () => faceReg.trainModel());
}
