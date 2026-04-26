// ================================
// ELEMEN DOM
// ================================
const uploadGallery = document.getElementById("uploadGallery");
const openCamera = document.getElementById("openCamera");

const previewImage = document.getElementById("previewImage");
const resultBox = document.getElementById("resultBox");
const diseaseName = document.getElementById("diseaseName");
const confidenceText = document.getElementById("confidence");

// Simpan file yang dipilih
let selectedFile = null;

// ================================
// HANDLE FILE (GALERI / KAMERA)
// ================================
function handleFile(input) {
  const file = input.files[0];
  if (!file) return;

  // Validasi file harus gambar
  if (!file.type.startsWith("image/")) {
    alert("File harus berupa gambar!");
    input.value = "";
    return;
  }

  selectedFile = file;

  // Tampilkan preview
  previewImage.src = URL.createObjectURL(file);
  previewImage.style.display = "block";

  // Reset hasil sebelumnya
  resultBox.style.display = "none";
}

// ================================
// EVENT LISTENER
// ================================
if (uploadGallery) {
  uploadGallery.addEventListener("change", function () {
    handleFile(this);
  });
}

if (openCamera) {
  openCamera.addEventListener("change", function () {
    handleFile(this);
  });
}

// ================================
// PREDIKSI KE BACKEND
// ================================
async function predict() {
  if (!selectedFile) {
    alert("Silakan pilih gambar terlebih dahulu!");
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);

  // Tampilkan loading
  resultBox.style.display = "block";
  diseaseName.textContent = "⏳ Menganalisis gambar...";
  confidenceText.textContent = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    // Jika error dari server
    if (!response.ok) {
      diseaseName.textContent = "❌ Terjadi kesalahan";
      confidenceText.textContent = data.error || "";
      return;
    }

    // Validasi gambar bukan daun jagung
    if (data.valid === false) {
      diseaseName.textContent = "❌ Gambar tidak valid";
      confidenceText.textContent = data.message || "Bukan daun jagung";
      return;
    }

    // Tampilkan hasil
    diseaseName.textContent = `🌽 Penyakit: ${data.prediction}`;
    confidenceText.textContent = `Tingkat Kepercayaan: ${data.confidence}`;

  } catch (error) {
    diseaseName.textContent = "❌ Server tidak merespon";
    confidenceText.textContent = "";
    console.error("Error:", error);
  }
}