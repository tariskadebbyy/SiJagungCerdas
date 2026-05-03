/**
 * ==========================================
 * INIT (AMAN UNTUK SEMUA HALAMAN)
 * ==========================================
 */
document.addEventListener("DOMContentLoaded", function () {

  /**
   * ==========================================
   * LOGIKA NAVIGASI (HAMBURGER MENU)
   * ==========================================
   */
  const menuToggle = document.getElementById("mobile-menu");
  const navMenu = document.querySelector(".nav-menu");

  // Pastikan elemen ada (biar tidak error di halaman lain)
  if (menuToggle && navMenu) {
    menuToggle.addEventListener("click", () => {
      navMenu.classList.toggle("active");
    });
  }

  /**
   * ==========================================
   * PENGATURAN ELEMEN DOM (INPUT & PREVIEW)
   * ==========================================
   */
  const uploadGallery = document.getElementById("uploadGallery");
  const openCamera = document.getElementById("openCamera");
  const previewImage = document.getElementById("previewImage");
  const resultBox = document.getElementById("resultBox");
  const diseaseName = document.getElementById("diseaseName");
  const confidenceText = document.getElementById("confidence");

  let selectedFile = null;

  // Fungsi untuk memproses file gambar
  function handleFile(input) {
    const file = input.files[0];
    if (!file) return;

    // Validasi harus gambar
    if (!file.type.startsWith("image/")) {
      alert("File harus berupa gambar!");
      input.value = "";
      return;
    }

    selectedFile = file;

    // Preview gambar
    if (previewImage) {
      previewImage.src = URL.createObjectURL(file);
      previewImage.style.display = "block";
    }

    // Sembunyikan hasil lama
    if (resultBox) resultBox.style.display = "none";
  }

  // Event upload
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

  /**
   * ==========================================
   * LOGIKA PREDIKSI KE BACKEND
   * ==========================================
   */
  window.predict = async function () {

    if (!selectedFile) {
      alert("Silakan pilih gambar terlebih dahulu!");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    // Loading state
    if (resultBox) resultBox.style.display = "block";
    if (diseaseName) diseaseName.textContent = "⏳ Menganalisis gambar...";
    if (confidenceText) confidenceText.textContent = "";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        if (diseaseName) diseaseName.textContent = "❌ Terjadi kesalahan";
        if (confidenceText) confidenceText.textContent = data.error || "Gagal menghubungi server";
        return;
      }

      if (data.valid === false) {
        if (diseaseName) diseaseName.textContent = "❌ Gambar tidak valid";
        if (confidenceText) confidenceText.textContent = data.message || "Bukan daun jagung";
        return;
      }

      // Hasil sukses
      if (diseaseName) {
        diseaseName.textContent = `🌽 Penyakit: ${data.prediction}`;
      }

      if (confidenceText) {
        confidenceText.textContent = `Tingkat Kepercayaan: ${data.confidence}`;
      }

    } catch (error) {
      if (diseaseName) diseaseName.textContent = "❌ Server tidak merespon";
      if (confidenceText) confidenceText.textContent = "Periksa koneksi Anda.";
      console.error("Error:", error);
    }
  };

});
