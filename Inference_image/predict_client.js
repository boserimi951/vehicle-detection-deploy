const formData = new FormData();
formData.append('file', imageFile);

fetch('http://13.62.189.78:8000/predict', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(detections => console.log(detections));