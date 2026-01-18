const PARAMETER_INFO = {
  T2M: { name: 'Air Temperature', unit: '°C' },
  PS: { name: 'Atm Pressure', unit: 'kPa' },
  QV2M: { name: 'Specific Humidity', unit: 'g/kg' },
  GWETTOP: { name: 'Top Soil Wetness', unit: '' },
  WS2M: { name: 'Wind Speed', unit: 'm/s' },
};

const ORBITAL_DATA = {
  mercury: { period: 88 },
  venus: { period: 224.7 },
  earth: { period: 365.25 },
  mars: { period: 687 },
  jupiter: { period: 4331 },
  saturn: { period: 10747 },
  uranus: { period: 30589 },
  neptune: { period: 59800 }
};

const PLANET_WEIGHTS = {
  mercury: 0.6,
  venus: 0.8,
  earth: 1.0,
  mars: 0.7,
  jupiter: 0.9,
  saturn: 0.5,
  uranus: 0.3,
  neptune: 0.2
};


const EPOCH = new Date("2000-01-01");
// Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
scene.background = new THREE.Color(0x000000);

camera.position.set(-15, 0, 50);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);


// Starry background
const starGeometry = new THREE.BufferGeometry();
const starCount = 10000;
const positions = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i++) {
  positions[i * 3] = (Math.random() - 0.5) * 2000;
  positions[i * 3 + 1] = (Math.random() - 0.5) * 2000;
  positions[i * 3 + 2] = (Math.random() - 0.5) * 2000;
}
starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const starMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.5 });
const stars = new THREE.Points(starGeometry, starMaterial);
scene.add(stars);

// Light
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);

const sunLight = new THREE.PointLight(0xffffff, 2);
scene.add(sunLight);

let isPlaying = false;

const playPauseBtn = document.getElementById("playPause");

function updatePlanetCards() {
  Object.entries(planets).forEach(([name, planet]) => {
    const angle = planet.userData.angle;
    const sinv = Math.sin(angle);
    const cosv = Math.cos(angle);
    const influence = PLANET_WEIGHTS[name] * sinv;

    planetCards[name].innerHTML = `
      <b>${name.toUpperCase()}</b><br>
      λ = ${angle.toFixed(2)} rad<br>
      sin(λ) = ${sinv.toFixed(3)}<br>
      cos(λ) = ${cosv.toFixed(3)}<br>
      Graha term = ${influence.toFixed(3)}
    `;
  });
}

// Texture loader
const loader = new THREE.TextureLoader();
function setPlanetPositionsFromDate(dateStr) {
  const selectedDate = new Date(dateStr);
  const daysSinceEpoch =
    (selectedDate - EPOCH) / (1000 * 60 * 60 * 24);

  Object.entries(planets).forEach(([name, planet]) => {
    const period = ORBITAL_DATA[name].period;
    planet.userData.angle =
      2 * Math.PI * (daysSinceEpoch / period);
  });
}


// Planet factory
function createPlanet(size, texture, distance, orbitSpeed, rotationSpeed, eccentricity) {
  const geometry = new THREE.SphereGeometry(size, 64, 64);
  const material = new THREE.MeshBasicMaterial({
    map: loader.load(`textures/${texture}`)
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData = { distance, orbitSpeed, rotationSpeed, angle: 0, eccentricity };
  scene.add(mesh);
  return mesh;
}
function positionPlanetCards() {
  Object.entries(planets).forEach(([name, planet]) => {
    const vector = planet.position.clone();
    vector.project(camera);

    const x = (vector.x * 0.5 + 0.5) * window.innerWidth;
    const y = (-vector.y * 0.5 + 0.5) * window.innerHeight;

    planetCards[name].style.left = `${x}px`;
    planetCards[name].style.top = `${y}px`;
  });
}

// Create planets
const planets = {
  mercury: createPlanet(0.4, "mercury.jpg", 5, 4.15, 0.01, 0.2056),
  venus:   createPlanet(0.7, "venus.jpg", 7, 1.62, 0.008, 0.0068),
  earth:   createPlanet(0.8, "earth.jpg", 9, 1.00, 0.02, 0.0167),
  mars:    createPlanet(0.6, "mars.jpg", 11, 0.53, 0.018, 0.0934),
  jupiter: createPlanet(2.0, "jupiter.jpg", 15, 0.084, 0.04, 0.0489),
  saturn:  createPlanet(1.7, "saturn.jpg", 19, 0.034, 0.038, 0.0565),
  uranus:  createPlanet(1.6, "uranus.jpg", 30, 0.012, 0.03, 0.0472),
  neptune: createPlanet(1.5, "neptune.jpg", 45, 0.006, 0.032, 0.0086)
};

const planetCards = {};

Object.keys(planets).forEach(name => {
  const div = document.createElement("div");
  div.className = "planet-card";
  div.style.display = "none";
  document.body.appendChild(div);
  planetCards[name] = div;
});

// Add orbits
Object.values(planets).forEach(p => {
  const a = p.userData.distance;
  const e = p.userData.eccentricity;
  const b = a * Math.sqrt(1 - e * e);
  const points = [];
  for (let i = 0; i <= 64; i++) {
    const angle = (i / 64) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(angle) * a, 0, Math.sin(angle) * b));
  }
  const orbitGeometry = new THREE.BufferGeometry().setFromPoints(points);
  const orbitMaterial = new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 1, gapSize: 1, linewidth: 0.001, transparent: true, opacity: 0.3 });
  const orbit = new THREE.Line(orbitGeometry, orbitMaterial);
  orbit.computeLineDistances();
  scene.add(orbit);
});

// Saturn rings
const ringGeo = new THREE.RingGeometry(2.2, 3.2, 64);
const ringMat = new THREE.MeshBasicMaterial({
  map: loader.load("textures/saturn_ring.png"),
  side: THREE.DoubleSide,
  transparent: true
});
const ring = new THREE.Mesh(ringGeo, ringMat);
ring.rotation.x = Math.PI / 2;
planets.saturn.add(ring);

// Sun
const sunGeo = new THREE.SphereGeometry(3, 64, 64);
const sunMat = new THREE.MeshBasicMaterial({ map: loader.load("textures/sun.jpg") });
const sun = new THREE.Mesh(sunGeo, sunMat);
scene.add(sun);

// Date picker for temperature
// const datePicker = document.getElementById("datePicker");
// datePicker.addEventListener("change", () => {
//   const dateStr = datePicker.value;
//   fetch(`http://127.0.0.1:8000/predict?date=${dateStr}`)
//     .then(response => response.json())
//     .then(data => {
//       document.getElementById("temperature").textContent = `Temperature: ${data.predicted_temperature_celsius}°C`;
//     })
//     .catch(err => {
//       console.error("Error fetching temperature:", err);
//       document.getElementById("temperature").textContent = "Temperature: Error";
//     });
// });
playPauseBtn.addEventListener("click", () => {
  isPlaying = !isPlaying;
  playPauseBtn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";

  if (isPlaying) {
    Object.values(planetCards).forEach(c => c.style.display = "none");
  }
});

const datePicker = document.getElementById("datePicker");

function fetchTemperature(dateStr) {
  const citySelect = document.getElementById("citySelect");
  const city = citySelect ? citySelect.value : "bengaluru";
  
  const endpoint = city === "bengaluru" 
    ? 'http://127.0.0.1:5000/predict-bangaluru'
    : 'http://127.0.0.1:5000/predict-delhi';

  fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ date: dateStr })
  })
    .then(res => {
      if (!res.ok) throw new Error("Bad response");
      return res.json();
    })
    .then(data => {
      const predictions = data.predictions;
      const cityName = city === "bengaluru" ? "Bengaluru" : "Delhi";
      
      // Store weather data for summarize function
      currentWeatherData = predictions;
      
      let weatherHTML = `<div style="margin-bottom: 0.8rem; font-size: 1.1rem; color: #00f2ff;">${cityName} Weather</div>`;
      
      Object.entries(predictions).forEach(([key, value]) => {
        const info = PARAMETER_INFO[key] || { name: key, unit: '' };
        const unit = info.unit ? ` ${info.unit}` : '';
        weatherHTML += `<span class="param-label">${info.name}:</span> ${value ?? '--'}${unit}<br>`;
      });
      
      document.getElementById("temperature").innerHTML = weatherHTML;
    })
    .catch(err => {
      console.error("Temperature fetch failed:", err);
      document.getElementById("temperature").textContent = "Weather: Error";
    });
}

// Initial load
fetchTemperature(datePicker.value);
setPlanetPositionsFromDate(datePicker.value);

// On date change
datePicker.addEventListener("change", () => {
  fetchTemperature(datePicker.value);
  setPlanetPositionsFromDate(datePicker.value);
  updatePlanetCards();
  Object.values(planetCards).forEach(c => c.style.display = "block");
});

// On city change
const citySelect = document.getElementById("citySelect");
if (citySelect) {
  citySelect.addEventListener("change", () => {
    fetchTemperature(datePicker.value);
  });
}

// Camera controls
let cameraAngle = 0;
let cameraTheta = Math.PI / 2;
const cameraDistance = 50;

document.getElementById("left").addEventListener("click", () => {
  cameraAngle -= 0.1;
});

document.getElementById("right").addEventListener("click", () => {
  cameraAngle += 0.1;
});

document.getElementById("up").addEventListener("click", () => {
  cameraTheta -= 0.1;
});

document.getElementById("down").addEventListener("click", () => {
  cameraTheta += 0.1;
});

// Time control
let timeScale = 1;
document.getElementById("slider").addEventListener("input", e => {
  timeScale = parseFloat(e.target.value);
});

// Weather data storage for potential reuse
let currentWeatherData = null;

// Summarize button handler (guarded because the button lives in React now)
const summarizeBtn = document.getElementById("summarizeBtn");
if (summarizeBtn) {
  summarizeBtn.addEventListener("click", async () => {
    const datePicker = document.getElementById("datePicker");
    const citySelect = document.getElementById("citySelect");
    const city = citySelect.value;
    const date = datePicker.value;
    
    summarizeBtn.disabled = true;
    summarizeBtn.textContent = "Generating...";
    
    try {
      // Collect current planet data
      const planetData = {};
      Object.entries(planets).forEach(([name, planet]) => {
        planetData[name] = {
          angle: planet.userData.angle,
          influence: PLANET_WEIGHTS[name] * Math.sin(planet.userData.angle)
        };
      });
      
      if (!currentWeatherData) {
        document.getElementById("summaryText").textContent = "Please fetch weather data first by changing the date.";
        document.getElementById("summaryModal").classList.add("active");
        summarizeBtn.disabled = false;
        summarizeBtn.textContent = "✨ Summarize";
        return;
      }
      
      const response = await fetch('http://127.0.0.1:5000/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          weather: currentWeatherData,
          planets: planetData,
          city: city,
          date: date
        })
      });
      
      if (!response.ok) throw new Error("Failed to summarize");
      const data = await response.json();
      
      document.getElementById("summaryText").textContent = data.summary;
      document.getElementById("summaryModal").classList.add("active");
    } catch (err) {
      console.error("Summarize failed:", err);
      document.getElementById("summaryText").textContent = "Error generating summary. Please try again.";
      document.getElementById("summaryModal").classList.add("active");
    } finally {
      summarizeBtn.disabled = false;
      summarizeBtn.textContent = "✨ Summarize";
    }
  });
}

// Close summary modal
document.getElementById("closeSummary").addEventListener("click", () => {
  document.getElementById("summaryModal").classList.remove("active");
});

// Close modal on background click
document.getElementById("summaryModal").addEventListener("click", (e) => {
  if (e.target.id === "summaryModal") {
    document.getElementById("summaryModal").classList.remove("active");
  }
});

// Animation loop
function animate() {
  requestAnimationFrame(animate);

  sun.rotation.y += 0.002;
  Object.values(planets).forEach(p => {
    if (isPlaying) {
      p.userData.angle += p.userData.orbitSpeed * 0.001 * timeScale;
    }
    const a = p.userData.distance;
    const e = p.userData.eccentricity;
    const b = a * Math.sqrt(1 - e * e);
    p.position.x = Math.cos(p.userData.angle) * a;
    p.position.z = Math.sin(p.userData.angle) * b;
    p.rotation.y += p.userData.rotationSpeed;
  });
    if (!isPlaying) {
      positionPlanetCards();
    }

  // Update camera position
  camera.position.x = cameraDistance * Math.sin(cameraTheta) * Math.sin(cameraAngle);
  camera.position.y = cameraDistance * Math.cos(cameraTheta);
  camera.position.z = cameraDistance * Math.sin(cameraTheta) * Math.cos(cameraAngle);
  camera.lookAt(0, 0, 0);

  renderer.render(scene, camera);
}

animate();

// Resize handling
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
